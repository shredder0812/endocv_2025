"""
Visualize matching comparison side-by-side for 3 trackers.
Creates synchronized video output showing all 3 trackers simultaneously.

Usage:
    python visualize_matching_comparison.py --video_path video_test_x/UTTQ/230411BVK004_Trim2.mp4 --model_weights model_yolo/thucquan.pt
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from boxmot import StrongSort, StrongSortXYSR, StrongSortTLUKF


class SideBySideComparison:
    """Create side-by-side comparison video."""
    
    def __init__(self, video_path, model_weights, output_path):
        self.video_path = Path(video_path)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_weights)
        self.model.fuse()
        
        # Open video
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize trackers
        reid_weights = Path("osnet_dcn_x0_5_endocv.pt")
        
        self.trackers = {
            'StrongSort\n(XYAH)': StrongSort(
                reid_weights, torch.device(self.device), fp16=False,
                max_dist=0.95, max_iou_dist=0.95, max_age=300, half=False
            ),
            'StrongSort XYSR\n(+Virtual)': StrongSortXYSR(
                reid_weights, torch.device(self.device), fp16=False,
                max_dist=0.95, max_iou_dist=0.95, max_age=300, half=False
            ),
            'TLUKF\n(TL+Virtual)': StrongSortTLUKF(
                reid_weights=reid_weights, device=torch.device(self.device),
                half=False, max_iou_dist=0.95, max_age=300, n_init=3,
                mc_lambda=0.995, ema_alpha=0.9
            )
        }
        
        # Output video (3 frames side by side)
        output_width = self.width * 3
        output_height = self.height + 100  # Extra space for labels
        
        self.writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps,
            (output_width, output_height)
        )
        
        # Statistics
        self.stats = {name: {'id_count': 0, 'virtual_count': 0} for name in self.trackers.keys()}
        
        print(f"📹 Input: {video_path}")
        print(f"💾 Output: {output_path}")
        print(f"📊 Resolution: {output_width}x{output_height} @ {self.fps:.2f} FPS")
    
    def _detect(self, frame, tracker_name):
        """Run detection with appropriate threshold."""
        if 'TLUKF' in tracker_name:
            conf = 0.3
        elif 'XYSR' in tracker_name:
            conf = 0.45
        else:
            conf = 0.6
        
        results = self.model(frame, stream=True, verbose=False, conf=conf, line_width=1)
        return results
    
    def _draw_tracks(self, frame, tracks, tracker_name):
        """Draw tracks on frame with color coding."""
        frame_copy = frame.copy()
        
        for track in tracks:
            x1, y1, x2, y2 = map(int, track[:4])
            id_ = int(track[4])
            conf = track[5]
            
            # Color coding based on confidence
            if 'TLUKF' in tracker_name:
                if conf >= 0.6:
                    color = (0, 255, 0)    # Green - Strong
                    label_prefix = "S"
                elif conf >= 0.35:
                    color = (255, 165, 0)  # Orange - Weak
                    label_prefix = "W"
                else:
                    color = (128, 128, 128)  # Gray - Virtual
                    label_prefix = "V"
                    self.stats[tracker_name]['virtual_count'] += 1
            elif 'XYSR' in tracker_name:
                if conf >= 0.45:
                    color = (0, 255, 0)    # Green - Real
                    label_prefix = "R"
                else:
                    color = (128, 128, 128)  # Gray - Virtual
                    label_prefix = "V"
                    self.stats[tracker_name]['virtual_count'] += 1
            else:
                color = (0, 255, 0)        # Green - Real
                label_prefix = "R"
            
            # Draw box
            thickness = 2 if label_prefix == "V" else 3
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{label_prefix}-{id_}"
            font_scale = 0.6
            font_thickness = 2
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            cv2.rectangle(frame_copy, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
            cv2.putText(frame_copy, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale, (255, 255, 255), font_thickness)
        
        return frame_copy
    
    def _create_combined_frame(self, frames, frame_id):
        """Combine 3 frames side by side with labels."""
        # Stack frames horizontally
        combined = np.hstack(frames)
        
        # Add top bar with labels and stats
        top_bar = np.ones((100, combined.shape[1], 3), dtype=np.uint8) * 240
        
        # Draw labels and stats for each tracker
        x_positions = [self.width // 2, self.width + self.width // 2, 2 * self.width + self.width // 2]
        
        for idx, (name, x_pos) in enumerate(zip(self.trackers.keys(), x_positions)):
            # Tracker name
            lines = name.split('\n')
            y_pos = 30
            for line in lines:
                (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.putText(top_bar, line, (x_pos - w // 2, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                y_pos += 25
            
            # Stats
            stats_text = f"Virtual: {self.stats[name]['virtual_count']}"
            (w, h), _ = cv2.getTextSize(stats_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(top_bar, stats_text, (x_pos - w // 2, y_pos + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        # Add frame number
        frame_text = f"Frame: {frame_id}"
        cv2.putText(top_bar, frame_text, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Combine top bar and frames
        output = np.vstack([top_bar, combined])
        
        return output
    
    def run(self, max_frames=None):
        """Run comparison."""
        frame_id = 0
        
        print(f"\n🚀 Starting comparison...")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frames_with_tracks = []
            
            # Process each tracker
            for tracker_name, tracker in self.trackers.items():
                # Reset virtual count for this frame
                self.stats[tracker_name]['virtual_count'] = 0
                
                # Detect
                detections_list = self._detect(frame, tracker_name)
                
                for dets in detections_list:
                    det_boxes = dets.boxes.data.to("cpu").numpy()
                    
                    # Update tracker
                    if det_boxes.size > 0:
                        tracks = tracker.update(det_boxes, frame)
                    else:
                        tracks = tracker.update(np.empty((0, 6), dtype=np.float32), frame)
                    
                    # Draw tracks
                    frame_with_tracks = self._draw_tracks(frame, tracks, tracker_name)
                    frames_with_tracks.append(frame_with_tracks)
            
            # Create combined frame
            combined_frame = self._create_combined_frame(frames_with_tracks, frame_id)
            
            # Write to output
            self.writer.write(combined_frame)
            
            # Progress
            if frame_id % 30 == 0:
                print(f"Frame {frame_id}: ", end="")
                for name in self.trackers.keys():
                    print(f"{name.split()[0]}: {self.stats[name]['virtual_count']}v ", end="")
                print()
            
            frame_id += 1
            
            if max_frames and frame_id >= max_frames:
                break
        
        self.cap.release()
        self.writer.release()
        
        print(f"\n✅ Comparison video created!")
        print(f"📊 Total frames: {frame_id}")
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'cap'):
            self.cap.release()
        if hasattr(self, 'writer'):
            self.writer.release()


def main():
    parser = argparse.ArgumentParser(description="Create side-by-side comparison video")
    parser.add_argument("--video_path", type=str, default='video_test_x/UTTQ/230411BVK106_Trim2.mp4', help="Path to video file")
    parser.add_argument("--model_weights", type=str, default='model_yolo/thucquan.pt', help="Path to YOLO model")
    parser.add_argument("--output_path", type=str, default="comparison_output.mp4", help="Output video path")
    parser.add_argument("--max_frames", type=int, default=600, help="Max frames to process")
    
    args = parser.parse_args()
    
    comparator = SideBySideComparison(args.video_path, args.model_weights, args.output_path)
    comparator.run(max_frames=args.max_frames)
    comparator.close()


if __name__ == "__main__":
    main()
