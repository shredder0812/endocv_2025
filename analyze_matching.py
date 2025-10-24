"""
Công cụ phân tích matching ID cho 3 pipeline:
1. StrongSort (XYAH KF - baseline)
2. StrongSortXYSR (XYSR KF với virtual boxes)
3. StrongSortTLUKF (Transfer Learning UKF)

Usage:
    python analyze_matching.py --video_path video_test_x/UTTQ/230411BVK004_Trim2.mp4 --model_weights model_yolo/thucquan.pt
"""

import argparse
import json
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from boxmot import StrongSort, StrongSortXYSR, StrongSortTLUKF
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict

class MatchingAnalyzer:
    """Phân tích chi tiết quá trình matching ID của tracker."""
    
    def __init__(self, tracker_type, model_weights, video_path, output_dir):
        self.tracker_type = tracker_type
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = self._load_model(model_weights)
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir) / self.tracker_type
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Tracking statistics
        self.frame_stats = []
        self.id_history = defaultdict(list)  # id -> [(frame, bbox, conf, status)]
        self.matching_events = []  # Các sự kiện matching quan trọng
        
        # Initialize tracker
        self.tracker = self._initialize_tracker()
        
        # Open video
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"🎬 Video: {video_path}")
        print(f"📊 Resolution: {self.width}x{self.height} @ {self.fps:.2f} FPS")
        print(f"🎯 Tracker: {self.tracker_type}")
        print(f"💾 Output: {self.output_dir}")
        
    def _load_model(self, weights):
        """Load YOLO model."""
        model = YOLO(weights)
        model.fuse()
        return model
    
    def _initialize_tracker(self):
        """Initialize tracker based on type."""
        reid_weights = Path("osnet_dcn_x0_5_endocv.pt")
        
        if self.tracker_type == "strongsort":
            # Baseline StrongSort (XYAH)
            tracker = StrongSort(
                reid_weights,
                torch.device(self.device),
                fp16=False,
                max_dist=0.95,
                max_iou_dist=0.95,
                max_age=300,
                half=False,
            )
            print("✅ Initialized StrongSort (XYAH Kalman Filter)")
            
        elif self.tracker_type == "strongsort_xysr":
            # StrongSortXYSR with virtual boxes
            tracker = StrongSortXYSR(
                reid_weights,
                torch.device(self.device),
                fp16=False,
                max_dist=0.95,
                max_iou_dist=0.95,
                max_age=300,
                half=False,
            )
            print("✅ Initialized StrongSortXYSR (XYSR Kalman Filter + Virtual Boxes)")
            
        elif self.tracker_type == "tlukf":
            # TLUKF with dual-tracker
            tracker = StrongSortTLUKF(
                reid_weights=reid_weights,
                device=torch.device(self.device),
                half=False,
                max_iou_dist=0.95,
                max_age=300,
                n_init=3,
                mc_lambda=0.995,
                ema_alpha=0.9,
            )
            print("✅ Initialized TLUKF (Transfer Learning UKF + Enhanced Matching)")
            
        else:
            raise ValueError(f"Unknown tracker type: {self.tracker_type}")
        
        return tracker
    
    def _detect(self, frame):
        """Run detection with appropriate confidence threshold."""
        if self.tracker_type == "tlukf":
            # TLUKF uses lower conf to get weak detections
            conf_thresh = 0.3
        else:
            # Others use higher conf
            conf_thresh = 0.6 if self.tracker_type == "strongsort" else 0.6
        
        results = self.model(frame, stream=True, verbose=False, conf=conf_thresh, line_width=1)
        return results
    
    def _analyze_frame(self, frame_id, detections, tracks):
        """Analyze matching results for current frame."""
        frame_info = {
            'frame_id': frame_id,
            'num_detections': len(detections) if len(detections) > 0 else 0,
            'num_tracks': len(tracks) if len(tracks) > 0 else 0,
            'detections': [],
            'tracks': [],
            'matching_info': {}
        }
        
        # Record detections
        if len(detections) > 0:
            for det in detections:
                x1, y1, x2, y2, conf, cls = det[:6]
                frame_info['detections'].append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(conf),
                    'class': int(cls)
                })
        
        # Record tracks
        if len(tracks) > 0:
            for track in tracks:
                x1, y1, x2, y2, id_, conf, cls = track[:7]
                track_id = int(id_)
                
                # Determine track status
                if self.tracker_type == "tlukf":
                    if conf >= 0.6:
                        status = "strong"  # Source + Primary updated
                    elif conf >= 0.35:
                        status = "weak"    # Only Primary updated
                    else:
                        status = "virtual" # TLUKF prediction
                elif self.tracker_type == "strongsort_xysr":
                    if conf >= 0.45:
                        status = "real"
                    else:
                        status = "virtual"
                else:
                    status = "real"
                
                track_data = {
                    'id': track_id,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(conf),
                    'class': int(cls),
                    'status': status
                }
                
                frame_info['tracks'].append(track_data)
                
                # Update ID history
                self.id_history[track_id].append({
                    'frame': frame_id,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(conf),
                    'status': status
                })
        
        # Detect matching events
        self._detect_matching_events(frame_id, frame_info)
        
        self.frame_stats.append(frame_info)
        
        return frame_info
    
    def _detect_matching_events(self, frame_id, frame_info):
        """Detect important matching events."""
        
        # Event 1: ID Switch (same bbox matched to different ID)
        if frame_id > 0:
            prev_frame = self.frame_stats[-1] if len(self.frame_stats) > 0 else None
            if prev_frame:
                for curr_track in frame_info['tracks']:
                    for prev_track in prev_frame['tracks']:
                        iou = self._calculate_iou(curr_track['bbox'], prev_track['bbox'])
                        if iou > 0.7 and curr_track['id'] != prev_track['id']:
                            # Potential ID switch
                            if curr_track['status'] != 'virtual' and prev_track['status'] != 'virtual':
                                self.matching_events.append({
                                    'frame': frame_id,
                                    'type': 'ID_SWITCH',
                                    'old_id': prev_track['id'],
                                    'new_id': curr_track['id'],
                                    'iou': float(iou),
                                    'bbox': curr_track['bbox']
                                })
        
        # Event 2: Track Lost (ID disappeared)
        if frame_id > 0 and len(self.frame_stats) > 0:
            prev_ids = {t['id'] for t in self.frame_stats[-1]['tracks']}
            curr_ids = {t['id'] for t in frame_info['tracks']}
            lost_ids = prev_ids - curr_ids
            
            for lost_id in lost_ids:
                # Check if it's really lost (not just virtual)
                prev_track = next((t for t in self.frame_stats[-1]['tracks'] if t['id'] == lost_id), None)
                if prev_track and prev_track['status'] != 'virtual':
                    self.matching_events.append({
                        'frame': frame_id,
                        'type': 'TRACK_LOST',
                        'id': lost_id,
                        'prev_status': prev_track['status']
                    })
        
        # Event 3: Track Recovered (ID reappeared)
        if frame_id > 0 and len(self.frame_stats) > 0:
            prev_ids = {t['id'] for t in self.frame_stats[-1]['tracks']}
            curr_ids = {t['id'] for t in frame_info['tracks']}
            new_ids = curr_ids - prev_ids
            
            # Check if any new ID was seen before
            for new_id in new_ids:
                if new_id in self.id_history and len(self.id_history[new_id]) > 1:
                    # This ID was seen before - recovered!
                    gap = frame_id - self.id_history[new_id][-2]['frame']
                    if gap > 1:
                        self.matching_events.append({
                            'frame': frame_id,
                            'type': 'TRACK_RECOVERED',
                            'id': new_id,
                            'gap_frames': gap
                        })
        
        # Event 4: Virtual Box Created (TLUKF/XYSR specific)
        if self.tracker_type in ['tlukf', 'strongsort_xysr']:
            virtual_tracks = [t for t in frame_info['tracks'] if t['status'] == 'virtual']
            if len(virtual_tracks) > 0:
                for vt in virtual_tracks:
                    self.matching_events.append({
                        'frame': frame_id,
                        'type': 'VIRTUAL_CREATED',
                        'id': vt['id'],
                        'confidence': vt['confidence']
                    })
    
    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0
    
    def run(self, max_frames=None):
        """Run tracking analysis."""
        frame_id = 0
        start_time = perf_counter()
        
        print(f"\n🚀 Starting analysis...")
        print(f"{'='*60}")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Detect objects
            detections_list = self._detect(frame)
            
            for dets in detections_list:
                det_boxes = dets.boxes.data.to("cpu").numpy()
                
                # Update tracker
                if det_boxes.size > 0:
                    tracks = self.tracker.update(det_boxes, frame)
                else:
                    tracks = self.tracker.update(np.empty((0, 6), dtype=np.float32), frame)
                
                # Analyze frame
                frame_info = self._analyze_frame(frame_id, det_boxes, tracks)
                
                # Print frame summary
                if frame_id % 30 == 0:  # Every 30 frames
                    print(f"Frame {frame_id:4d}: {frame_info['num_detections']} dets, "
                          f"{frame_info['num_tracks']} tracks, "
                          f"{len([t for t in frame_info['tracks'] if t['status'] == 'virtual'])} virtual")
            
            frame_id += 1
            
            if max_frames and frame_id >= max_frames:
                break
        
        end_time = perf_counter()
        total_time = end_time - start_time
        
        print(f"{'='*60}")
        print(f"✅ Analysis completed!")
        print(f"⏱️  Total time: {total_time:.2f}s ({frame_id/total_time:.2f} FPS)")
        print(f"📊 Frames processed: {frame_id}")
        print(f"🎯 Total tracks: {len(self.id_history)}")
        print(f"⚠️  Matching events: {len(self.matching_events)}")
        
        self.cap.release()
    
    def save_results(self):
        """Save analysis results to JSON."""
        results = {
            'tracker_type': self.tracker_type,
            'video_path': str(self.video_path),
            'video_info': {
                'width': self.width,
                'height': self.height,
                'fps': self.fps,
                'total_frames': self.total_frames
            },
            'statistics': {
                'total_tracks': len(self.id_history),
                'total_frames_processed': len(self.frame_stats),
                'matching_events': len(self.matching_events),
                'id_switches': len([e for e in self.matching_events if e['type'] == 'ID_SWITCH']),
                'tracks_lost': len([e for e in self.matching_events if e['type'] == 'TRACK_LOST']),
                'tracks_recovered': len([e for e in self.matching_events if e['type'] == 'TRACK_RECOVERED']),
                'virtual_boxes_created': len([e for e in self.matching_events if e['type'] == 'VIRTUAL_CREATED'])
            },
            'id_history': {str(k): v for k, v in self.id_history.items()},
            'matching_events': self.matching_events,
            'frame_stats': self.frame_stats
        }
        
        # Save to JSON
        output_file = self.output_dir / "matching_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        return results
    
    def visualize_results(self):
        """Create visualization plots."""
        print(f"\n📊 Creating visualizations...")
        
        # Plot 1: Tracks per frame
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Matching Analysis: {self.tracker_type}', fontsize=16)
        
        # Tracks count over time
        frames = [s['frame_id'] for s in self.frame_stats]
        num_tracks = [s['num_tracks'] for s in self.frame_stats]
        num_detections = [s['num_detections'] for s in self.frame_stats]
        
        axes[0, 0].plot(frames, num_tracks, label='Tracks', color='blue', linewidth=2)
        axes[0, 0].plot(frames, num_detections, label='Detections', color='red', alpha=0.5)
        axes[0, 0].set_xlabel('Frame')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Tracks vs Detections')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # ID switches over time
        id_switch_frames = [e['frame'] for e in self.matching_events if e['type'] == 'ID_SWITCH']
        axes[0, 1].scatter(id_switch_frames, [1]*len(id_switch_frames), color='red', alpha=0.6, s=50)
        axes[0, 1].set_xlabel('Frame')
        axes[0, 1].set_ylabel('Event')
        axes[0, 1].set_title(f'ID Switches: {len(id_switch_frames)}')
        axes[0, 1].set_yticks([])
        axes[0, 1].grid(True, alpha=0.3)
        
        # Track lifecycle (track duration)
        track_durations = {}
        for track_id, history in self.id_history.items():
            duration = len(history)
            track_durations[track_id] = duration
        
        if len(track_durations) > 0:
            axes[1, 0].bar(range(len(track_durations)), list(track_durations.values()), color='green', alpha=0.7)
            axes[1, 0].set_xlabel('Track ID')
            axes[1, 0].set_ylabel('Duration (frames)')
            axes[1, 0].set_title('Track Durations')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Event distribution
        event_types = [e['type'] for e in self.matching_events]
        event_counts = {}
        for event_type in set(event_types):
            event_counts[event_type] = event_types.count(event_type)
        
        if len(event_counts) > 0:
            axes[1, 1].bar(event_counts.keys(), event_counts.values(), color='orange', alpha=0.7)
            axes[1, 1].set_xlabel('Event Type')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Matching Events Distribution')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / "matching_visualization.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"📈 Visualization saved to: {plot_file}")
        
        plt.close()
    
    def print_summary(self):
        """Print detailed summary."""
        print(f"\n{'='*60}")
        print(f"📋 MATCHING ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"Tracker: {self.tracker_type}")
        print(f"Video: {self.video_path.name}")
        print(f"\n📊 Statistics:")
        print(f"  - Total Tracks: {len(self.id_history)}")
        print(f"  - Frames Processed: {len(self.frame_stats)}")
        print(f"  - ID Switches: {len([e for e in self.matching_events if e['type'] == 'ID_SWITCH'])}")
        print(f"  - Tracks Lost: {len([e for e in self.matching_events if e['type'] == 'TRACK_LOST'])}")
        print(f"  - Tracks Recovered: {len([e for e in self.matching_events if e['type'] == 'TRACK_RECOVERED'])}")
        
        if self.tracker_type in ['tlukf', 'strongsort_xysr']:
            print(f"  - Virtual Boxes: {len([e for e in self.matching_events if e['type'] == 'VIRTUAL_CREATED'])}")
        
        print(f"\n🎯 Top 5 Longest Tracks:")
        sorted_tracks = sorted(self.id_history.items(), key=lambda x: len(x[1]), reverse=True)[:5]
        for rank, (track_id, history) in enumerate(sorted_tracks, 1):
            print(f"  {rank}. Track {track_id}: {len(history)} frames")
        
        print(f"\n⚠️  Critical Events:")
        id_switches = [e for e in self.matching_events if e['type'] == 'ID_SWITCH']
        if len(id_switches) > 0:
            print(f"  ID Switches (first 5):")
            for event in id_switches[:5]:
                print(f"    Frame {event['frame']}: ID {event['old_id']} → {event['new_id']} (IoU: {event['iou']:.3f})")
        
        recoveries = [e for e in self.matching_events if e['type'] == 'TRACK_RECOVERED']
        if len(recoveries) > 0:
            print(f"\n  Track Recoveries (first 5):")
            for event in recoveries[:5]:
                print(f"    Frame {event['frame']}: ID {event['id']} recovered after {event['gap_frames']} frames")
        
        print(f"{'='*60}\n")


def compare_trackers(video_path, model_weights, output_dir, max_frames=None):
    """Compare all three trackers on same video."""
    tracker_types = ["strongsort", "strongsort_xysr", "tlukf"]
    results = {}
    
    print(f"\n🔬 COMPARATIVE ANALYSIS")
    print(f"{'='*60}")
    print(f"Video: {video_path}")
    print(f"Trackers: {', '.join(tracker_types)}")
    print(f"{'='*60}\n")
    
    for tracker_type in tracker_types:
        print(f"\n{'#'*60}")
        print(f"# Analyzing: {tracker_type.upper()}")
        print(f"{'#'*60}\n")
        
        analyzer = MatchingAnalyzer(tracker_type, model_weights, video_path, output_dir)
        analyzer.run(max_frames=max_frames)
        result = analyzer.save_results()
        analyzer.visualize_results()
        analyzer.print_summary()
        
        results[tracker_type] = result
    
    # Comparative summary
    print(f"\n{'='*60}")
    print(f"📊 COMPARATIVE SUMMARY")
    print(f"{'='*60}\n")
    
    print(f"{'Metric':<30} | {'StrongSort':>15} | {'StrongSort XYSR':>15} | {'TLUKF':>15}")
    print(f"{'-'*80}")
    
    metrics = [
        ('Total Tracks', 'total_tracks'),
        ('ID Switches', 'id_switches'),
        ('Tracks Lost', 'tracks_lost'),
        ('Tracks Recovered', 'tracks_recovered'),
        ('Virtual Boxes', 'virtual_boxes_created')
    ]
    
    for metric_name, metric_key in metrics:
        values = []
        for tracker_type in tracker_types:
            value = results[tracker_type]['statistics'].get(metric_key, 0)
            values.append(f"{value:>15}")
        
        print(f"{metric_name:<30} | {' | '.join(values)}")
    
    print(f"{'='*60}\n")
    
    # Save comparison
    comparison_file = Path(output_dir) / "comparison_summary.json"
    with open(comparison_file, 'w') as f:
        json.dump({
            'video': str(video_path),
            'trackers': tracker_types,
            'results': {k: v['statistics'] for k, v in results.items()}
        }, f, indent=2)
    
    print(f"💾 Comparison saved to: {comparison_file}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze ID matching for different trackers")
    parser.add_argument("--video_path", type=str, required=True, help="Path to video file")
    parser.add_argument("--model_weights", type=str, required=True, help="Path to YOLO model weights")
    parser.add_argument("--tracker", type=str, choices=["strongsort", "strongsort_xysr", "tlukf", "all"], 
                        default="all", help="Tracker to analyze (default: all)")
    parser.add_argument("--output_dir", type=str, default="matching_analysis", help="Output directory")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum frames to process (default: all)")
    
    args = parser.parse_args()
    
    if args.tracker == "all":
        # Compare all trackers
        compare_trackers(args.video_path, args.model_weights, args.output_dir, args.max_frames)
    else:
        # Analyze single tracker
        analyzer = MatchingAnalyzer(args.tracker, args.model_weights, args.video_path, args.output_dir)
        analyzer.run(max_frames=args.max_frames)
        analyzer.save_results()
        analyzer.visualize_results()
        analyzer.print_summary()


if __name__ == "__main__":
    main()
