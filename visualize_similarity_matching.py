"""
Visualize Similarity Measurement and Matching Process
======================================================
Tool để kiểm tra chi tiết quá trình matching ID thông qua similarity measurement
cho 3 phương pháp: StrongSort (XYAH), StrongSortXYSR, TLUKF

Features:
- Cost matrix visualization (appearance + motion)
- Gating mask visualization
- Matching assignments
- Track-Detection associations
- Side-by-side comparison of 3 methods
"""

import numpy as np
import cv2
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from ultralytics import YOLO
from boxmot import StrongSort, StrongSortXYSR, StrongSortTLUKF
import argparse
from collections import defaultdict
import json

# Monkey patch để capture matching data
class MatchingDataCollector:
    """Collector để lưu dữ liệu matching process"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.frame_data = {
            'cost_matrix': None,
            'gating_mask': None,
            'assignments': None,
            'unmatched_tracks': None,
            'unmatched_detections': None,
            'tracks': [],
            'detections': [],
            'appearance_dists': None,
            'motion_dists': None,
            'track_statuses': []  # strong/weak/virtual
        }
    
    def collect(self, **kwargs):
        """Collect matching data"""
        for key, value in kwargs.items():
            if key in self.frame_data:
                self.frame_data[key] = value

# Global collectors for each tracker
collectors = {
    'strongsort': MatchingDataCollector(),
    'strongsort_xysr': MatchingDataCollector(),
    'tlukf': MatchingDataCollector()
}

def patch_tracker_for_data_collection(tracker, collector, tracker_type):
    """Monkey patch tracker để collect matching data"""
    
    if tracker_type == "strongsort":
        # Patch StrongSort (XYAH)
        original_match = tracker.tracker._match
        
        def patched_match(detections):
            # Call original
            matches, unmatched_tracks, unmatched_dets = original_match(detections)
            
            # Collect data
            if len(tracker.tracker.tracks) > 0 and len(detections) > 0:
                try:
                    # Get appearance distances
                    features = []
                    for d in detections:
                        if hasattr(d, '__len__') and len(d) > 6:
                            features.append(d[-1])
                    
                    if len(features) > 0:
                        features = np.array(features)
                        cost_matrix = tracker.tracker.metric.distance(
                            [t.features for t in tracker.tracker.tracks], features
                        )
                        
                        # Get track info
                        track_info = []
                        for t in tracker.tracker.tracks:
                            bbox = t.mean[:4].copy()
                            tid = getattr(t, 'track_id', getattr(t, 'id', 0))
                            track_info.append((bbox, tid, 'strong'))
                        
                        collector.collect(
                            cost_matrix=cost_matrix.copy(),
                            assignments=matches.copy() if len(matches) > 0 else np.array([]),
                            unmatched_tracks=unmatched_tracks.copy(),
                            unmatched_detections=unmatched_dets.copy(),
                            tracks=track_info,
                            detections=[d[:4].copy() if hasattr(d, '__len__') else d.tlbr for d in detections],
                            appearance_dists=cost_matrix.copy(),
                            track_statuses=['strong'] * len(tracker.tracker.tracks)
                        )
                except Exception as e:
                    print(f"Error collecting StrongSort data: {e}")
            
            return matches, unmatched_tracks, unmatched_dets
        
        tracker.tracker._match = patched_match
    
    elif tracker_type in ["strongsort_xysr", "tlukf"]:
        # Patch StrongSortXYSR and TLUKF (same base class)
        original_match = tracker.tracker._match
        
        def patched_match(detections, only_position=False):
            # Call original - check signature compatibility
            try:
                # Try with only_position parameter (TLUKF)
                matches, unmatched_tracks, unmatched_dets = original_match(detections, only_position)
            except TypeError:
                # Fallback to single argument (StrongSortXYSR)
                matches, unmatched_tracks, unmatched_dets = original_match(detections)
            
            # Collect data
            if len(tracker.tracker.tracks) > 0 and len(detections) > 0:
                try:
                    # Get track info with status
                    track_info = []
                    track_statuses = []
                    for t in tracker.tracker.tracks:
                        bbox = t.to_tlbr()
                        # Get track ID - different attribute names
                        tid = getattr(t, 'track_id', getattr(t, 'id', 0))
                        
                        # Determine status based on confidence
                        if hasattr(t, 'conf'):
                            if t.conf >= 0.6:
                                status = 'strong'
                            elif t.conf >= 0.3:
                                status = 'weak'
                            else:
                                status = 'virtual'
                        elif t.time_since_update >= 1:
                            status = 'virtual'
                        else:
                            status = 'strong'
                        track_info.append((bbox, tid, status))
                        track_statuses.append(status)
                    
                    # Get cost matrix if available
                    features = []
                    for d in detections:
                        if hasattr(d, '__len__') and len(d) > 6:
                            features.append(d[-1])
                    
                    if len(features) > 0 and hasattr(tracker.tracker, 'metric'):
                        try:
                            features = np.array(features)
                            cost_matrix = tracker.tracker.metric.distance(
                                [t.features for t in tracker.tracker.tracks], features
                            )
                        except:
                            cost_matrix = np.ones((len(tracker.tracker.tracks), len(detections)))
                    else:
                        cost_matrix = np.ones((len(tracker.tracker.tracks), len(detections)))
                    
                    # Get detection bboxes
                    det_bboxes = []
                    for d in detections:
                        if hasattr(d, '__len__'):
                            det_bboxes.append(d[:4].copy())
                        elif hasattr(d, 'tlbr'):
                            det_bboxes.append(d.tlbr)
                        else:
                            det_bboxes.append([0,0,0,0])
                    
                    collector.collect(
                        cost_matrix=cost_matrix.copy(),
                        assignments=matches.copy() if len(matches) > 0 else np.array([]),
                        unmatched_tracks=unmatched_tracks.copy(),
                        unmatched_detections=unmatched_dets.copy(),
                        tracks=track_info,
                        detections=det_bboxes,
                        appearance_dists=cost_matrix.copy(),
                        track_statuses=track_statuses
                    )
                except Exception as e:
                    print(f"Error collecting {tracker_type} data: {e}")
            
            return matches, unmatched_tracks, unmatched_dets
        
        tracker.tracker._match = patched_match

class SimilarityMatchingVisualizer:
    """Visualizer cho similarity measurement và matching process"""
    
    def __init__(self, video_path, model_weights, output_dir="similarity_analysis", max_frames=None):
        self.video_path = Path(video_path)
        self.model_weights = Path(model_weights)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.max_frames = max_frames
        
        # Device
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load model
        print(f"Loading model: {self.model_weights}")
        self.model = YOLO(str(self.model_weights))
        
        # Initialize trackers
        print("Initializing trackers...")
        reid_weights = Path("osnet_dcn_x0_5_endocv.pt")
        
        self.trackers = {
            'strongsort': StrongSort(
                reid_weights, torch.device(self.device),
                fp16=False, max_dist=0.95, max_iou_dist=0.95, max_age=300, half=False
            ),
            'strongsort_xysr': StrongSortXYSR(
                reid_weights, torch.device(self.device),
                fp16=False, max_dist=0.95, max_iou_dist=0.95, max_age=300, half=False
            ),
            'tlukf': StrongSortTLUKF(
                reid_weights, torch.device(self.device),
                fp16=False, max_dist=0.95, max_iou_dist=0.95, max_age=300, half=False
            )
        }
        
        # Patch trackers to collect data
        for name, tracker in self.trackers.items():
            patch_tracker_for_data_collection(tracker, collectors[name], name)
        
        # Video capture
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup matplotlib style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (20, 12)
    
    def visualize_frame(self, frame_id, frame, show_plots=True):
        """Visualize matching process for current frame"""
        
        # Detect objects
        results = self.model(frame, stream=True, verbose=False, conf=0.3, line_width=1)
        
        for dets in results:
            det_boxes = dets.boxes.data.to("cpu").numpy()
            if det_boxes.size == 0:
                continue
            
            # Update all trackers
            all_tracks = {}
            for name, tracker in self.trackers.items():
                if name == "strongsort":
                    # StrongSort uses conf=0.6
                    high_conf_dets = det_boxes[det_boxes[:, 4] >= 0.6]
                    if len(high_conf_dets) > 0:
                        tracks = tracker.update(high_conf_dets, frame)
                        all_tracks[name] = tracks
                    else:
                        all_tracks[name] = np.array([])
                elif name == "strongsort_xysr":
                    # XYSR uses conf=0.45
                    medium_conf_dets = det_boxes[det_boxes[:, 4] >= 0.45]
                    if len(medium_conf_dets) > 0:
                        tracks = tracker.update(medium_conf_dets, frame)
                        all_tracks[name] = tracks
                    else:
                        all_tracks[name] = np.array([])
                else:  # tlukf
                    # TLUKF uses all detections conf=0.3
                    tracks = tracker.update(det_boxes, frame)
                    all_tracks[name] = tracks
            
            # Visualize if there are matches
            if show_plots and any(len(collectors[name].frame_data['tracks']) > 0 
                                 for name in self.trackers.keys()):
                self._create_visualization(frame_id, frame, all_tracks)
    
    def _create_visualization(self, frame_id, frame, all_tracks):
        """Create comprehensive visualization"""
        
        fig = plt.figure(figsize=(24, 16))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        tracker_names = ['strongsort', 'strongsort_xysr', 'tlukf']
        display_names = ['StrongSort (XYAH)', 'StrongSort (XYSR)', 'TLUKF']
        
        for idx, (tracker_name, display_name) in enumerate(zip(tracker_names, display_names)):
            collector = collectors[tracker_name]
            data = collector.frame_data
            
            if len(data['tracks']) == 0:
                continue
            
            # Row 1: Cost Matrix
            ax_cost = fig.add_subplot(gs[0, idx])
            self._plot_cost_matrix(ax_cost, data, display_name)
            
            # Row 2: Track-Detection Matching
            ax_match = fig.add_subplot(gs[1, idx])
            self._plot_matching(ax_match, data, frame, display_name)
            
            # Row 3: Distance Distribution
            ax_dist = fig.add_subplot(gs[2, idx])
            self._plot_distance_distribution(ax_dist, data, display_name)
        
        # Column 4: Summary comparison
        ax_summary = fig.add_subplot(gs[:, 3])
        self._plot_summary_comparison(ax_summary, all_tracks, tracker_names, display_names)
        
        plt.suptitle(f'Similarity Measurement Analysis - Frame {frame_id}', 
                     fontsize=16, fontweight='bold')
        
        # Save
        output_path = self.output_dir / f'similarity_frame_{frame_id:04d}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization: {output_path}")
    
    def _plot_cost_matrix(self, ax, data, title):
        """Plot cost matrix with assignments"""
        if data['cost_matrix'] is None or data['cost_matrix'].size == 0:
            ax.text(0.5, 0.5, 'No cost matrix', ha='center', va='center')
            ax.set_title(f'{title}\nCost Matrix')
            return
        
        cost_matrix = data['cost_matrix']
        n_tracks, n_dets = cost_matrix.shape
        
        # Plot heatmap
        sns.heatmap(cost_matrix, annot=True, fmt='.2f', cmap='RdYlGn_r',
                   cbar_kws={'label': 'Distance'}, ax=ax,
                   vmin=0, vmax=1, center=0.5)
        
        # Mark assignments
        if data['assignments'] is not None and len(data['assignments']) > 0:
            for track_idx, det_idx in data['assignments']:
                ax.add_patch(patches.Rectangle(
                    (det_idx, track_idx), 1, 1,
                    fill=False, edgecolor='blue', linewidth=3
                ))
        
        # Mark unmatched
        if data['unmatched_tracks'] is not None:
            for track_idx in data['unmatched_tracks']:
                ax.add_patch(patches.Rectangle(
                    (-0.5, track_idx), 0.5, 1,
                    fill=True, facecolor='red', alpha=0.3
                ))
        
        if data['unmatched_detections'] is not None:
            for det_idx in data['unmatched_detections']:
                ax.add_patch(patches.Rectangle(
                    (det_idx, -0.5), 1, 0.5,
                    fill=True, facecolor='orange', alpha=0.3
                ))
        
        ax.set_xlabel('Detections')
        ax.set_ylabel('Tracks')
        ax.set_title(f'{title}\nCost Matrix (Blue=Match, Red=Unmatched Track)')
        
        # Add track status labels
        if data['track_statuses']:
            yticks = ax.get_yticks()
            yticklabels = [f"T{i}\n({s})" for i, s in enumerate(data['track_statuses'])]
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels, fontsize=8)
    
    def _plot_matching(self, ax, data, frame, title):
        """Plot track-detection matching on frame"""
        # Draw frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax.imshow(frame_rgb)
        
        # Draw detections (green)
        for det_bbox in data['detections']:
            x1, y1, x2, y2 = det_bbox
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                    linewidth=2, edgecolor='green',
                                    facecolor='none', linestyle='--',
                                    label='Detection')
            ax.add_patch(rect)
        
        # Draw tracks with status-based colors
        colors = {'strong': 'blue', 'weak': 'orange', 'virtual': 'gray'}
        for track_bbox, track_id, status in data['tracks']:
            x1, y1, x2, y2 = track_bbox
            color = colors.get(status, 'blue')
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                    linewidth=3, edgecolor=color,
                                    facecolor='none',
                                    label=f'Track ({status})')
            ax.add_patch(rect)
            ax.text(x1, y1-5, f'ID:{track_id}', color=color,
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Draw matching lines
        if data['assignments'] is not None and len(data['assignments']) > 0:
            for track_idx, det_idx in data['assignments']:
                if track_idx < len(data['tracks']) and det_idx < len(data['detections']):
                    track_bbox, _, _ = data['tracks'][track_idx]
                    det_bbox = data['detections'][det_idx]
                    
                    # Center points
                    track_center = ((track_bbox[0]+track_bbox[2])/2,
                                  (track_bbox[1]+track_bbox[3])/2)
                    det_center = ((det_bbox[0]+det_bbox[2])/2,
                                (det_bbox[1]+det_bbox[3])/2)
                    
                    ax.plot([track_center[0], det_center[0]],
                           [track_center[1], det_center[1]],
                           'b-', linewidth=2, alpha=0.6)
        
        ax.set_title(f'{title}\nTrack-Detection Matching')
        ax.axis('off')
        
        # Remove duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)
    
    def _plot_distance_distribution(self, ax, data, title):
        """Plot distribution of distances"""
        if data['cost_matrix'] is None or data['cost_matrix'].size == 0:
            ax.text(0.5, 0.5, 'No distance data', ha='center', va='center')
            ax.set_title(f'{title}\nDistance Distribution')
            return
        
        # Flatten cost matrix
        all_dists = data['cost_matrix'].flatten()
        
        # Get matched distances
        matched_dists = []
        if data['assignments'] is not None and len(data['assignments']) > 0:
            for track_idx, det_idx in data['assignments']:
                matched_dists.append(data['cost_matrix'][track_idx, det_idx])
        
        # Plot histograms
        if len(all_dists) > 0:
            ax.hist(all_dists, bins=20, alpha=0.5, label='All pairs', color='gray')
        if len(matched_dists) > 0:
            ax.hist(matched_dists, bins=10, alpha=0.7, label='Matched pairs', color='blue')
        
        ax.axvline(x=0.5, color='r', linestyle='--', label='Threshold (0.5)')
        ax.set_xlabel('Distance')
        ax.set_ylabel('Count')
        ax.set_title(f'{title}\nDistance Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        if len(matched_dists) > 0:
            stats_text = f"Matches: {len(matched_dists)}\n"
            stats_text += f"Mean: {np.mean(matched_dists):.3f}\n"
            stats_text += f"Min: {np.min(matched_dists):.3f}\n"
            stats_text += f"Max: {np.max(matched_dists):.3f}"
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=9)
    
    def _plot_summary_comparison(self, ax, all_tracks, tracker_names, display_names):
        """Plot summary comparison of all trackers"""
        stats = []
        for name in tracker_names:
            collector = collectors[name]
            data = collector.frame_data
            
            n_tracks = len(data['tracks'])
            n_dets = len(data['detections'])
            n_matches = len(data['assignments']) if data['assignments'] is not None else 0
            n_unmatched_tracks = len(data['unmatched_tracks']) if data['unmatched_tracks'] is not None else 0
            n_unmatched_dets = len(data['unmatched_detections']) if data['unmatched_detections'] is not None else 0
            
            # Count by status
            status_counts = {'strong': 0, 'weak': 0, 'virtual': 0}
            for _, _, status in data['tracks']:
                status_counts[status] += 1
            
            stats.append({
                'name': name,
                'tracks': n_tracks,
                'detections': n_dets,
                'matches': n_matches,
                'unmatched_tracks': n_unmatched_tracks,
                'unmatched_dets': n_unmatched_dets,
                'strong': status_counts['strong'],
                'weak': status_counts['weak'],
                'virtual': status_counts['virtual']
            })
        
        # Create comparison table
        ax.axis('tight')
        ax.axis('off')
        
        table_data = []
        table_data.append(['Metric'] + display_names)
        table_data.append(['Total Tracks'] + [str(s['tracks']) for s in stats])
        table_data.append(['├─ Strong'] + [str(s['strong']) for s in stats])
        table_data.append(['├─ Weak'] + [str(s['weak']) for s in stats])
        table_data.append(['└─ Virtual'] + [str(s['virtual']) for s in stats])
        table_data.append(['Detections'] + [str(s['detections']) for s in stats])
        table_data.append(['Matches'] + [str(s['matches']) for s in stats])
        table_data.append(['Unmatched Tracks'] + [str(s['unmatched_tracks']) for s in stats])
        table_data.append(['Unmatched Dets'] + [str(s['unmatched_dets']) for s in stats])
        
        table = ax.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color header
        for i in range(4):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color metrics column
        for i in range(1, len(table_data)):
            table[(i, 0)].set_facecolor('#E0E0E0')
            table[(i, 0)].set_text_props(weight='bold')
        
        ax.set_title('Summary Comparison', fontsize=14, fontweight='bold', pad=20)
    
    def run(self):
        """Run visualization for video"""
        print(f"\nProcessing video: {self.video_path}")
        print(f"Output directory: {self.output_dir}")
        
        frame_id = 0
        saved_frames = []
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Only visualize interesting frames (with detections)
            # Check every 10 frames to avoid too many outputs
            if frame_id % 10 == 0:
                self.visualize_frame(frame_id, frame, show_plots=True)
                saved_frames.append(frame_id)
            else:
                self.visualize_frame(frame_id, frame, show_plots=False)
            
            frame_id += 1
            
            if self.max_frames and frame_id >= self.max_frames:
                break
            
            if frame_id % 50 == 0:
                print(f"Processed {frame_id} frames...")
        
        self.cap.release()
        
        print(f"\n✅ Visualization complete!")
        print(f"Total frames processed: {frame_id}")
        print(f"Visualizations saved: {len(saved_frames)}")
        print(f"Output directory: {self.output_dir}")
        
        # Create summary
        self._create_summary(saved_frames)
    
    def _create_summary(self, saved_frames):
        """Create summary of visualizations"""
        summary = {
            'video': str(self.video_path),
            'total_frames': saved_frames[-1] if saved_frames else 0,
            'visualized_frames': saved_frames,
            'output_dir': str(self.output_dir)
        }
        
        summary_path = self.output_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary saved: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize Similarity Measurement and Matching Process")
    parser.add_argument("--video_path", type=str, required=True,
                       help="Path to input video")
    parser.add_argument("--model_weights", type=str, required=True,
                       help="Path to YOLO model weights")
    parser.add_argument("--output_dir", type=str, default="similarity_analysis",
                       help="Output directory for visualizations")
    parser.add_argument("--max_frames", type=int, default=200,
                       help="Maximum frames to process (default: 200)")
    
    args = parser.parse_args()
    
    visualizer = SimilarityMatchingVisualizer(
        video_path=args.video_path,
        model_weights=args.model_weights,
        output_dir=args.output_dir,
        max_frames=args.max_frames
    )
    
    visualizer.run()

if __name__ == "__main__":
    main()
