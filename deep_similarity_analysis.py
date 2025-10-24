"""
Deep Similarity Measurement Analysis Tool for Multi-Object Tracking

This tool provides comprehensive analysis of the similarity measurement process
across three tracking methods: StrongSort (XYAH), StrongSortXYSR, and TLUKF.

Key Analysis Features:
1. Feature Quality Analysis: Check ReID feature quality and consistency
2. Distance Matrix Analysis: Appearance vs Motion distance breakdown
3. Gating Analysis: Track when/why gating rejects matches
4. Assignment Conflicts: Detect multiple tracks competing for same detection
5. ID Switch Detection: Pinpoint exact frames and reasons for ID switches
6. Cost Matrix Evolution: Track how costs change over time for same track-det pairs

Output:
- Detailed frame-by-frame analysis logs
- Statistical summaries (mean distance, gating rate, conflict rate)
- ID switch events with root cause analysis
- Visualization of problematic frames
"""

import argparse
import json
import sys
import numpy as np
import torch
import cv2
from pathlib import Path
from ultralytics import YOLO
from boxmot import StrongSort, StrongSortXYSR, StrongSortTLUKF
from collections import defaultdict
from datetime import datetime


class FeatureQualityAnalyzer:
    """Analyze ReID feature quality and consistency."""
    
    def __init__(self):
        self.track_features = defaultdict(list)  # track_id -> list of features
        self.feature_stats = {
            'norm_mean': [],
            'norm_std': [],
            'cosine_similarity_within_track': defaultdict(list),
        }
    
    def add_track_feature(self, track_id, feature):
        """Store feature for a track."""
        if feature is not None and len(feature) > 0:
            feature_np = np.asarray(feature).flatten()
            self.track_features[track_id].append(feature_np)
            
            # Compute feature norm
            norm = np.linalg.norm(feature_np)
            self.feature_stats['norm_mean'].append(norm)
            
            # Compute cosine similarity with previous features of same track
            if len(self.track_features[track_id]) > 1:
                prev_feature = self.track_features[track_id][-2]
                similarity = np.dot(feature_np, prev_feature) / (
                    np.linalg.norm(feature_np) * np.linalg.norm(prev_feature) + 1e-6
                )
                self.feature_stats['cosine_similarity_within_track'][track_id].append(similarity)
    
    def get_feature_consistency(self, track_id):
        """Get feature consistency score for a track (0-1, higher is better)."""
        if track_id not in self.feature_stats['cosine_similarity_within_track']:
            return None
        similarities = self.feature_stats['cosine_similarity_within_track'][track_id]
        if len(similarities) == 0:
            return None
        return np.mean(similarities)
    
    def get_summary(self):
        """Get summary statistics."""
        summary = {
            'avg_feature_norm': np.mean(self.feature_stats['norm_mean']) if self.feature_stats['norm_mean'] else 0,
            'std_feature_norm': np.std(self.feature_stats['norm_mean']) if self.feature_stats['norm_mean'] else 0,
        }
        
        # Per-track consistency
        track_consistency = {}
        for track_id, similarities in self.feature_stats['cosine_similarity_within_track'].items():
            if len(similarities) > 0:
                track_consistency[track_id] = {
                    'mean_similarity': np.mean(similarities),
                    'std_similarity': np.std(similarities),
                    'min_similarity': np.min(similarities),
                    'num_samples': len(similarities)
                }
        summary['track_consistency'] = track_consistency
        
        return summary


class DistanceMatrixAnalyzer:
    """Analyze appearance and motion distance matrices."""
    
    def __init__(self):
        self.frame_data = []
        self.distance_stats = {
            'appearance': [],
            'motion': [],
            'combined': [],
        }
    
    def add_frame_data(self, frame_id, cost_matrix, tracks, detections, matches, 
                       appearance_dist=None, motion_dist=None):
        """Store distance matrix data for a frame."""
        frame_info = {
            'frame_id': frame_id,
            'cost_matrix': cost_matrix.copy() if cost_matrix is not None else None,
            'num_tracks': len(tracks),
            'num_detections': len(detections),
            'num_matches': len(matches),
            'appearance_dist': appearance_dist.copy() if appearance_dist is not None else None,
            'motion_dist': motion_dist.copy() if motion_dist is not None else None,
        }
        
        self.frame_data.append(frame_info)
        
        # Collect statistics
        if cost_matrix is not None and cost_matrix.size > 0:
            self.distance_stats['combined'].extend(cost_matrix.flatten().tolist())
        
        if appearance_dist is not None and appearance_dist.size > 0:
            self.distance_stats['appearance'].extend(appearance_dist.flatten().tolist())
        
        if motion_dist is not None and motion_dist.size > 0:
            self.distance_stats['motion'].extend(motion_dist.flatten().tolist())
    
    def get_summary(self):
        """Get summary statistics."""
        summary = {}
        
        for dist_type in ['appearance', 'motion', 'combined']:
            if len(self.distance_stats[dist_type]) > 0:
                summary[f'{dist_type}_mean'] = np.mean(self.distance_stats[dist_type])
                summary[f'{dist_type}_std'] = np.std(self.distance_stats[dist_type])
                summary[f'{dist_type}_min'] = np.min(self.distance_stats[dist_type])
                summary[f'{dist_type}_max'] = np.max(self.distance_stats[dist_type])
                summary[f'{dist_type}_median'] = np.median(self.distance_stats[dist_type])
        
        # Frame-level statistics
        summary['total_frames'] = len(self.frame_data)
        summary['avg_tracks_per_frame'] = np.mean([f['num_tracks'] for f in self.frame_data])
        summary['avg_detections_per_frame'] = np.mean([f['num_detections'] for f in self.frame_data])
        summary['avg_matches_per_frame'] = np.mean([f['num_matches'] for f in self.frame_data])
        
        return summary


class GatingAnalyzer:
    """Analyze gating decisions and their effects."""
    
    def __init__(self):
        self.gating_events = []
        self.gating_stats = {
            'total_pairs': 0,
            'gated_pairs': 0,
            'frames_with_gating': 0,
        }
    
    def add_gating_info(self, frame_id, gating_mask, cost_matrix, tracks, detections):
        """Store gating information for a frame."""
        if gating_mask is None or cost_matrix is None:
            return
        
        num_gated = np.sum(gating_mask)
        total_pairs = gating_mask.size
        
        self.gating_stats['total_pairs'] += total_pairs
        self.gating_stats['gated_pairs'] += num_gated
        
        if num_gated > 0:
            self.gating_stats['frames_with_gating'] += 1
            
            # Record gated pairs with their distances
            gated_positions = np.where(gating_mask)
            for track_idx, det_idx in zip(gated_positions[0], gated_positions[1]):
                try:
                    # Validate indices are within bounds
                    if (track_idx < len(tracks) and det_idx < len(detections) and
                        track_idx < cost_matrix.shape[0] and det_idx < cost_matrix.shape[1]):
                        event = {
                            'frame_id': frame_id,
                            'track_idx': int(track_idx),
                            'det_idx': int(det_idx),
                            'distance': cost_matrix[track_idx, det_idx] if cost_matrix[track_idx, det_idx] != np.inf else -1,
                            'track_id': getattr(tracks[track_idx], 'track_id', getattr(tracks[track_idx], 'id', track_idx)),
                        }
                        self.gating_events.append(event)
                except (IndexError, AttributeError):
                    # Skip invalid entries
                    continue
    
    def get_summary(self):
        """Get gating summary statistics."""
        gating_rate = (self.gating_stats['gated_pairs'] / self.gating_stats['total_pairs'] 
                       if self.gating_stats['total_pairs'] > 0 else 0)
        
        summary = {
            'total_pairs_checked': self.gating_stats['total_pairs'],
            'total_gated': self.gating_stats['gated_pairs'],
            'gating_rate': gating_rate,
            'frames_with_gating': self.gating_stats['frames_with_gating'],
            'total_gating_events': len(self.gating_events),
        }
        
        # Gating distance distribution
        if len(self.gating_events) > 0:
            distances = [e['distance'] for e in self.gating_events if e['distance'] >= 0]
            if len(distances) > 0:
                summary['gated_distance_mean'] = np.mean(distances)
                summary['gated_distance_std'] = np.std(distances)
        
        return summary


class AssignmentConflictAnalyzer:
    """Analyze assignment conflicts where multiple tracks compete for same detection."""
    
    def __init__(self):
        self.conflicts = []
        self.conflict_stats = {
            'total_conflicts': 0,
            'avg_competitors': [],
        }
    
    def add_assignment_data(self, frame_id, cost_matrix, matches, tracks, detections):
        """Detect and record assignment conflicts."""
        if cost_matrix is None or len(tracks) == 0 or len(detections) == 0:
            return
        
        # Cost matrix might be smaller than actual tracks/detections
        # It only contains the subset used in matching
        max_track_idx = cost_matrix.shape[0] if cost_matrix.ndim > 0 else 0
        max_det_idx = cost_matrix.shape[1] if cost_matrix.ndim > 1 else 0
        
        # For each detection, find all tracks that could match (distance < threshold)
        threshold = 0.7  # Typical threshold for conflicts
        
        for det_idx in range(min(len(detections), max_det_idx)):
            competing_tracks = []
            
            for track_idx in range(min(len(tracks), max_track_idx)):
                try:
                    distance = cost_matrix[track_idx, det_idx]
                    if distance != np.inf and distance < threshold:
                        track_id = getattr(tracks[track_idx], 'track_id', getattr(tracks[track_idx], 'id', track_idx))
                        competing_tracks.append({
                            'track_idx': track_idx,
                            'track_id': track_id,
                            'distance': distance,
                        })
                except IndexError:
                    # Skip if index out of bounds
                    continue
            
            # Conflict detected if multiple tracks compete
            if len(competing_tracks) > 1:
                # Find which track was actually assigned
                assigned_track = None
                for match in matches:
                    # Safely extract indices
                    if isinstance(match, (list, tuple)) and len(match) >= 2:
                        if match[1] == det_idx:
                            assigned_track = match[0]
                            break
                
                conflict = {
                    'frame_id': frame_id,
                    'det_idx': det_idx,
                    'num_competitors': len(competing_tracks),
                    'competing_tracks': competing_tracks,
                    'assigned_track_idx': assigned_track,
                }
                
                self.conflicts.append(conflict)
                self.conflict_stats['total_conflicts'] += 1
                self.conflict_stats['avg_competitors'].append(len(competing_tracks))
    
    def get_summary(self):
        """Get conflict summary statistics."""
        summary = {
            'total_conflicts': self.conflict_stats['total_conflicts'],
            'avg_competitors_per_conflict': (np.mean(self.conflict_stats['avg_competitors']) 
                                             if self.conflict_stats['avg_competitors'] else 0),
        }
        
        # Conflict distribution
        if len(self.conflicts) > 0:
            frames_with_conflicts = len(set(c['frame_id'] for c in self.conflicts))
            summary['frames_with_conflicts'] = frames_with_conflicts
            summary['conflicts_per_frame'] = self.conflict_stats['total_conflicts'] / frames_with_conflicts
        
        return summary


class IDSwitchDetector:
    """Detect and analyze ID switches."""
    
    def __init__(self):
        self.track_history = defaultdict(list)  # track_id -> list of (frame_id, bbox, feature)
        self.id_switches = []
        self.previous_assignments = {}  # det_spatial_id -> track_id (from previous frame)
        self.all_track_ids = set()  # Track all IDs seen so far
        self.new_id_appearances = []  # Track when new IDs appear
    
    def add_frame_assignments(self, frame_id, matches, tracks, detections, features=None):
        """Track assignments and detect ID switches."""
        current_assignments = {}
        current_track_ids = set()
        
        for match in matches:
            # Safely extract indices from match
            if isinstance(match, (list, tuple)) and len(match) >= 2:
                track_idx, det_idx = match[0], match[1]
            else:
                continue
            
            # Validate indices are within bounds
            if track_idx >= len(tracks) or det_idx >= len(detections):
                continue
            
            track_id = getattr(tracks[track_idx], 'track_id', getattr(tracks[track_idx], 'id', track_idx))
            current_track_ids.add(track_id)
            
            # Check if this is a NEW ID (not seen before)
            if track_id not in self.all_track_ids:
                self.all_track_ids.add(track_id)
                if frame_id > 5:  # Only count as "new" after initialization phase
                    self.new_id_appearances.append({
                        'frame_id': frame_id,
                        'new_track_id': track_id,
                        'type': 'new_id'
                    })
            
            # Get detection bbox safely
            try:
                if hasattr(detections[det_idx], '__len__'):
                    det_bbox = detections[det_idx][:4]
                elif hasattr(detections[det_idx], 'tlbr'):
                    det_bbox = detections[det_idx].tlbr
                elif hasattr(detections[det_idx], 'tlwh'):
                    t, l, w, h = detections[det_idx].tlwh
                    det_bbox = [t, l, t+w, l+h]
                else:
                    det_bbox = [0, 0, 0, 0]
            except:
                det_bbox = [0, 0, 0, 0]
            
            # Create spatial ID for detection (based on position)
            det_center = ((det_bbox[0] + det_bbox[2]) / 2, (det_bbox[1] + det_bbox[3]) / 2)
            det_spatial_id = self._get_spatial_id(det_center)
            
            # Record history
            feature = features[det_idx] if features and det_idx < len(features) else None
            self.track_history[track_id].append({
                'frame_id': frame_id,
                'bbox': det_bbox,
                'spatial_id': det_spatial_id,
                'feature': feature,
            })
            
            # Check for ID switch (same spatial location, different ID)
            if det_spatial_id in self.previous_assignments:
                prev_track_id = self.previous_assignments[det_spatial_id]
                if prev_track_id != track_id:
                    # ID switch detected!
                    switch_event = {
                        'frame_id': frame_id,
                        'det_spatial_id': det_spatial_id,
                        'det_bbox': det_bbox,
                        'old_track_id': prev_track_id,
                        'new_track_id': track_id,
                        'old_track_last_seen': self._get_last_frame(prev_track_id, frame_id),
                        'new_track_first_seen': self._get_first_frame(track_id),
                        'type': 'spatial_switch'
                    }
                    self.id_switches.append(switch_event)
            
            current_assignments[det_spatial_id] = track_id
        
        self.previous_assignments = current_assignments
    
    def _get_spatial_id(self, center, grid_size=50):
        """Create spatial ID based on position (discretized grid)."""
        grid_x = int(center[0] // grid_size)
        grid_y = int(center[1] // grid_size)
        return (grid_x, grid_y)
    
    def _get_last_frame(self, track_id, before_frame):
        """Get last frame where track was seen before given frame."""
        if track_id not in self.track_history:
            return None
        history = self.track_history[track_id]
        for entry in reversed(history):
            if entry['frame_id'] < before_frame:
                return entry['frame_id']
        return None
    
    def _get_first_frame(self, track_id):
        """Get first frame where track was seen."""
        if track_id not in self.track_history:
            return None
        return self.track_history[track_id][0]['frame_id']
    
    def get_summary(self):
        """Get ID switch summary."""
        # Count both spatial switches and new ID appearances
        total_id_issues = len(self.id_switches) + len(self.new_id_appearances)
        
        summary = {
            'total_id_switches': len(self.id_switches),
            'new_id_appearances': len(self.new_id_appearances),
            'total_id_issues': total_id_issues,
            'unique_tracks_total': len(self.all_track_ids),
        }
        
        if len(self.id_switches) > 0:
            summary['unique_tracks_in_switches'] = len(set(
                [s['old_track_id'] for s in self.id_switches] + 
                [s['new_track_id'] for s in self.id_switches]
            ))
        
        # ID switch frames
        all_problem_frames = []
        if len(self.id_switches) > 0:
            switch_frames = [s['frame_id'] for s in self.id_switches]
            all_problem_frames.extend(switch_frames)
            summary['spatial_switch_frames'] = sorted(set(switch_frames))
        
        if len(self.new_id_appearances) > 0:
            new_id_frames = [n['frame_id'] for n in self.new_id_appearances]
            all_problem_frames.extend(new_id_frames)
            summary['new_id_frames'] = sorted(set(new_id_frames))
        
        if len(all_problem_frames) > 0:
            summary['all_problem_frames'] = sorted(set(all_problem_frames))
            summary['avg_issues_per_frame'] = total_id_issues / len(set(all_problem_frames))
        
        return summary


class DeepSimilarityAnalyzer:
    """Main analyzer coordinating all sub-analyzers."""
    
    def __init__(self, tracker_name):
        self.tracker_name = tracker_name
        self.feature_analyzer = FeatureQualityAnalyzer()
        self.distance_analyzer = DistanceMatrixAnalyzer()
        self.gating_analyzer = GatingAnalyzer()
        self.conflict_analyzer = AssignmentConflictAnalyzer()
        self.id_switch_detector = IDSwitchDetector()
        
        self.frame_logs = []
    
    def analyze_frame(self, frame_id, cost_matrix, matches, unmatched_tracks, unmatched_dets,
                     tracks, detections, features=None, gating_mask=None, 
                     appearance_dist=None, motion_dist=None):
        """Analyze a single frame's matching process."""
        
        # Feature quality analysis
        if features is not None:
            for match in matches:
                # Safely extract indices
                if isinstance(match, (list, tuple)) and len(match) >= 2:
                    track_idx, det_idx = match[0], match[1]
                else:
                    continue
                
                # Validate indices
                if track_idx >= len(tracks) or det_idx >= len(features):
                    continue
                
                if features[det_idx] is not None:
                    track_id = getattr(tracks[track_idx], 'track_id', getattr(tracks[track_idx], 'id', track_idx))
                    self.feature_analyzer.add_track_feature(track_id, features[det_idx])
        
        # Distance matrix analysis
        self.distance_analyzer.add_frame_data(
            frame_id, cost_matrix, tracks, detections, matches,
            appearance_dist, motion_dist
        )
        
        # Gating analysis
        self.gating_analyzer.add_gating_info(frame_id, gating_mask, cost_matrix, tracks, detections)
        
        # Assignment conflict analysis
        self.conflict_analyzer.add_assignment_data(frame_id, cost_matrix, matches, tracks, detections)
        
        # ID switch detection
        self.id_switch_detector.add_frame_assignments(frame_id, matches, tracks, detections, features)
        
        # Frame log
        log_entry = {
            'frame_id': frame_id,
            'num_tracks': len(tracks),
            'num_detections': len(detections),
            'num_matches': len(matches),
            'num_unmatched_tracks': len(unmatched_tracks),
            'num_unmatched_dets': len(unmatched_dets),
            'cost_matrix_shape': cost_matrix.shape if cost_matrix is not None else None,
        }
        
        # Add cost matrix statistics to log
        if cost_matrix is not None and cost_matrix.size > 0:
            valid_costs = cost_matrix[cost_matrix != np.inf]
            if len(valid_costs) > 0:
                log_entry['cost_matrix_mean'] = float(np.mean(valid_costs))
                log_entry['cost_matrix_min'] = float(np.min(valid_costs))
                log_entry['cost_matrix_max'] = float(np.max(valid_costs))
        
        self.frame_logs.append(log_entry)
    
    def get_full_report(self):
        """Generate comprehensive analysis report."""
        report = {
            'tracker_name': self.tracker_name,
            'timestamp': datetime.now().isoformat(),
            'feature_quality': self.feature_analyzer.get_summary(),
            'distance_analysis': self.distance_analyzer.get_summary(),
            'gating_analysis': self.gating_analyzer.get_summary(),
            'conflict_analysis': self.conflict_analyzer.get_summary(),
            'id_switch_analysis': self.id_switch_detector.get_summary(),
            'frame_logs': self.frame_logs,
            'id_switch_events': self.id_switch_detector.id_switches,
            'new_id_events': self.id_switch_detector.new_id_appearances,
        }
        
        # Add problematic frames list
        problematic_frames = []
        for log in self.frame_logs:
            # Flag frames with high cost, many conflicts, or ID switches
            is_problematic = False
            reason = []
            
            if 'cost_matrix_mean' in log and log['cost_matrix_mean'] > 0.7:
                is_problematic = True
                reason.append('high_cost')
            
            if log['num_unmatched_tracks'] > 2:
                is_problematic = True
                reason.append('many_unmatched_tracks')
            
            if log['num_unmatched_dets'] > 2:
                is_problematic = True
                reason.append('many_unmatched_dets')
            
            # Check if this frame has ID switch
            for switch in self.id_switch_detector.id_switches:
                if switch['frame_id'] == log['frame_id']:
                    is_problematic = True
                    reason.append('id_switch')
                    break
            
            if is_problematic:
                problematic_frames.append({
                    'frame_id': log['frame_id'],
                    'reasons': reason,
                    'cost_mean': log.get('cost_matrix_mean', None),
                })
        
        report['problematic_frames'] = problematic_frames
        
        return report


def patch_tracker_for_deep_analysis(tracker, analyzer, tracker_type='strongsort'):
    """Patch tracker's internal methods to collect detailed analysis data."""
    
    # Patch the metric.distance method to capture appearance distance
    if hasattr(tracker.tracker, 'metric'):
        original_distance = tracker.tracker.metric.distance
        
        def instrumented_distance(features, targets):
            cost_matrix = original_distance(features, targets)
            # Store appearance distance for analysis
            if not hasattr(analyzer, '_temp_appearance_dist'):
                analyzer._temp_appearance_dist = {}
            frame_id = getattr(tracker, 'frame_idx', 0)
            analyzer._temp_appearance_dist[frame_id] = cost_matrix.copy()
            return cost_matrix
        
        tracker.tracker.metric.distance = instrumented_distance
    
    # Patch gating functions to capture gating masks
    from boxmot.trackers.strongsort.sort import linear_assignment
    
    original_gate_cost_matrix = linear_assignment.gate_cost_matrix
    
    def instrumented_gate_cost_matrix(cost_matrix, tracks, dets, track_indices, 
                                     detection_indices, mc_lambda=0.98):
        # Call original
        gated_matrix = original_gate_cost_matrix(
            cost_matrix, tracks, dets, track_indices, detection_indices, mc_lambda
        )
        
        # Store gating mask (where gated_matrix != cost_matrix or is inf)
        gating_mask = (gated_matrix == np.inf) & (cost_matrix != np.inf)
        
        if not hasattr(analyzer, '_temp_gating_mask'):
            analyzer._temp_gating_mask = {}
        frame_id = getattr(tracker, 'frame_idx', 0)
        analyzer._temp_gating_mask[frame_id] = gating_mask
        
        return gated_matrix
    
    linear_assignment.gate_cost_matrix = instrumented_gate_cost_matrix
    
    # Patch the _match method to collect final matching results
    original_match = tracker.tracker._match
    
    def instrumented_match(detections):
        # Call original
        try:
            matches, unmatched_tracks, unmatched_dets = original_match(detections)
        except TypeError:
            # Some trackers may not have certain parameters
            try:
                matches, unmatched_tracks, unmatched_dets = original_match(detections, False)
            except:
                return [], [], []
        
        # Collect data for analysis
        try:
            tracks = tracker.tracker.tracks
            frame_id = getattr(tracker, 'frame_idx', 0)
            
            # Extract stored data
            cost_matrix = None
            if hasattr(analyzer, '_temp_appearance_dist'):
                cost_matrix = analyzer._temp_appearance_dist.get(frame_id, None)
            
            gating_mask = None
            if hasattr(analyzer, '_temp_gating_mask'):
                gating_mask = analyzer._temp_gating_mask.get(frame_id, None)
            
            # Extract features from detections - SAFELY
            features = []
            for det in detections:
                try:
                    if hasattr(det, 'feat'):
                        features.append(det.feat)
                    elif hasattr(det, '__len__') and len(det) > 6:
                        features.append(det[-1])
                    else:
                        features.append(None)
                except:
                    features.append(None)
            
            # Convert matches to consistent format: list of [track_idx, det_idx]
            safe_matches = []
            for match in matches:
                try:
                    if isinstance(match, (list, tuple)) and len(match) >= 2:
                        safe_matches.append([int(match[0]), int(match[1])])
                    elif isinstance(match, np.ndarray) and match.size >= 2:
                        safe_matches.append([int(match[0]), int(match[1])])
                except:
                    continue
            
            # Analyze with safe matches
            analyzer.analyze_frame(
                frame_id, cost_matrix, safe_matches, unmatched_tracks, unmatched_dets,
                tracks, detections, features, gating_mask
            )
            
            # Clean up temp storage for this frame
            if hasattr(analyzer, '_temp_appearance_dist') and frame_id in analyzer._temp_appearance_dist:
                del analyzer._temp_appearance_dist[frame_id]
            if hasattr(analyzer, '_temp_gating_mask') and frame_id in analyzer._temp_gating_mask:
                del analyzer._temp_gating_mask[frame_id]
                
        except Exception as e:
            # Silently skip errors during analysis collection
            # Uncomment below for debugging:
            # import traceback
            # print(f"Error collecting analysis data for {tracker_type}: {e}")
            # traceback.print_exc()
            pass
        
        return matches, unmatched_tracks, unmatched_dets
    
    tracker.tracker._match = instrumented_match


def run_deep_analysis(video_path, model_weights, output_dir, max_frames=100):
    """Run deep similarity analysis on all three trackers."""
    
    print(f"Starting Deep Similarity Analysis")
    print(f"Video: {video_path}")
    print(f"Model: {model_weights}")
    print(f"Max frames: {max_frames if max_frames > 0 else 'ALL'}")
    print("="*80)
    
    # Setup
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_weights)
    reid_weights = Path("osnet_dcn_x0_5_endocv.pt")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize trackers
    trackers = {
        'StrongSort (XYAH)': StrongSort(
            reid_weights, torch.device(device), fp16=False,
            max_dist=0.95, max_iou_dist=0.95, max_age=300, half=False
        ),
        'StrongSortXYSR': StrongSortXYSR(
            reid_weights, torch.device(device), fp16=False,
            max_dist=0.95, max_iou_dist=0.95, max_age=300, half=False
        ),
        'TLUKF': StrongSortTLUKF(
            reid_weights, torch.device(device), fp16=False,
            max_dist=0.95, max_iou_dist=0.95, max_age=300, half=False
        ),
    }
    
    # Initialize analyzers
    analyzers = {
        name: DeepSimilarityAnalyzer(name) for name in trackers.keys()
    }
    
    # Patch trackers
    patch_tracker_for_deep_analysis(trackers['StrongSort (XYAH)'], analyzers['StrongSort (XYAH)'], 'strongsort')
    patch_tracker_for_deep_analysis(trackers['StrongSortXYSR'], analyzers['StrongSortXYSR'], 'xysr')
    patch_tracker_for_deep_analysis(trackers['TLUKF'], analyzers['TLUKF'], 'tlukf')
    
    # Process video
    cap = cv2.VideoCapture(str(video_path))
    frame_id = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if max_frames > 0 and frame_id >= max_frames:
            break
        
        # Detect
        results = model(frame, stream=True, verbose=False, conf=0.3, line_width=1)
        
        for dets in results:
            det_boxes = dets.boxes.data.to("cpu").numpy()
            
            if det_boxes.size > 0:
                # Update all trackers
                for name, tracker in trackers.items():
                    tracker.frame_idx = frame_id
                    try:
                        tracker.update(det_boxes, frame)
                    except Exception as e:
                        print(f"Error in {name} at frame {frame_id}: {e}")
        
        frame_id += 1
        if frame_id % 50 == 0:
            print(f"Processed {frame_id} frames...")
    
    cap.release()
    
    print(f"\n{'='*80}")
    print(f"Analysis complete! Total frames: {frame_id}")
    print(f"{'='*80}\n")
    
    # Generate reports
    reports = {}
    for name, analyzer in analyzers.items():
        print(f"\nGenerating report for {name}...")
        report = analyzer.get_full_report()
        reports[name] = report
        
        # Save individual report
        report_file = output_dir / f"deep_analysis_{name.replace(' ', '_')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        print(f"\n{name} Summary:")
        print(f"  Feature Quality:")
        print(f"    Avg feature norm: {report['feature_quality'].get('avg_feature_norm', 0):.4f}")
        print(f"  Distance Analysis:")
        if 'combined_mean' in report['distance_analysis']:
            print(f"    Combined distance mean: {report['distance_analysis']['combined_mean']:.4f}")
        if 'appearance_mean' in report['distance_analysis']:
            print(f"    Appearance distance mean: {report['distance_analysis']['appearance_mean']:.4f}")
        if 'motion_mean' in report['distance_analysis']:
            print(f"    Motion distance mean: {report['distance_analysis']['motion_mean']:.4f}")
        print(f"  Gating Analysis:")
        print(f"    Gating rate: {report['gating_analysis'].get('gating_rate', 0):.2%}")
        print(f"    Frames with gating: {report['gating_analysis'].get('frames_with_gating', 0)}")
        print(f"  Conflict Analysis:")
        print(f"    Total conflicts: {report['conflict_analysis'].get('total_conflicts', 0)}")
        print(f"  ID Switch Analysis:")
        print(f"    Spatial ID switches: {report['id_switch_analysis'].get('total_id_switches', 0)}")
        print(f"    New ID appearances: {report['id_switch_analysis'].get('new_id_appearances', 0)}")
        print(f"    Total ID issues: {report['id_switch_analysis'].get('total_id_issues', 0)}")
        print(f"    Total unique IDs: {report['id_switch_analysis'].get('unique_tracks_total', 0)}")
        if 'all_problem_frames' in report['id_switch_analysis']:
            problem_frames = report['id_switch_analysis']['all_problem_frames']
            if len(problem_frames) <= 10:
                print(f"    Problem frames: {problem_frames}")
            else:
                print(f"    Problem frames: {problem_frames[:5]} ... {problem_frames[-5:]} (total: {len(problem_frames)})")
    
    # Save comparison report
    comparison_file = output_dir / "deep_analysis_comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(reports, f, indent=2, default=str)
    
    # Generate comparison summary
    comparison_summary = {
        'total_frames': frame_id,
        'trackers': {}
    }
    
    for name, report in reports.items():
        comparison_summary['trackers'][name] = {
            'id_switches': report['id_switch_analysis'].get('total_id_switches', 0),
            'new_ids': report['id_switch_analysis'].get('new_id_appearances', 0),
            'total_id_issues': report['id_switch_analysis'].get('total_id_issues', 0),
            'unique_ids': report['id_switch_analysis'].get('unique_tracks_total', 0),
            'gating_rate': report['gating_analysis'].get('gating_rate', 0),
            'avg_cost': report['distance_analysis'].get('combined_mean', 0),
            'avg_matches_per_frame': report['distance_analysis'].get('avg_matches_per_frame', 0),
            'conflicts': report['conflict_analysis'].get('total_conflicts', 0),
            'problematic_frames_count': len(report.get('problematic_frames', [])),
        }
    
    # Save summary
    summary_file = output_dir / "comparison_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(comparison_summary, f, indent=2)
    
    # Print comparison table
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY (3 Trackers)")
    print(f"{'='*80}")
    print(f"{'Metric':<30} {'StrongSort':<15} {'XYSR':<15} {'TLUKF':<15}")
    print(f"{'-'*80}")
    
    metrics = [
        ('Spatial ID Switches', 'id_switches'),
        ('New ID Appearances', 'new_ids'),
        ('Total ID Issues', 'total_id_issues'),
        ('Unique Track IDs', 'unique_ids'),
        ('Gating Rate (%)', 'gating_rate', lambda x: f"{x*100:.1f}%"),
        ('Avg Cost', 'avg_cost', lambda x: f"{x:.4f}"),
        ('Avg Matches/Frame', 'avg_matches_per_frame', lambda x: f"{x:.2f}"),
        ('Conflicts', 'conflicts'),
        ('Problematic Frames', 'problematic_frames_count'),
    ]
    
    for metric_name, metric_key, *formatter in metrics:
        row = f"{metric_name:<30}"
        for name in ['StrongSort (XYAH)', 'StrongSortXYSR', 'TLUKF']:
            value = comparison_summary['trackers'][name][metric_key]
            if formatter:
                value_str = formatter[0](value)
            else:
                value_str = str(value)
            row += f"{value_str:<15}"
        print(row)
    
    print(f"{'='*80}\n")
    
    # Recommendations
    print("RECOMMENDATIONS:")
    
    # Find best tracker by ID switches
    best_tracker = min(
        comparison_summary['trackers'].items(),
        key=lambda x: x[1]['total_id_issues']
    )[0]
    print(f"✅ Least ID Issues: {best_tracker}")
    
    # Find best by gating rate
    best_motion = min(
        comparison_summary['trackers'].items(),
        key=lambda x: x[1]['gating_rate']
    )[0]
    print(f"✅ Best Motion Model: {best_motion}")
    
    # Find best by cost
    best_appearance = min(
        comparison_summary['trackers'].items(),
        key=lambda x: x[1]['avg_cost']
    )[0]
    print(f"✅ Best Appearance Matching: {best_appearance}")
    
    print(f"\n{'='*80}")
    print(f"Reports saved to: {output_dir}")
    print(f"  - Individual reports: deep_analysis_<tracker_name>.json")
    print(f"  - Comparison report: deep_analysis_comparison.json")
    print(f"  - Comparison summary: comparison_summary.json")
    print(f"{'='*80}")
    
    return reports


def main():
    parser = argparse.ArgumentParser(description="Deep Similarity Measurement Analysis")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--model_weights", type=str, required=True, help="Path to YOLO model weights")
    parser.add_argument("--output_dir", type=str, default="deep_analysis_results", help="Output directory")
    parser.add_argument("--max_frames", type=int, default=100, help="Maximum frames to process (0 for all)")
    
    args = parser.parse_args()
    
    run_deep_analysis(
        video_path=args.video_path,
        model_weights=args.model_weights,
        output_dir=args.output_dir,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()
