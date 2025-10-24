"""
Inspect matching analysis results.
Đọc và hiển thị kết quả từ matching_analysis.json
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_results(json_path):
    """Load results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def print_summary(results):
    """Print summary statistics."""
    print("\n" + "="*60)
    print(f"📋 MATCHING ANALYSIS SUMMARY")
    print("="*60)
    print(f"Tracker: {results['tracker_type']}")
    print(f"Video: {Path(results['video_path']).name}")
    
    stats = results['statistics']
    print(f"\n📊 Statistics:")
    print(f"  Total Tracks: {stats['total_tracks']}")
    print(f"  Frames Processed: {stats['total_frames_processed']}")
    print(f"  ID Switches: {stats['id_switches']}")
    print(f"  Tracks Lost: {stats['tracks_lost']}")
    print(f"  Tracks Recovered: {stats['tracks_recovered']}")
    
    if 'virtual_boxes_created' in stats:
        print(f"  Virtual Boxes: {stats['virtual_boxes_created']}")


def print_track_details(results, top_n=5):
    """Print detailed track information."""
    id_history = results['id_history']
    
    # Sort by duration
    sorted_tracks = sorted(id_history.items(), 
                          key=lambda x: len(x[1]), 
                          reverse=True)
    
    print(f"\n🎯 Top {top_n} Longest Tracks:")
    for rank, (track_id, history) in enumerate(sorted_tracks[:top_n], 1):
        duration = len(history)
        first_frame = history[0]['frame']
        last_frame = history[-1]['frame']
        
        # Count status types
        status_counts = defaultdict(int)
        for entry in history:
            status_counts[entry['status']] += 1
        
        status_str = ", ".join([f"{k}: {v}" for k, v in status_counts.items()])
        
        print(f"  {rank}. Track {track_id}:")
        print(f"     - Duration: {duration} frames ({first_frame} → {last_frame})")
        print(f"     - Status: {status_str}")


def print_matching_events(results, max_events=10):
    """Print matching events."""
    events = results['matching_events']
    
    if len(events) == 0:
        print("\n✅ No critical matching events!")
        return
    
    print(f"\n⚠️  Critical Events ({len(events)} total):")
    
    # Group by type
    events_by_type = defaultdict(list)
    for event in events:
        events_by_type[event['type']].append(event)
    
    for event_type, type_events in events_by_type.items():
        print(f"\n  {event_type} ({len(type_events)} events):")
        
        for event in type_events[:max_events]:
            if event_type == 'ID_SWITCH':
                print(f"    Frame {event['frame']}: "
                      f"ID {event['old_id']} → {event['new_id']} "
                      f"(IoU: {event['iou']:.3f})")
            
            elif event_type == 'TRACK_LOST':
                print(f"    Frame {event['frame']}: "
                      f"ID {event['id']} lost "
                      f"(was {event['prev_status']})")
            
            elif event_type == 'TRACK_RECOVERED':
                print(f"    Frame {event['frame']}: "
                      f"ID {event['id']} recovered "
                      f"after {event['gap_frames']} frames")
            
            elif event_type == 'VIRTUAL_CREATED':
                print(f"    Frame {event['frame']}: "
                      f"Virtual box for ID {event['id']} "
                      f"(conf: {event['confidence']:.2f})")


def print_frame_analysis(results, frame_id):
    """Print analysis for specific frame."""
    frame_stats = results['frame_stats']
    
    if frame_id >= len(frame_stats):
        print(f"❌ Frame {frame_id} not found in results")
        return
    
    frame_info = frame_stats[frame_id]
    
    print(f"\n📊 Frame {frame_id} Analysis:")
    print(f"  Detections: {frame_info['num_detections']}")
    print(f"  Tracks: {frame_info['num_tracks']}")
    
    if len(frame_info['detections']) > 0:
        print(f"\n  Detections:")
        for i, det in enumerate(frame_info['detections'], 1):
            print(f"    {i}. bbox={det['bbox']}, conf={det['confidence']:.3f}")
    
    if len(frame_info['tracks']) > 0:
        print(f"\n  Tracks:")
        for track in frame_info['tracks']:
            print(f"    ID {track['id']}: "
                  f"bbox={track['bbox']}, "
                  f"conf={track['confidence']:.3f}, "
                  f"status={track['status']}")


def compare_results(json_paths):
    """Compare multiple tracker results."""
    results_list = []
    for json_path in json_paths:
        results = load_results(json_path)
        results_list.append(results)
    
    print("\n" + "="*80)
    print("📊 COMPARATIVE ANALYSIS")
    print("="*80)
    
    # Header
    trackers = [r['tracker_type'] for r in results_list]
    print(f"\n{'Metric':<30} | " + " | ".join([f"{t:>20}" for t in trackers]))
    print("-"*80)
    
    # Metrics
    metrics = [
        ('Total Tracks', 'total_tracks'),
        ('Frames Processed', 'total_frames_processed'),
        ('ID Switches', 'id_switches'),
        ('Tracks Lost', 'tracks_lost'),
        ('Tracks Recovered', 'tracks_recovered'),
        ('Virtual Boxes', 'virtual_boxes_created')
    ]
    
    for metric_name, metric_key in metrics:
        values = []
        for result in results_list:
            value = result['statistics'].get(metric_key, 0)
            values.append(f"{value:>20}")
        
        print(f"{metric_name:<30} | " + " | ".join(values))
    
    print("="*80)
    
    # Winner analysis
    print("\n🏆 Best Performer:")
    
    # ID switches (lower is better)
    id_switches = [r['statistics']['id_switches'] for r in results_list]
    best_switches = min(id_switches)
    best_tracker = trackers[id_switches.index(best_switches)]
    print(f"  ID Consistency: {best_tracker} ({best_switches} switches)")
    
    # Track recovery (higher is better)
    recoveries = [r['statistics'].get('tracks_recovered', 0) for r in results_list]
    best_recovery = max(recoveries)
    best_tracker = trackers[recoveries.index(best_recovery)]
    print(f"  Track Recovery: {best_tracker} ({best_recovery} recoveries)")


def main():
    parser = argparse.ArgumentParser(description="Inspect matching analysis results")
    parser.add_argument("json_path", type=str, nargs='+', 
                       help="Path to matching_analysis.json file(s)")
    parser.add_argument("--frame", type=int, default=None,
                       help="Show details for specific frame")
    parser.add_argument("--top_n", type=int, default=5,
                       help="Number of top tracks to show")
    parser.add_argument("--max_events", type=int, default=10,
                       help="Max events to show per type")
    
    args = parser.parse_args()
    
    if len(args.json_path) == 1:
        # Single result inspection
        json_path = Path(args.json_path[0])
        
        if not json_path.exists():
            print(f"❌ File not found: {json_path}")
            return
        
        results = load_results(json_path)
        
        # Print summary
        print_summary(results)
        
        # Print track details
        print_track_details(results, top_n=args.top_n)
        
        # Print matching events
        print_matching_events(results, max_events=args.max_events)
        
        # Print frame analysis if requested
        if args.frame is not None:
            print_frame_analysis(results, args.frame)
    
    else:
        # Compare multiple results
        json_paths = [Path(p) for p in args.json_path]
        
        # Check all exist
        for json_path in json_paths:
            if not json_path.exists():
                print(f"❌ File not found: {json_path}")
                return
        
        compare_results(json_paths)


if __name__ == "__main__":
    main()
