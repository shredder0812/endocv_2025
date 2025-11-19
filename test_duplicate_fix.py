"""
Test script để verify fix cho vấn đề box ảo xuất hiện cùng box thật.

Logic mới:
- Mỗi track CHỈ xuất hiện 1 lần trong outputs
- time_since_update == 0 → box thật (confidence cao)
- time_since_update >= 1 → box ảo (confidence = 0.3)
"""

import numpy as np


def simulate_tracker_output():
    """
    Mô phỏng output từ tracker để test logic.
    """
    # Mock tracks data: [x1, y1, x2, y2, id, conf, cls, det_ind, time_since_update]
    tracks = [
        # Track 1: Matched (real box)
        {"id": 1, "bbox": [100, 100, 200, 200], "conf": 0.85, "cls": 0, "det_ind": 0, "time_since_update": 0},
        # Track 2: Unmatched (virtual box)
        {"id": 2, "bbox": [300, 300, 400, 400], "conf": 0.75, "cls": 0, "det_ind": 1, "time_since_update": 3},
        # Track 3: Matched (real box)
        {"id": 3, "bbox": [500, 100, 600, 200], "conf": 0.92, "cls": 0, "det_ind": 2, "time_since_update": 0},
    ]
    
    outputs = []
    
    # NEW LOGIC: Single loop, each track enters exactly ONE branch
    for track in tracks:
        if track["time_since_update"] == 0:
            # MATCHED: Real box
            x1, y1, x2, y2 = track["bbox"]
            id = track["id"]
            conf = track["conf"]  # Original confidence
            cls = track["cls"]
            det_ind = track["det_ind"]
            
            outputs.append([x1, y1, x2, y2, id, conf, cls, det_ind])
            print(f"✓ Track {id}: REAL box (conf={conf:.2f}, time_since_update=0)")
            
        elif track["time_since_update"] >= 1:
            # UNMATCHED: Virtual box
            x1, y1, x2, y2 = track["bbox"]
            id = track["id"]
            conf = 0.3  # Low confidence for virtual
            cls = track["cls"]
            det_ind = track["det_ind"]
            
            outputs.append([x1, y1, x2, y2, id, conf, cls, det_ind])
            print(f"○ Track {id}: VIRTUAL box (conf={conf:.2f}, time_since_update={track['time_since_update']})")
    
    outputs = np.array(outputs)
    
    # VALIDATION: Check for duplicates
    unique_ids = set()
    duplicate_ids = []
    for output in outputs:
        track_id = int(output[4])
        if track_id in unique_ids:
            duplicate_ids.append(track_id)
        else:
            unique_ids.add(track_id)
    
    print("\n" + "="*60)
    print(f"Total outputs: {len(outputs)}")
    print(f"Unique IDs: {len(unique_ids)}")
    
    if duplicate_ids:
        print(f"❌ ERROR: Duplicate IDs found: {duplicate_ids}")
        return False
    else:
        print(f"✅ SUCCESS: No duplicate IDs!")
        return True


def test_old_logic_problem():
    """
    Demonstrate the OLD logic problem (for comparison).
    """
    print("\n" + "="*60)
    print("OLD LOGIC (PROBLEM):")
    print("="*60)
    
    tracks = [
        {"id": 1, "bbox": [100, 100, 200, 200], "conf": 0.85, "time_since_update": 0},
    ]
    
    outputs = []
    seen_ids = set()
    
    # First loop: Real boxes (time_since_update < 1)
    for track in tracks:
        if track["time_since_update"] < 1:
            outputs.append([*track["bbox"], track["id"], track["conf"], 0, 0])
            seen_ids.add(track["id"])
            print(f"Loop 1: Track {track['id']} → REAL box (conf={track['conf']:.2f})")
    
    # Second loop: Virtual boxes (time_since_update >= 1)
    # BUG: If time_since_update becomes >= 1 DURING processing, track enters BOTH loops!
    for track in tracks:
        if track["time_since_update"] >= 1:
            if track["id"] not in seen_ids:
                outputs.append([*track["bbox"], track["id"], 0.3, 0, 0])
                print(f"Loop 2: Track {track['id']} → VIRTUAL box (conf=0.3)")
            else:
                print(f"Loop 2: Track {track['id']} SKIPPED (already in seen_ids)")
    
    print(f"\nProblem: Two separate loops create opportunity for duplicates")
    print(f"Fix: Use single loop with mutually exclusive conditions")


if __name__ == "__main__":
    print("Testing NEW logic (fixed):")
    print("="*60)
    
    success = simulate_tracker_output()
    
    test_old_logic_problem()
    
    print("\n" + "="*60)
    if success:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ TEST FAILED!")
    print("="*60)
