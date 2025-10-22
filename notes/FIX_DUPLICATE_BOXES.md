# FIX: Box Ảo và Box Thật Cùng Xuất Hiện (Duplicate ID)

## 🔴 Vấn đề

Trong video output, **cùng 1 frame xuất hiện 2 boxes với CÙNG ID**:
- **Box đỏ (thick, lớn)**: `2_Viem_thuc_quan, ID: 2` - Real detection
- **Box xám (thin, nhỏ)**: `Virtual 2_Viem_thuc_quan, ID: 2` - Virtual box

Cả 2 boxes **cùng tồn tại trong 1 frame** mặc dù đã có logic if/else trong `strongsort.py`.

## 🔍 Root Cause

### Nguyên nhân 1: Multiple Track Objects với cùng ID

Có khả năng `self.tracker.tracks` chứa **nhiều hơn 1 track object** với cùng `track.id`.

Ví dụ:
```python
self.tracker.tracks = [
    Track(id=2, time_since_update=0, ...),  # Real track, matched
    Track(id=2, time_since_update=5, ...),  # Old track, not deleted yet
]
```

Khi loop qua `self.tracker.tracks`:
```python
for track in self.tracker.tracks:
    if track.time_since_update < 1:
        # Output track 1 with ID=2, real box ✅
    else:
        # Output track 2 with ID=2, virtual box ❌
```

**Kết quả**: 2 boxes cùng ID=2 được output!

### Nguyên nhân 2: Tracker không xóa old tracks

TLUKF tracker có `max_age=300` (rất lớn!), nghĩa là track bị miss sẽ tồn tại 300 frames trước khi bị xóa.

Nếu có track cũ với `time_since_update > 0` (đang bị miss) và tracker tạo track MỚI với cùng ID do re-identification, sẽ có **2 tracks cùng ID** trong `self.tracker.tracks`.

## ✅ Giải pháp

### Fix 1: Deduplication trong `strongsort.py`

Thêm `seen_ids` set để track IDs đã output:

```python
# BEFORE ❌
outputs = []
for track in self.tracker.tracks:
    if track.time_since_update < 1:
        outputs.append([..., id, ...])  # Real
    else:
        outputs.append([..., id, ...])  # Virtual
# → Có thể output 2 boxes cùng ID!

# AFTER ✅
outputs = []
seen_ids = set()

for track in self.tracker.tracks:
    # Skip if already output this ID
    if track.id in seen_ids:
        continue
    
    if track.time_since_update < 1:
        outputs.append([..., id, ...])
        seen_ids.add(id)  # Mark as output
    else:
        outputs.append([..., id, ...])
        seen_ids.add(id)  # Mark as output
```

**Logic**:
1. Duyệt qua tracks theo thứ tự
2. Nếu ID đã được output → skip
3. Output track đầu tiên với mỗi ID
4. Add ID vào `seen_ids`

**Ưu tiên**: Track với `time_since_update < 1` (real) sẽ được output trước nếu nó xuất hiện trước trong list.

### Fix 2: Deduplication trong `pipeline.py`

Thêm filter trong pipeline để đảm bảo:

```python
# Group by ID and keep only HIGHEST confidence
unique_tracks = {}
for track in tracks:
    track_id = int(track[4])
    track_conf = track[5]
    
    if track_id not in unique_tracks:
        unique_tracks[track_id] = track
    else:
        # Keep track with HIGHER confidence (real > virtual)
        if track_conf > unique_tracks[track_id][5]:
            unique_tracks[track_id] = track

tracks = np.array(list(unique_tracks.values()))
```

**Logic**:
- Real boxes có `conf >= 0.6` (từ YOLO detection)
- Virtual boxes có `conf = 0.3` (fixed value)
- Giữ track với confidence CAO HƠN → real boxes được ưu tiên

### Fix 3: Debug Logging

Thêm logging để phát hiện duplicates:

```python
track_ids = [int(t[4]) for t in tracks]
if len(track_ids) != len(set(track_ids)):
    # Duplicates found!
    duplicates = [id for id, count in Counter(track_ids).items() if count > 1]
    print(f"⚠️ Frame {frame_id}: Duplicate IDs: {duplicates}")
    for dup_id in duplicates:
        dup_tracks = [t for t in tracks if int(t[4]) == dup_id]
        for i, t in enumerate(dup_tracks):
            print(f"  ID {dup_id} #{i+1}: conf={t[5]:.3f}")
```

## 📝 Files Modified

1. ✅ `boxmot/boxmot/trackers/strongsort/strongsort.py`
   - Added `seen_ids` set for deduplication
   - Skip tracks if ID already output
   - Mark IDs as output in both real and virtual branches

2. ✅ `osnet_dcn_pipeline_tlukf_xysr.py`
   - Added duplicate ID detection and logging
   - Added deduplication by confidence before drawing
   - Keep track with highest confidence per ID

## 🧪 Validation

### Test 1: Check for Duplicates in Output

```python
import pandas as pd
from collections import Counter

df = pd.read_csv('tracking_result.csv')

for frame in df['frame_idx'].unique():
    frame_data = df[df['frame_idx'] == frame]
    id_counts = Counter(frame_data['object_id'])
    
    duplicates = {id: count for id, count in id_counts.items() if count > 1}
    if duplicates:
        print(f"Frame {frame}: Duplicate IDs: {duplicates}")
        for dup_id in duplicates:
            dup_boxes = frame_data[frame_data['object_id'] == dup_id]
            print(dup_boxes[['object_id', 'notes', 'x1', 'y1', 'x2', 'y2']])
```

**Expected**: No duplicates (len(id_counts) == len(frame_data))

### Test 2: Verify Priority (Real > Virtual)

```python
df = pd.read_csv('tracking_result.csv')

# For each track, check if Virtual NEVER appears when Tracking exists
for track_id in df['object_id'].unique():
    track = df[df['object_id'] == track_id].sort_values('frame_idx')
    
    for i in range(len(track) - 1):
        curr_note = track.iloc[i]['notes']
        
        # Virtual should only appear in GAPS (no Tracking in same frame)
        if curr_note == 'Virtual':
            # Check: Is there NO real box in this frame?
            same_frame = df[(df['frame_idx'] == track.iloc[i]['frame_idx']) & 
                           (df['object_id'] == track_id)]
            has_real = (same_frame['notes'] == 'Tracking').any()
            
            if has_real:
                print(f"❌ ERROR: Track {track_id} has BOTH real and virtual in frame {track.iloc[i]['frame_idx']}")
            else:
                print(f"✅ OK: Track {track_id} virtual box in frame {track.iloc[i]['frame_idx']} (gap)")
```

**Expected**: No errors, all virtual boxes only appear in gaps.

## 📊 Before/After

### BEFORE ❌

```
Frame 100:
  ID 2 (conf=0.85): Real box [100, 100, 200, 200] ← From detection
  ID 2 (conf=0.30): Virtual box [95, 98, 195, 198] ← From old track
  
→ User sees 2 boxes on screen!
```

### AFTER ✅

```
Frame 100:
  Tracker returns 2 tracks with ID=2
  → seen_ids check: Skip 2nd track with ID=2
  → Output: Only 1 box (real, conf=0.85)
  
Frame 100:
  ID 2 (conf=0.85): Real box [100, 100, 200, 200] ← Only this one
  
→ User sees 1 box on screen ✅
```

## 💡 Why This Happens

### TLUKF max_age=300

```python
return StrongSortTLUKF(
    max_age=300,  # ← Track survives 300 frames after last detection!
    ...
)
```

Với `max_age=300`, track bị miss sẽ tồn tại **10 giây** (ở 30fps) trước khi bị xóa.

Nếu trong thời gian này:
1. Object biến mất khỏi view
2. Object khác xuất hiện
3. ReID nhầm → assign cùng ID

→ Có 2 tracks với cùng ID:
- Track cũ: `time_since_update=50` (virtual)
- Track mới: `time_since_update=0` (real)

### Solution: Deduplication ưu tiên Real

- Real boxes luôn có `time_since_update=0` (vừa matched)
- Virtual boxes có `time_since_update>0` (đã bị miss)
- Deduplication giữ track đầu tiên trong list
- Hoặc giữ track với confidence cao nhất (real > virtual)

## ✅ Summary

| Issue | Root Cause | Solution |
|-------|-----------|----------|
| Duplicate boxes same ID | Multiple track objects with same ID in tracker.tracks | Deduplication using `seen_ids` set in strongsort.py |
| Real + Virtual overlap | Old tracks not deleted, new track reuses ID | Skip tracks if ID already output (real has priority) |
| Pipeline draws all | No filtering before visualization | Add confidence-based deduplication in pipeline |

Cả 3 layers của defense:
1. ✅ **strongsort.py**: `seen_ids` set prevents duplicate output
2. ✅ **pipeline.py**: Confidence-based deduplication before drawing  
3. ✅ **Logging**: Debug detection for validation

Bây giờ chỉ có **1 box per ID per frame** được vẽ!
