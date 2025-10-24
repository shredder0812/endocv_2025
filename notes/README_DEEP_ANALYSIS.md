# Deep Similarity Measurement Analysis Tool

## Mục Đích

Tool này phân tích **toàn bộ luồng xử lý similarity measurement** của 3 phương pháp tracking:
1. **StrongSort (XYAH)** - Kalman Filter với XYAH state
2. **StrongSortXYSR** - Kalman Filter với XYSR state  
3. **TLUKF** - Transfer Learning Unscented Kalman Filter

Mục tiêu: **Hiểu sâu** về:
- Similarity measurement đang hoạt động thế nào
- Gặp vấn đề gì trong quá trình matching
- Tại sao bị ID switch
- Toàn bộ luồng từ detection → matching → assignment

---

## Các Thành Phần Phân Tích

### 1. Feature Quality Analysis
**Mục đích**: Kiểm tra chất lượng ReID feature vectors

**Metrics**:
- `avg_feature_norm`: Độ lớn trung bình của feature vector
- `track_consistency`: Độ tương đồng của features trong cùng 1 track qua các frame
  * `mean_similarity`: Cosine similarity trung bình (càng cao càng tốt, lý tưởng > 0.8)
  * `std_similarity`: Độ dao động (càng thấp càng stable)
  * `min_similarity`: Similarity thấp nhất (cảnh báo nếu < 0.5)

**Ý nghĩa**:
- Feature norm cao + consistency cao → ReID model hoạt động tốt
- Feature norm thấp → Feature không discriminative
- Consistency thấp → Appearance thay đổi nhiều (lighting, occlusion, deformation)

**Cách đọc**:
```json
{
  "avg_feature_norm": 12.5,  // Tốt nếu > 10
  "track_consistency": {
    "track_1": {
      "mean_similarity": 0.85,  // Tốt! Feature ổn định
      "std_similarity": 0.05,   // Tốt! Dao động thấp
      "min_similarity": 0.72    // OK, không có frame nào quá khác biệt
    }
  }
}
```

---

### 2. Distance Matrix Analysis
**Mục đích**: Phân tích khoảng cách appearance (cosine) và motion (Mahalanobis)

**Metrics**:
- `appearance_mean/std/min/max`: Thống kê khoảng cách appearance
  * Mean < 0.3: Matching tốt
  * Mean 0.3-0.7: Matching trung bình
  * Mean > 0.7: Matching kém
  
- `combined_mean/std/min/max`: Cost matrix sau khi kết hợp appearance + motion
  * Giá trị này được dùng cho Hungarian algorithm
  
- `avg_matches_per_frame`: Số lượng track-detection được match trung bình

**Ý nghĩa**:
- Appearance distance cao → ReID không nhận diện được (lighting change, occlusion)
- Combined distance cao nhưng vẫn match → Dựa vào motion model (Kalman prediction)

**Cách đọc**:
```json
{
  "appearance_mean": 0.45,  // Moderate - ReID không hoàn hảo
  "combined_mean": 0.35,    // Tốt hơn nhờ motion model
  "avg_matches_per_frame": 0.98  // Gần như luôn match được
}
```

---

### 3. Gating Analysis
**Mục đích**: Phân tích khi nào gating loại bỏ matches (do motion model không phù hợp)

**Metrics**:
- `gating_rate`: Tỷ lệ track-detection pairs bị gating loại bỏ
  * < 10%: Tốt - motion model accurate
  * 10-30%: Moderate - motion model đôi khi sai
  * > 30%: Kém - motion model không reliable
  
- `frames_with_gating`: Số frame có gating events
- `gated_distance_mean`: Distance trung bình của các pairs bị gating

**Ý nghĩa**:
- Gating rate cao → Motion model prediction xa thực tế (non-linear motion, camera shake)
- Gated distance thấp → False rejection (motion model quá strict)

**Cách đọc**:
```json
{
  "gating_rate": 0.15,  // 15% pairs bị gating - acceptable
  "frames_with_gating": 45,  // 45/100 frames có gating
  "gated_distance_mean": 0.65  // Distance trung bình của pairs bị từ chối
}
```

---

### 4. Assignment Conflict Analysis
**Mục đích**: Phát hiện khi nhiều tracks cùng tranh giành 1 detection

**Metrics**:
- `total_conflicts`: Tổng số conflicts
- `avg_competitors_per_conflict`: Số tracks trung bình tranh giành 1 detection
- `frames_with_conflicts`: Số frame có conflicts

**Ý nghĩa**:
- Conflicts nhiều → Tracks quá gần nhau (crowded scene, fragmentation)
- Avg competitors cao → Quá nhiều tracks cho 1 object (ID fragmentation)

**Potential ID Switch Causes**:
- Track A đang match detection D
- Track B cũng muốn match D (distance gần bằng A)
- Nếu B có priority cao hơn (theo cascade) → **ID switch**

**Cách đọc**:
```json
{
  "total_conflicts": 12,  // 12 lần có nhiều tracks tranh giành
  "avg_competitors_per_conflict": 2.3,  // Trung bình 2-3 tracks cạnh tranh
  "frames_with_conflicts": 8  // Xảy ra ở 8 frames
}
```

---

### 5. ID Switch Detection
**Mục đích**: Phát hiện chính xác frame nào bị ID switch và nguyên nhân

**Metrics**:
- `total_id_switches`: Tổng số lần đổi ID
- `id_switch_frames`: Danh sách frame số bị ID switch
- `unique_tracks_involved`: Số tracks bị ảnh hưởng

**Detailed Events**:
Mỗi ID switch event chứa:
```json
{
  "frame_id": 250,
  "det_spatial_id": [5, 10],  // Grid position của detection
  "det_bbox": [100, 200, 150, 250],
  "old_track_id": 1,  // Track cũ đang theo dõi object này
  "new_track_id": 3,  // Track mới chiếm object này
  "old_track_last_seen": 248,  // Frame cuối track cũ thấy object
  "new_track_first_seen": 240  // Frame đầu track mới xuất hiện
}
```

**Root Cause Analysis**:
Từ thông tin trên, có thể suy luận:
- **Occlusion Recovery**: `old_track_last_seen` xa `frame_id` → Track cũ mất object, track mới recover
- **Fragmentation**: `new_track_first_seen` gần `frame_id` → Track mới vừa tạo, chiếm luôn object
- **Swap**: `old_track_last_seen` = `frame_id - 1` → 2 tracks hoán đổi ngay lập tức

---

## Workflow Phân Tích

### Bước 1: Chạy Tool
```bash
python deep_similarity_analysis.py     --video_path video_test_x/UTTQ/230411BVK106_Trim2.mp4     --model_weights model_yolo/thucquan.pt     --max_frames 403    --output_dir deep_analysis_results
```

**Output**:
```
deep_analysis_results/
├── deep_analysis_StrongSort_(XYAH).json  # Report cho StrongSort
├── deep_analysis_StrongSortXYSR.json     # Report cho XYSR
├── deep_analysis_TLUKF.json              # Report cho TLUKF
└── deep_analysis_comparison.json         # So sánh 3 trackers
```

### Bước 2: Đọc Summary
Tool sẽ in ra console:

```
StrongSort (XYAH) Summary:
  Feature Quality:
    Avg feature norm: 12.456
  Distance Analysis:
    Combined distance mean: 0.4231
    Appearance distance mean: 0.5123
  Gating Analysis:
    Gating rate: 12.34%
    Frames with gating: 45
  Conflict Analysis:
    Total conflicts: 23
  ID Switch Analysis:
    Total ID switches: 5
    ID switch frames: [45, 89, 123, 201, 267]
```

### Bước 3: Phân Tích Chi Tiết

#### A. Tìm Root Cause của ID Switch

1. Mở file JSON tương ứng
2. Tìm `id_switch_events`
3. Xem frame bị ID switch (ví dụ frame 250):

```json
{
  "frame_id": 250,
  "old_track_id": 1,
  "new_track_id": 3,
  "old_track_last_seen": 248
}
```

4. Kiểm tra `frame_logs` tại frame 250:

```json
{
  "frame_id": 250,
  "num_tracks": 2,
  "num_detections": 1,
  "cost_matrix_mean": 0.85,  // ⚠️ Distance rất cao!
  "cost_matrix_min": 0.82
}
```

5. **Phân tích**:
   - Cost matrix mean = 0.85 → Appearance matching THẤT BẠI
   - Num tracks (2) > num detections (1) → Conflicts!
   - Track 1 và Track 3 cùng tranh giành 1 detection
   - Track 3 được chọn (có thể do cascade priority hoặc distance nhỉnh hơn)
   - → **Root cause**: Appearance feature kém + conflict → ID switch

#### B. Kiểm Tra Feature Quality

Tìm track bị ID switch (track 1):

```json
{
  "track_consistency": {
    "1": {
      "mean_similarity": 0.62,  // ⚠️ Thấp! (<0.7)
      "min_similarity": 0.45    // ⚠️ Rất thấp!
    }
  }
}
```

**Kết luận**: Feature của track 1 không stable → Appearance matching không đáng tin cậy

#### C. So Sánh 3 Trackers

Mở `deep_analysis_comparison.json`:

```json
{
  "StrongSort (XYAH)": {
    "id_switch_analysis": {
      "total_id_switches": 5
    }
  },
  "StrongSortXYSR": {
    "id_switch_analysis": {
      "total_id_switches": 3  // Ít hơn!
    }
  },
  "TLUKF": {
    "id_switch_analysis": {
      "total_id_switches": 1  // Tốt nhất!
    }
  }
}
```

**Kết luận**: TLUKF có ít ID switch nhất → Multi-hypothesis + XYSR state hiệu quả hơn

---

## Interpretation Guide

### Scenario 1: High Appearance Distance
**Dấu hiệu**:
- `appearance_mean > 0.7`
- `gating_rate < 10%` (motion model vẫn tốt)
- `total_id_switches` cao

**Nguyên nhân**:
- ReID model không phù hợp với domain (endoscopy specific)
- Lighting change mạnh trong video
- Occlusion nhiều

**Giải pháp**:
- Re-train ReID model với endoscopy data
- Tăng weight của motion model
- Sử dụng TLUKF (robust hơn với appearance failure)

---

### Scenario 2: High Gating Rate
**Dấu hiệu**:
- `gating_rate > 30%`
- `appearance_mean < 0.5` (appearance tốt)
- `total_id_switches` cao

**Nguyên nhân**:
- Motion model không phù hợp (XYAH vs XYSR vs TLUKF)
- Camera motion compensation thất bại
- Non-linear motion (scope moving)

**Giải pháp**:
- Chuyển sang XYSR (handle scale change)
- Chuyển sang TLUKF (UKF handle non-linear motion)
- Cải thiện CMC (camera motion compensation)

---

### Scenario 3: Many Conflicts
**Dấu hiệu**:
- `total_conflicts` cao
- `avg_competitors_per_conflict > 3`
- `total_id_switches` cao

**Nguyên nhân**:
- Fragmentation: 1 object → nhiều tracks
- Crowded scene: Nhiều objects gần nhau
- Detection threshold quá thấp

**Giải pháp**:
- Tăng detection confidence threshold
- Tăng `n_init` (số frame cần confirm track)
- Apply NMS aggressive hơn

---

## Advanced Analysis: Frame-by-Frame Debugging

### Bước 1: Tìm Frame Problematic
Từ summary, xác định frame có vấn đề (ví dụ frame 250 - ID switch)

### Bước 2: Extract Frame Data
```python
import json

# Load report
with open('deep_analysis_results/deep_analysis_TLUKF.json') as f:
    report = json.load(f)

# Find frame 250
frame_log = next(f for f in report['frame_logs'] if f['frame_id'] == 250)
print(frame_log)

# Find ID switch event at frame 250
id_switch = next(e for e in report['id_switch_events'] if e['frame_id'] == 250)
print(id_switch)
```

### Bước 3: Visualize (Manual)
1. Chạy visualization tool từ session trước:
```bash
python visualize_similarity_matching.py \
    --video_path video_test_x/UTTQ/230411BVK106_Trim2.mp4 \
    --model_weights model_yolo/thucquan.pt \
    --start_frame 245 \
    --max_frames 10  # Visualize 245-255
```

2. Xem frame 250 trong `similarity_analysis/frame_250.png`
3. Đối chiếu với data từ JSON report

---

## Comparison với Tool Trước

### `visualize_similarity_matching.py` (Visual Tool)
- **Mục đích**: Xem cost matrix, matching, distance distribution qua visualization
- **Output**: PNG images với heatmaps và boxes
- **Ưu điểm**: Trực quan, dễ hiểu, phù hợp cho presentation
- **Nhược điểm**: Không có metrics chi tiết, khó so sánh numbers

### `deep_similarity_analysis.py` (Analysis Tool - MỚI)
- **Mục đích**: Phân tích chi tiết toàn bộ luồng matching với metrics
- **Output**: JSON reports với đầy đủ thống kê
- **Ưu điểm**: 
  * Feature quality analysis
  * Gating analysis  
  * ID switch detection với root cause
  * Assignment conflict tracking
  * Dễ dàng so sánh 3 trackers qua numbers
- **Nhược điểm**: Không có visualization trực quan

### Workflow Kết Hợp
1. **Bước 1**: Chạy `deep_similarity_analysis.py` để có overview
2. **Bước 2**: Xác định frames problematic từ JSON report
3. **Bước 3**: Chạy `visualize_similarity_matching.py` trên những frames đó để xem chi tiết visual
4. **Bước 4**: Đối chiếu metrics (JSON) với visualization (PNG) để hiểu root cause

---

## Expected Output Example

### Terminal Output
```
Starting Deep Similarity Analysis
Video: video_test_x/UTTQ/230411BVK106_Trim2.mp4
Model: model_yolo/thucquan.pt
Max frames: 200
================================================================================
Processed 50 frames...
Processed 100 frames...
Processed 150 frames...
Processed 200 frames...

================================================================================
Analysis complete! Total frames: 200
================================================================================

Generating report for StrongSort (XYAH)...

StrongSort (XYAH) Summary:
  Feature Quality:
    Avg feature norm: 12.456
  Distance Analysis:
    Combined distance mean: 0.4231
    Appearance distance mean: 0.5123
    Motion distance mean: 0.3512
  Gating Analysis:
    Gating rate: 12.34%
    Frames with gating: 45
  Conflict Analysis:
    Total conflicts: 23
  ID Switch Analysis:
    Total ID switches: 5
    ID switch frames: [45, 89, 123, 201, 267]

Generating report for StrongSortXYSR...
[...]

Generating report for TLUKF...
[...]

================================================================================
Reports saved to: deep_analysis_results
  - Individual reports: deep_analysis_<tracker_name>.json
  - Comparison report: deep_analysis_comparison.json
================================================================================
```

### JSON Report Structure
```json
{
  "tracker_name": "TLUKF",
  "timestamp": "2025-10-25T10:30:00",
  
  "feature_quality": {
    "avg_feature_norm": 12.456,
    "std_feature_norm": 1.234,
    "track_consistency": {
      "1": {
        "mean_similarity": 0.85,
        "std_similarity": 0.05,
        "min_similarity": 0.72,
        "num_samples": 45
      }
    }
  },
  
  "distance_analysis": {
    "appearance_mean": 0.5123,
    "appearance_std": 0.234,
    "combined_mean": 0.4231,
    "total_frames": 200,
    "avg_matches_per_frame": 0.98
  },
  
  "gating_analysis": {
    "total_pairs_checked": 1250,
    "total_gated": 154,
    "gating_rate": 0.1232,
    "frames_with_gating": 45,
    "gated_distance_mean": 0.65
  },
  
  "conflict_analysis": {
    "total_conflicts": 23,
    "avg_competitors_per_conflict": 2.3,
    "frames_with_conflicts": 15
  },
  
  "id_switch_analysis": {
    "total_id_switches": 5,
    "unique_tracks_involved": 8,
    "id_switch_frames": [45, 89, 123, 201, 267]
  },
  
  "frame_logs": [
    {
      "frame_id": 0,
      "num_tracks": 1,
      "num_detections": 1,
      "num_matches": 1,
      "num_unmatched_tracks": 0,
      "num_unmatched_dets": 0,
      "cost_matrix_shape": [1, 1],
      "cost_matrix_mean": 0.123,
      "cost_matrix_min": 0.123,
      "cost_matrix_max": 0.123
    }
  ],
  
  "id_switch_events": [
    {
      "frame_id": 250,
      "det_spatial_id": [5, 10],
      "det_bbox": [100, 200, 150, 250],
      "old_track_id": 1,
      "new_track_id": 3,
      "old_track_last_seen": 248,
      "new_track_first_seen": 240
    }
  ]
}
```

---

## Next Steps

Sau khi có report từ tool này, bạn có thể:

1. **Identify Weakest Component**:
   - Feature quality kém? → Re-train ReID
   - Gating rate cao? → Improve motion model
   - Conflicts nhiều? → Tuning detection threshold

2. **Choose Best Tracker**:
   - So sánh `total_id_switches` của 3 trackers
   - Tracker có ít ID switch nhất → Best choice

3. **Targeted Improvements**:
   - Tìm frames có `cost_matrix_mean > 0.7` → Investigate why
   - Tìm tracks có `mean_similarity < 0.6` → Check appearance features
   - Tìm ID switch events → Understand root causes

4. **Visualize Problematic Frames**:
   - Dùng `visualize_similarity_matching.py` trên những frames từ `id_switch_frames`
   - Xem chính xác điều gì xảy ra trong cost matrix

---

## Troubleshooting

### 1. Tool chạy rất chậm
**Nguyên nhân**: Monkey patching tạo overhead

**Giải pháp**:
- Giảm `--max_frames` xuống 100-200
- Hoặc chạy trên subset của video (--start_frame)

### 2. Cost matrix luôn None trong report
**Nguyên nhân**: Patching không hoạt động đúng

**Kiểm tra**:
```python
# Xem _temp_appearance_dist có được tạo không
print(hasattr(analyzer, '_temp_appearance_dist'))
```

### 3. ID switch events trống
**Nguyên nhân**: Không có ID switch trong video đoạn test

**Giải pháp**:
- Tăng `--max_frames` để chạy toàn bộ video
- Hoặc chọn video khác có nhiều ID switch hơn

---

## Summary

Tool này cung cấp **phân tích toàn diện** về:
1. ✅ Feature quality → Hiểu ReID model hoạt động tốt không
2. ✅ Distance matrices → Appearance vs Motion contribution
3. ✅ Gating decisions → Khi nào motion model reject matches
4. ✅ Assignment conflicts → Nhiều tracks tranh giành 1 detection
5. ✅ ID switches → Phát hiện chính xác và root cause analysis

Kết hợp với `visualize_similarity_matching.py` để có cả **metrics** (JSON) và **visualization** (PNG).

🎯 **Mục tiêu cuối cùng**: Hiểu rõ **TẠI SAO** tracker bị ID switch và **LÀM SAO** để fix!
```bash
# Bước 1: Chạy deep analysis (có metrics chi tiết)
python deep_similarity_analysis.py     --video_path video_test_x/UTTQ/230411BVK106_Trim2.mp4     --model_weights model_yolo/thucquan.pt     --max_frames 200

# Bước 2: Xem comparison summary
cat deep_analysis_results/comparison_summary.json

# Bước 3: Tìm problematic frames từ JSON
python -c "
import json
with open('deep_analysis_results/deep_analysis_TLUKF.json') as f:
    data = json.load(f)
    print('Problematic frames:', [f['frame_id'] for f in data['problematic_frames']])
    print('ID switch frames:', data['id_switch_analysis']['id_switch_frames'])
"

# Bước 4: Visualize những frames đó
python visualize_similarity_matching.py \
    --video_path video_test_x/UTTQ/230411BVK106_Trim2.mp4 \
    --model_weights model_yolo/thucquan.pt \
    --frames_to_save 45,89,123,201,267  # Từ ID switch frames
```