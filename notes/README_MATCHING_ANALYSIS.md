# 🔬 ID Matching Analysis Tools

Công cụ phân tích và so sánh matching ID giữa 3 phương pháp tracking:

1. **StrongSort (XYAH)** - Baseline với XYAH Kalman Filter
2. **StrongSort XYSR** - XYSR Kalman Filter + Virtual Boxes
3. **TLUKF** - Transfer Learning UKF + Enhanced Similarity Measurement

---

## 📋 Tổng quan

### Các công cụ

1. **`analyze_matching.py`** - Phân tích chi tiết matching process
   - Thu thập statistics về ID switches, tracks lost/recovered
   - Detect các matching events quan trọng
   - Tạo visualization plots
   - Export JSON với đầy đủ thông tin

2. **`visualize_matching_comparison.py`** - So sánh trực quan
   - Tạo video side-by-side 3 trackers
   - Color-coded boxes (Strong/Weak/Virtual)
   - Real-time statistics

---

## 🚀 Sử dụng

### 1. Phân tích chi tiết một tracker

```bash
# Phân tích StrongSort
python analyze_matching.py \
    --video_path video_test_x/UTTQ/230411BVK004_Trim2.mp4 \
    --model_weights model_yolo/thucquan.pt \
    --tracker strongsort \
    --output_dir matching_analysis

# Phân tích StrongSort XYSR
python analyze_matching.py \
    --video_path video_test_x/UTTQ/230411BVK004_Trim2.mp4 \
    --model_weights model_yolo/thucquan.pt \
    --tracker strongsort_xysr \
    --output_dir matching_analysis

# Phân tích TLUKF
python analyze_matching.py \
    --video_path video_test_x/UTTQ/230411BVK004_Trim2.mp4 \
    --model_weights model_yolo/thucquan.pt \
    --tracker tlukf \
    --output_dir matching_analysis
```

### 2. So sánh TẤT CẢ trackers

```bash
python analyze_matching.py     --video_path video_test_x/UTTQ/230411BVK106_Trim2.mp4     --model_weights model_yolo/thucquan.pt     --tracker all     --output_dir matching_analysis
```

**Output:**
```
matching_analysis/
├── strongsort/
│   ├── matching_analysis.json      # Raw data
│   └── matching_visualization.png  # Plots
├── strongsort_xysr/
│   ├── matching_analysis.json
│   └── matching_visualization.png
├── tlukf/
│   ├── matching_analysis.json
│   └── matching_visualization.png
└── comparison_summary.json          # Comparative stats
```

### 3. Tạo video so sánh side-by-side

```bash
python visualize_matching_comparison.py \
    --video_path video_test_x/UTTQ/230411BVK004_Trim2.mp4 \
    --model_weights model_yolo/thucquan.pt \
    --output_path comparison_output.mp4
```

**Output:** Video 3 cột với color-coded boxes:
- **Green**: Strong/Real detection
- **Orange**: Weak detection (TLUKF only)
- **Gray**: Virtual box

### 4. Test với số frame giới hạn

```bash
# Chỉ phân tích 300 frames đầu
python analyze_matching.py \
    --video_path video_test_x/UTTQ/230411BVK004_Trim2.mp4 \
    --model_weights model_yolo/thucquan.pt \
    --tracker all \
    --max_frames 300
```

---

## 📊 Metrics được đo

### 1. Tracking Performance

- **Total Tracks**: Tổng số tracks được tạo
- **Track Duration**: Độ dài của mỗi track (frames)
- **Tracks per Frame**: Số lượng tracks active mỗi frame

### 2. Matching Quality

- **ID Switches**: Số lần ID bị đổi cho cùng một object
  - Thấp hơn = Tốt hơn
  - Đo bằng IoU > 0.7 giữa frames liên tiếp

- **Tracks Lost**: Số lần track bị mất
  - Track biến mất và không recover được

- **Tracks Recovered**: Số lần track được recover
  - Track xuất hiện lại sau khi bị mất
  - Cao hơn = Virtual box strategy hiệu quả

### 3. Virtual Box Usage (XYSR & TLUKF)

- **Virtual Boxes Created**: Số lượng virtual boxes
  - TLUKF: Expected cao hơn (uses conf 0.3)
  - XYSR: Trung bình (uses conf 0.45)

### 4. Detection Coverage

- **Detection vs Tracks**: So sánh số detections và tracks
  - Tracks > Detections: Virtual boxes đang hoạt động
  - Tracks < Detections: Nhiều detections không được track

---

## 📈 Đọc kết quả

### Example Output

```
📋 MATCHING ANALYSIS SUMMARY
============================================================
Tracker: tlukf
Video: 230411BVK004_Trim2.mp4

📊 Statistics:
  - Total Tracks: 8
  - Frames Processed: 450
  - ID Switches: 2
  - Tracks Lost: 3
  - Tracks Recovered: 2
  - Virtual Boxes: 127

🎯 Top 5 Longest Tracks:
  1. Track 1: 380 frames
  2. Track 2: 245 frames
  3. Track 3: 89 frames
  4. Track 5: 67 frames
  5. Track 7: 34 frames

⚠️  Critical Events:
  ID Switches:
    Frame 125: ID 2 → 5 (IoU: 0.823)
    Frame 289: ID 3 → 7 (IoU: 0.756)

  Track Recoveries:
    Frame 167: ID 1 recovered after 8 frames
    Frame 301: ID 2 recovered after 15 frames
============================================================
```

### Comparative Summary

```
📊 COMPARATIVE SUMMARY
============================================================

Metric                         |     StrongSort | StrongSort XYSR |          TLUKF
--------------------------------------------------------------------------------
Total Tracks                   |              9 |               8 |              8
ID Switches                    |              5 |               3 |              2
Tracks Lost                    |              7 |               4 |              3
Tracks Recovered               |              0 |               1 |              2
Virtual Boxes                  |              0 |              45 |            127
============================================================
```

**Phân tích:**
- **TLUKF**: Ít ID switches nhất (2) → Best ID consistency
- **TLUKF**: Nhiều virtual boxes nhất (127) → Active prediction
- **TLUKF**: Recover được nhiều tracks (2) → Virtual box strategy works

---

## 🎨 Visualization Plots

### 1. Tracks vs Detections
- Blue line: Number of tracks
- Red line: Number of detections
- **Gap analysis**: Khi tracks > detections → Virtual boxes active

### 2. ID Switches Over Time
- Red dots: ID switch events
- **Pattern analysis**: Clusters of switches → Problematic frames

### 3. Track Durations
- Bar chart: Duration of each track
- **Consistency analysis**: Longer tracks → Better tracking

### 4. Event Distribution
- Bar chart: Types of matching events
- **Strategy analysis**: Virtual vs Real event balance

---

## 🔍 Matching Events Explained

### 1. ID_SWITCH
```json
{
  "frame": 125,
  "type": "ID_SWITCH",
  "old_id": 2,
  "new_id": 5,
  "iou": 0.823,
  "bbox": [100, 200, 150, 250]
}
```
**Ý nghĩa**: Cùng một object nhưng ID thay đổi từ 2 → 5
**Nguyên nhân**: 
- Appearance matching failed
- Motion prediction sai
- Occlusion recovery không tốt

### 2. TRACK_LOST
```json
{
  "frame": 167,
  "type": "TRACK_LOST",
  "id": 3,
  "prev_status": "real"
}
```
**Ý nghĩa**: Track ID 3 biến mất
**Nguyên nhân**:
- Object ra khỏi frame
- Occlusion quá lâu (> max_age)
- Detection confidence thấp

### 3. TRACK_RECOVERED
```json
{
  "frame": 301,
  "type": "TRACK_RECOVERED",
  "id": 2,
  "gap_frames": 15
}
```
**Ý nghĩa**: Track ID 2 xuất hiện lại sau 15 frames
**Nguyên nhân**:
- Virtual box strategy thành công
- Appearance matching re-identified
- TLUKF transfer learning maintained ID

### 4. VIRTUAL_CREATED
```json
{
  "frame": 180,
  "type": "VIRTUAL_CREATED",
  "id": 5,
  "confidence": 0.30
}
```
**Ý nghĩa**: Virtual box được tạo cho track 5
**Nguyên nhân**:
- No real detection available (time_since_update ≥ 1)
- TLUKF/XYSR prediction used
- Maintaining ID consistency during gap

---

## 💡 Interpretation Guide

### So sánh 3 phương pháp

#### StrongSort (XYAH)
- **Pros**: Simple, fast, baseline
- **Cons**: 
  - Nhiều ID switches
  - Không handle gaps tốt
  - Không có virtual boxes
- **Use case**: Real-time tracking, ít occlusion

#### StrongSort XYSR (+Virtual)
- **Pros**: 
  - Virtual boxes giảm ID switches
  - Better gap handling
- **Cons**: 
  - Conf threshold 0.45 → miss weak detections
  - Virtual strategy đơn giản (linear prediction)
- **Use case**: Moderate occlusion scenarios

#### TLUKF (Transfer Learning + Virtual)
- **Pros**: 
  - Best ID consistency (ít switches nhất)
  - Enhanced similarity measurement (ALL boxes participate)
  - Dual-tracker với transfer learning
  - Virtual boxes từ non-linear TLUKF prediction
  - Conf 0.3 → catch weak detections
- **Cons**: 
  - Slower (more computation)
  - More virtual boxes (but more ID recovery)
- **Use case**: Complex scenarios, high occlusion, medical videos

---

## 🎯 Recommended Workflow

### Step 1: Quick comparison
```bash
python visualize_matching_comparison.py \
    --video_path your_video.mp4 \
    --model_weights your_model.pt \
    --max_frames 300
```
→ Xem video side-by-side để có cảm giác trực quan

### Step 2: Detailed analysis
```bash
python analyze_matching.py \
    --video_path your_video.mp4 \
    --model_weights your_model.pt \
    --tracker all
```
→ Xem số liệu chi tiết và plots

### Step 3: Evaluate metrics
- **ID Switches**: Lower is better
- **Track Recovery**: Higher is better (for XYSR/TLUKF)
- **Track Duration**: Longer is better
- **Virtual Boxes**: Balance needed (not too many, not too few)

### Step 4: Choose tracker
- **StrongSort**: If speed critical, simple scene
- **XYSR**: If moderate occlusion, need balance
- **TLUKF**: If accuracy critical, complex scene

---

## 📚 Output Files

### matching_analysis.json
```json
{
  "tracker_type": "tlukf",
  "video_path": "...",
  "statistics": {
    "total_tracks": 8,
    "id_switches": 2,
    "tracks_lost": 3,
    "tracks_recovered": 2,
    "virtual_boxes_created": 127
  },
  "id_history": {
    "1": [
      {"frame": 0, "bbox": [...], "confidence": 0.85, "status": "strong"},
      {"frame": 1, "bbox": [...], "confidence": 0.82, "status": "strong"},
      ...
    ]
  },
  "matching_events": [
    {"frame": 125, "type": "ID_SWITCH", ...},
    ...
  ],
  "frame_stats": [
    {"frame_id": 0, "num_detections": 2, "num_tracks": 2, ...},
    ...
  ]
}
```

### comparison_summary.json
```json
{
  "video": "your_video.mp4",
  "trackers": ["strongsort", "strongsort_xysr", "tlukf"],
  "results": {
    "strongsort": {
      "total_tracks": 9,
      "id_switches": 5,
      ...
    },
    "strongsort_xysr": {...},
    "tlukf": {...}
  }
}
```

---

## 🐛 Troubleshooting

### Issue: "AttributeError: 'KalmanFilterXYAH' object has no attribute 'x'"
**Solution**: Run `python osnet_dcn_pipeline_kf.py` first to ensure tracker fixes applied

### Issue: Video output blank
**Solution**: Check YOLO model matches video content (thucquan.pt for UTTQ, etc.)

### Issue: Slow processing
**Solution**: Use `--max_frames 300` for faster testing

### Issue: matplotlib not found
**Solution**: `pip install matplotlib`

---

## 📝 Tips

1. **Test short first**: Use `--max_frames 300` để test nhanh
2. **Compare same video**: Đảm bảo so sánh cùng một video
3. **Check confidence**: TLUKF dùng conf=0.3, StrongSort dùng 0.6
4. **Virtual boxes**: Nhiều không phải xấu - check recovery rate
5. **ID consistency**: Priority metric - lower switches = better

---

## 🔗 Related Files

- `osnet_dcn_pipeline_kf.py` - StrongSort baseline
- `osnet_dcn_pipeline_kf_xysr.py` - StrongSort XYSR
- `osnet_dcn_pipeline_tlukf_xysr.py` - TLUKF
- `README_TLUKF.md` - TLUKF technical documentation
- `boxmot/boxmot/trackers/strongsort/sort/tracker.py` - Enhanced matching logic

---

## 📧 Support

For issues or questions:
1. Check console output for errors
2. Review JSON output files
3. Compare with example outputs above
4. Check tracker initialization parameters

**Happy Tracking! 🎯**
