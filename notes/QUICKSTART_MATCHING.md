# 🚀 Quick Start - ID Matching Analysis

## Các lệnh thường dùng

### 1. Test nhanh (100 frames)
```bash
python quick_test_matching.py
```
→ Tạo `matching_test_quick/` với kết quả 3 trackers

### 2. So sánh đầy đủ
```bash
python analyze_matching.py \
    --video_path video_test_x/UTTQ/230411BVK004_Trim2.mp4 \
    --model_weights model_yolo/thucquan.pt \
    --tracker all
```

### 3. Tạo video comparison
```bash
python visualize_matching_comparison.py \
    --video_path video_test_x/UTTQ/230411BVK004_Trim2.mp4 \
    --model_weights model_yolo/thucquan.pt \
    --output_path comparison.mp4 \
    --max_frames 300
```

### 4. Xem kết quả đã có

#### Một tracker
```bash
python inspect_matching_results.py matching_analysis/tlukf/matching_analysis.json
```

#### So sánh nhiều trackers
```bash
python inspect_matching_results.py \
    matching_analysis/strongsort/matching_analysis.json \
    matching_analysis/strongsort_xysr/matching_analysis.json \
    matching_analysis/tlukf/matching_analysis.json
```

#### Xem chi tiết frame cụ thể
```bash
python inspect_matching_results.py \
    matching_analysis/tlukf/matching_analysis.json \
    --frame 125
```

---

## 📊 Đọc nhanh kết quả

### Console Output Example

```
📋 MATCHING ANALYSIS SUMMARY
============================================================
Tracker: tlukf
Video: 230411BVK004_Trim2.mp4

📊 Statistics:
  Total Tracks: 8
  Frames Processed: 450
  ID Switches: 2        ← Lower is better ✅
  Tracks Lost: 3
  Tracks Recovered: 2   ← Higher is better ✅
  Virtual Boxes: 127

🎯 Top 5 Longest Tracks:
  1. Track 1:
     - Duration: 380 frames (0 → 379)
     - Status: strong: 280, weak: 50, virtual: 50   ← Mix is good ✅

⚠️  Critical Events:
  ID_SWITCH (2 events):
    Frame 125: ID 2 → 5 (IoU: 0.823)   ← High IoU switch is bad ❌
    Frame 289: ID 3 → 7 (IoU: 0.756)

  TRACK_RECOVERED (2 events):
    Frame 167: ID 1 recovered after 8 frames   ← Good recovery ✅
    Frame 301: ID 2 recovered after 15 frames  ← Excellent recovery ✅
```

### Key Indicators

✅ **Good Signs:**
- ID Switches < 5
- Track Recovery > 0 (for XYSR/TLUKF)
- Long track durations
- Balanced status mix (strong/weak/virtual)

❌ **Bad Signs:**
- ID Switches > 10
- High IoU switches (> 0.7)
- Many short tracks
- Zero recoveries (for XYSR/TLUKF)

---

## 🎯 Comparison Table

| Metric | StrongSort | XYSR | TLUKF | Winner |
|--------|-----------|------|-------|--------|
| ID Switches | 5 | 3 | **2** | ✅ TLUKF |
| Tracks Lost | 7 | 4 | **3** | ✅ TLUKF |
| Tracks Recovered | 0 | 1 | **2** | ✅ TLUKF |
| Virtual Boxes | 0 | 45 | 127 | N/A |
| Speed | **Fast** | Medium | Slow | ✅ StrongSort |

**Recommendation:** TLUKF for accuracy, StrongSort for speed

---

## 📁 Output Structure

```
matching_analysis/
├── strongsort/
│   ├── matching_analysis.json        ← Raw data
│   └── matching_visualization.png    ← 4 plots
├── strongsort_xysr/
│   ├── matching_analysis.json
│   └── matching_visualization.png
├── tlukf/
│   ├── matching_analysis.json
│   └── matching_visualization.png
└── comparison_summary.json            ← Comparative stats
```

---

## 🎨 Visualization Guide

### Plot 1: Tracks vs Detections
- **Blue line**: Active tracks
- **Red line**: Detections
- **Gap (blue > red)**: Virtual boxes active ✅

### Plot 2: ID Switches
- **Red dots**: Switch events
- **Clusters**: Problematic periods ❌
- **Sparse**: Good consistency ✅

### Plot 3: Track Durations
- **Tall bars**: Long-lived tracks ✅
- **Short bars**: Fragmented tracking ❌

### Plot 4: Event Distribution
- **TRACK_RECOVERED**: Virtual box success ✅
- **ID_SWITCH**: Matching failures ❌
- **VIRTUAL_CREATED**: Prediction activity

---

## 💡 Tips

1. **Start small**: Test với `--max_frames 100` trước
2. **Compare same video**: Đảm bảo cùng video, cùng model
3. **Check thresholds**: 
   - StrongSort: conf=0.6
   - XYSR: conf=0.45
   - TLUKF: conf=0.3
4. **Analyze events**: Focus on ID_SWITCH với IoU cao
5. **Balance metrics**: Không chỉ xem một metric

---

## 🐛 Common Issues

**Issue**: "No module named 'matplotlib'"
```bash
pip install matplotlib
```

**Issue**: "Video not found"
```bash
# Check path
ls video_test_x/UTTQ/
```

**Issue**: Processing too slow
```bash
# Use max_frames
python analyze_matching.py ... --max_frames 300
```

**Issue**: Results look wrong
```bash
# Check model matches video
# UTTQ → thucquan.pt
# UTDD → daday.pt
# HTT → htt.pt
```

---

## 📞 Quick Help

```bash
# Show help
python analyze_matching.py --help
python visualize_matching_comparison.py --help
python inspect_matching_results.py --help

# List available videos
ls video_test_x/UTTQ/

# List available models
ls model_yolo/
```

---

## 🎓 Understanding Output

### ID Switch với IoU cao (>0.7) = BAD
```
Frame 125: ID 2 → 5 (IoU: 0.823)
```
→ Cùng object nhưng ID thay đổi = matching failed

### Track Recovered = GOOD
```
Frame 301: ID 2 recovered after 15 frames
```
→ Virtual box strategy thành công

### Virtual Boxes nhiều = GOOD (with context)
```
Virtual Boxes: 127
Tracks Recovered: 2
```
→ Virtual boxes giúp maintain ID

### Mix status balanced = GOOD
```
Status: strong: 280, weak: 50, virtual: 50
```
→ Real detections + predictions balanced

---

**Happy Analysis! 🔬**
