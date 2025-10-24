# 📊 Similarity Measurement Visualization Tool

## Tổng quan

Tool này giúp **kiểm tra chi tiết quá trình matching ID** thông qua **similarity measurement** cho 3 phương pháp tracking:

1. **StrongSort (XYAH)** - Baseline với Kalman Filter XYAH, conf ≥ 0.6
2. **StrongSort (XYSR)** - Kalman Filter XYSR + virtual boxes, conf ≥ 0.45
3. **TLUKF** - Transfer Learning UKF + enhanced matching, conf ≥ 0.3

### ✨ Features chính

- ✅ **Cost Matrix Visualization**: Xem chi tiết ma trận khoảng cách (appearance + motion)
- ✅ **Gating Mask**: Hiển thị các cặp track-detection hợp lệ/không hợp lệ
- ✅ **Matching Assignments**: Xem kết quả matching cuối cùng
- ✅ **Track-Detection Associations**: Vẽ đường kết nối giữa tracks và detections
- ✅ **Side-by-Side Comparison**: So sánh 3 phương pháp cùng lúc
- ✅ **Status Tracking**: Phân biệt strong/weak/virtual boxes

---

## 📦 Cài đặt dependencies

```bash
pip install matplotlib seaborn opencv-python ultralytics torch numpy
```

---

## 🚀 Cách sử dụng

### 1. Chạy visualization cơ bản

```bash
python visualize_similarity_matching.py \
    --video_path video_test_x/UTTQ/230411BVK106_Trim2.mp4 \
    --model_weights model_yolo/thucquan.pt \
    --output_dir similarity_analysis \
    --max_frames 200
```

### 2. Chạy với video khác

```bash
# Video UTTQ
python visualize_similarity_matching.py \
    --video_path video_test_x/UTTQ/230411BVK004_Trim2.mp4 \
    --model_weights model_yolo/thucquan.pt \
    --output_dir similarity_uttq_004

# Video UTDD
python visualize_similarity_matching.py \
    --video_path video_test_x/UTDD/some_video.mp4 \
    --model_weights model_yolo/daday.pt \
    --output_dir similarity_utdd
```

### 3. Xử lý toàn bộ video (không giới hạn frames)

```bash
python visualize_similarity_matching.py     --video_path video_test_x/UTTQ/230411BVK106_Trim2.mp4     --model_weights model_yolo/thucquan.pt    --max_frames 0
```

**Lưu ý**: Để tránh quá nhiều output, tool chỉ tạo visualization mỗi 10 frames.

---

## 📊 Hiểu các biểu đồ

### Layout tổng quan

```
┌─────────────────────────────────────────────────────────────┐
│              Similarity Measurement Analysis - Frame XXX     │
├──────────────┬──────────────┬──────────────┬───────────────┤
│  StrongSort  │    XYSR      │    TLUKF     │   Summary     │
│  (XYAH)      │              │              │  Comparison   │
├──────────────┼──────────────┼──────────────┤               │
│              │              │              │               │
│ Cost Matrix  │ Cost Matrix  │ Cost Matrix  │   Statistics  │
│              │              │              │   Table       │
├──────────────┼──────────────┼──────────────┤               │
│              │              │              │               │
│   Matching   │   Matching   │   Matching   │               │
│  on Frame    │  on Frame    │  on Frame    │               │
│              │              │              │               │
├──────────────┼──────────────┼──────────────┤               │
│              │              │              │               │
│  Distance    │  Distance    │  Distance    │               │
│ Distribution │ Distribution │ Distribution │               │
│              │              │              │               │
└──────────────┴──────────────┴──────────────┴───────────────┘
```

---

### 🔥 Biểu đồ 1: Cost Matrix (Hàng 1)

**Mục đích**: Hiển thị ma trận khoảng cách giữa tracks và detections

#### Cách đọc:

```
        D0    D1    D2    D3    ← Detections (cột)
T0    0.23  0.87  0.95  0.72
T1    0.91  0.15  0.89  0.94     ← Tracks (hàng)
T2    0.85  0.93  0.28  0.88
```

**Màu sắc**:
- 🟢 **Xanh lá (0.0-0.3)**: Khoảng cách RẤT GẦN → Rất có thể match ✅
- 🟡 **Vàng (0.3-0.5)**: Khoảng cách TRUNG BÌNH → Có thể match
- 🔴 **Đỏ (0.5-1.0)**: Khoảng cách XA → Không nên match ❌

**Ký hiệu**:
- 🔵 **Khung xanh dương**: Cặp được match (assignment)
- 🔴 **Vùng đỏ bên trái**: Track không match được (unmatched track)
- 🟠 **Vùng cam bên trên**: Detection không match được (unmatched detection)

**Track Status Labels**:
```
T0 (strong)   ← Real detection với conf ≥ 0.6
T1 (weak)     ← Real detection với 0.3 ≤ conf < 0.6 (chỉ TLUKF)
T2 (virtual)  ← Virtual box từ prediction (XYSR/TLUKF)
```

#### Ví dụ giải thích:

```
        D0    D1    D2
T0    [0.23] 0.87  0.95    ← T0 matched với D0 (distance=0.23, tốt!)
T1    0.91  [0.15] 0.89    ← T1 matched với D1 (distance=0.15, rất tốt!)
T2    0.85  0.93   0.28    ← T2 KHÔNG match (distance thấp nhất=0.28 nhưng D2 đã unmatched)
```

---

### 🎯 Biểu đồ 2: Track-Detection Matching (Hàng 2)

**Mục đích**: Hiển thị kết quả matching trên frame thực tế

#### Màu sắc boxes:

**Detections**:
- 🟢 **Green dashed**: Detection từ YOLO (dotted line)

**Tracks** (màu phụ thuộc vào status):
- 🔵 **Blue solid**: Strong track (conf ≥ 0.6)
- 🟠 **Orange solid**: Weak track (0.3 ≤ conf < 0.6) - CHỈ có ở TLUKF
- ⚫ **Gray solid**: Virtual box (prediction khi không có detection)

**Matching Lines**:
- 🔵 **Blue line**: Đường nối từ track center → detection center
  - Line xuất hiện = Track matched với Detection thành công ✅

#### Ví dụ visualization:

```
┌─────────────────────────────────┐
│                                 │
│  ┌──────┐                       │
│  │ ID:1 │  ────────────→  ┌───┐ │  ← Blue line = Match successful
│  └──────┘  (Blue)         └───┘ │
│   (Blue box = Strong)   (Green = Detection)
│                                 │
│  ┌──────┐                       │
│  │ ID:2 │  (No line)            │  ← No line = Virtual box (no detection)
│  └──────┘                       │
│   (Gray = Virtual)              │
│                                 │
└─────────────────────────────────┘
```

**Insight**:
- Nhiều blue lines = Nhiều matches thành công
- Gray boxes không có line = Virtual boxes duy trì tracking khi detection mất
- Orange boxes (weak) = TLUKF accept low-confidence detections

---

### 📈 Biểu đồ 3: Distance Distribution (Hàng 3)

**Mục đích**: So sánh phân phối khoảng cách của tất cả cặp vs các cặp matched

#### Thành phần:

1. **Gray histogram**: ALL possible track-detection pairs
   - Tất cả khoảng cách trong cost matrix
   - Cho thấy "không gian" matching

2. **Blue histogram**: MATCHED pairs only
   - Chỉ các cặp được chọn để match
   - Nên tập trung ở vùng khoảng cách thấp

3. **Red dashed line**: Threshold (thường 0.5)
   - Ngưỡng quyết định có match hay không

#### Cách đọc:

**✅ GOOD matching**:
```
Count
  │     ████
  │     ████        ██  ← Gray (all pairs)
  │ ████████    ████
  │ ████████████████
  │ ████████████████ ██  ← Blue (matched) tập trung bên TRÁI
  └──────────────────────→ Distance
    0.0      0.5      1.0
            ↑ Threshold
```
→ Matched pairs có distance thấp (< 0.5), matching tốt!

**❌ BAD matching**:
```
Count
  │             ████
  │         ████████  ← Blue scattered, xa threshold
  │     ████████████
  │ ████████████████  ← Gray everywhere
  └──────────────────────→ Distance
    0.0      0.5      1.0
            ↑ Threshold
```
→ Matched pairs có distance cao (> 0.5), matching kém!

**Statistics box** (góc phải trên):
```
Matches: 3          ← Số cặp matched
Mean: 0.187         ← Trung bình distance (càng thấp càng tốt)
Min: 0.150          ← Distance thấp nhất
Max: 0.230          ← Distance cao nhất
```

**Mục tiêu**:
- Mean < 0.3: Excellent ✅
- Mean 0.3-0.5: Good 👍
- Mean > 0.5: Poor ⚠️

---

### 📋 Biểu đồ 4: Summary Comparison (Cột 4)

**Mục đích**: So sánh thống kê 3 phương pháp cùng lúc

#### Bảng thống kê:

```
┌─────────────────┬────────────┬────────────┬────────────┐
│ Metric          │ StrongSort │    XYSR    │   TLUKF    │
├─────────────────┼────────────┼────────────┼────────────┤
│ Total Tracks    │     2      │     2      │     3      │
│ ├─ Strong       │     2      │     2      │     1      │
│ ├─ Weak         │     0      │     0      │     1      │ ← CHỈ TLUKF
│ └─ Virtual      │     0      │     0      │     1      │ ← XYSR/TLUKF
├─────────────────┼────────────┼────────────┼────────────┤
│ Detections      │     2      │     2      │     3      │
│ Matches         │     2      │     2      │     2      │
│ Unmatched Trks  │     0      │     0      │     1      │ ← Virtual
│ Unmatched Dets  │     0      │     0      │     1      │ ← Low-conf
└─────────────────┴────────────┴────────────┴────────────┘
```

#### Cách phân tích:

**1. Total Tracks**
- TLUKF > XYSR > StrongSort
- TLUKF có nhiều tracks nhất vì accept conf thấp

**2. Track Status Breakdown**
- **Strong**: Real detections conf ≥ 0.6
- **Weak**: Real detections 0.3 ≤ conf < 0.6 (CHỈ TLUKF có)
- **Virtual**: Predicted boxes khi không có detection

**3. Matches vs Unmatched**
- **High Matches**: Matching tốt
- **High Unmatched Tracks**: Nhiều virtual boxes hoặc tracks bị lost
- **High Unmatched Dets**: Nhiều new objects hoặc false positives

#### So sánh patterns:

**Pattern 1: StrongSort conservative**
```
Total Tracks: 2
Strong: 2, Weak: 0, Virtual: 0
Matches: 2, Unmatched: 0
→ Chỉ track high-confidence, không có virtual
```

**Pattern 2: XYSR moderate**
```
Total Tracks: 2
Strong: 2, Weak: 0, Virtual: 0 (hoặc 1)
Matches: 2, Unmatched: 0-1
→ Accept medium-confidence, có virtual khi cần
```

**Pattern 3: TLUKF aggressive**
```
Total Tracks: 3
Strong: 1, Weak: 1, Virtual: 1
Matches: 2, Unmatched: 1
→ Accept low-confidence, nhiều virtual, track nhiều nhất
```

---

## 🎯 Kịch bản phân tích

### Scenario 1: So sánh quality matching

**Câu hỏi**: Phương pháp nào có matching chất lượng cao nhất?

**Cách phân tích**:

1. **Xem Distance Distribution** (hàng 3):
   - So sánh Mean distance của matched pairs
   - Phương pháp nào có Mean thấp nhất = matching tốt nhất

2. **Xem Cost Matrix** (hàng 1):
   - Count số ô có màu xanh lá (distance < 0.3) được match
   - Nhiều = matching chính xác

**Ví dụ kết quả**:
```
StrongSort: Mean=0.18 → Excellent (nhưng match ít)
XYSR:       Mean=0.25 → Good (match vừa)
TLUKF:      Mean=0.32 → OK (match nhiều nhưng quality trung bình)
```

**Kết luận**: StrongSort có quality cao nhất nhưng bỏ qua nhiều objects

---

### Scenario 2: Kiểm tra virtual box strategy

**Câu hỏi**: Virtual boxes có giúp duy trì tracking không?

**Cách phân tích**:

1. **Xem Summary Table** (cột 4):
   - Count số Virtual tracks
   - XYSR/TLUKF phải có > 0

2. **Xem Track-Detection Matching** (hàng 2):
   - Count số gray boxes (virtual)
   - Xem vị trí: có ở khu vực objects thực không?

3. **So sánh với StrongSort**:
   - StrongSort: Virtual=0 → tracks bị lost
   - XYSR/TLUKF: Virtual>0 → tracks được maintain

**Ví dụ**:
```
Frame 100:
StrongSort: Tracks=1 (mất 1 object)
TLUKF:      Tracks=2 (1 strong + 1 virtual) → Maintain tracking ✅
```

---

### Scenario 3: Weak detection acceptance (TLUKF only)

**Câu hỏi**: TLUKF có lợi dụng được low-confidence detections không?

**Cách phân tích**:

1. **Xem Summary Table**:
   - Count số Weak tracks (chỉ TLUKF có)
   - Weak > 0 → TLUKF đang dùng conf 0.3-0.6

2. **Xem Track-Detection Matching**:
   - Count số orange boxes
   - Vị trí có hợp lý không?

3. **Xem Cost Matrix**:
   - Check distance của weak tracks
   - Distance thấp → weak detection vẫn match tốt

**Ví dụ**:
```
TLUKF Frame 50:
Strong: 1 (conf=0.72)
Weak: 1 (conf=0.48) → Orange box, matched với distance=0.21 ✅
Virtual: 0

→ Weak detection được accept và match tốt!
```

---

## 📁 Output Structure

```
similarity_analysis/
├── similarity_frame_0000.png      ← Frame 0
├── similarity_frame_0010.png      ← Frame 10
├── similarity_frame_0020.png      ← Frame 20
├── ...
├── similarity_frame_0190.png      ← Frame 190
└── summary.json                   ← Metadata
```

### summary.json format:

```json
{
  "video": "video_test_x/UTTQ/230411BVK106_Trim2.mp4",
  "total_frames": 190,
  "visualized_frames": [0, 10, 20, 30, ..., 190],
  "output_dir": "similarity_analysis"
}
```

---

## 🔍 Tips & Best Practices

### 1. Chọn frames quan trọng

**Vấn đề**: Quá nhiều visualizations

**Giải pháp**:
```bash
# Chỉ visualize 100 frames đầu
python visualize_similarity_matching.py ... --max_frames 100

# Hoặc edit code để visualize frames cụ thể
# Line 227: if frame_id % 10 == 0:
# → Thay bằng: if frame_id in [50, 100, 150, 200]:
```

### 2. Tìm frames có ID switches

**Cách làm**:
1. Chạy `analyze_matching.py` trước để tìm ID switches
2. Xem JSON output, lấy frame numbers có ID_SWITCH
3. Chỉnh sửa code để visualize đúng frames đó

**Ví dụ**:
```python
# Thay line 227
interesting_frames = [125, 289]  # Từ analyze_matching.py
if frame_id in interesting_frames:
    self.visualize_frame(frame_id, frame, show_plots=True)
```

### 3. So sánh cost matrix patterns

**Cách đọc patterns**:

**Pattern A: Diagonal dominance** (GOOD ✅)
```
Cost Matrix:
     D0   D1   D2
T0  [0.1] 0.8  0.9   ← Diagonal có giá trị thấp
T1   0.9 [0.2] 0.8   ← Track i match với Detection i
T2   0.8  0.9 [0.1]  ← Consistent matching
```
→ Stable tracking, IDs consistent

**Pattern B: Scattered low values** (RISKY ⚠️)
```
Cost Matrix:
     D0   D1   D2
T0  [0.3] 0.2  0.8   ← Multiple low values per row
T1   0.2 [0.3] 0.1   ← Ambiguous matches
T2   0.8  0.1 [0.4]  ← Risk of ID switches
```
→ Multiple candidates, potential confusion

**Pattern C: High values everywhere** (BAD ❌)
```
Cost Matrix:
     D0   D1   D2
T0   0.8  0.9  0.7   ← All high distances
T1   0.9  0.8  0.9   ← No good matches
T2   0.7  0.9  0.8   ← Objects changed significantly
```
→ Objects moved too much, tracking may fail

### 4. Giải thích unmatched patterns

**Unmatched Tracks (Red on left)**:
```
T0  [✗] 0.8  0.9  0.7  ← Red mark = không match được
T1      0.2  0.1  0.9
T2      0.9  0.8  0.3
```
**Nguyên nhân**:
- Object biến mất khỏi frame
- Object bị occlusion
- Detection quality quá thấp
- Tracker prediction sai

**Giải pháp**: Virtual box sẽ maintain ID

**Unmatched Detections (Orange on top)**:
```
        D0   D1   D2
                 [✗]  ← Orange mark = detection không match
T0     0.2  0.1  0.9
T1     0.9  0.8  0.8
```
**Nguyên nhân**:
- New object xuất hiện
- False positive từ YOLO
- Object được track bỏ qua (conf thấp ở StrongSort)

**Giải pháp**: Tạo track mới hoặc ignore

---

## 🐛 Troubleshooting

### Issue 1: "No cost matrix" trong plots

**Nguyên nhân**: Frame không có tracks hoặc detections

**Giải pháp**:
- Bình thường, skip frame đó
- Nếu tất cả frames đều vậy → check model weights, video path

### Issue 2: Visualization quá nhỏ, không đọc được

**Giải pháp**: Tăng DPI
```python
# Line 304: plt.savefig(output_path, dpi=150, ...)
# Thay bằng:
plt.savefig(output_path, dpi=300, bbox_inches='tight')
```

### Issue 3: Màu sắc không rõ

**Giải pháp**: Thay colormap
```python
# Line 290: sns.heatmap(..., cmap='RdYlGn_r', ...)
# Thay bằng:
sns.heatmap(..., cmap='coolwarm', ...)  # Blue-Red
# Hoặc:
sns.heatmap(..., cmap='viridis', ...)   # Yellow-Purple
```

### Issue 4: Quá chậm, xử lý lâu

**Nguyên nhân**: Visualize quá nhiều frames

**Giải pháp**:
```python
# Line 227: if frame_id % 10 == 0:
# Thay bằng:
if frame_id % 20 == 0:  # Visualize mỗi 20 frames thay vì 10
```

### Issue 5: Out of memory

**Nguyên nhân**: Video quá dài, matplotlib cache quá nhiều

**Giải pháp**:
- Giảm `max_frames`
- Giảm DPI (150 → 100)
- Clear matplotlib cache:
```python
# Thêm vào cuối _create_visualization():
plt.close('all')
import gc
gc.collect()
```

---

## 📚 Advanced Usage

### Customize visualization layout

Edit `_create_visualization()` method:

```python
# Thay đổi grid layout
gs = GridSpec(4, 3, ...)  # 4 rows, 3 columns

# Thêm subplot mới
ax_new = fig.add_subplot(gs[3, :])  # Row 4, all columns
self._plot_custom(ax_new, data)
```

### Export data to JSON

Thêm vào cuối `visualize_frame()`:

```python
# Export matching data
data_export = {
    'frame_id': frame_id,
    'cost_matrices': {},
    'assignments': {}
}

for name in self.trackers.keys():
    data = collectors[name].frame_data
    data_export['cost_matrices'][name] = data['cost_matrix'].tolist() if data['cost_matrix'] is not None else []
    data_export['assignments'][name] = data['assignments'].tolist() if data['assignments'] is not None else []

# Save
json_path = self.output_dir / f'matching_data_{frame_id:04d}.json'
with open(json_path, 'w') as f:
    json.dump(data_export, f, indent=2)
```

### Create comparison video

Sử dụng tool khác: `visualize_matching_comparison.py` để tạo video

---

## 🎓 Interpretation Guide

### Khi nào chọn StrongSort?

✅ **Chọn khi**:
- Cần speed cao, real-time
- YOLO detection quality tốt (conf > 0.6 luôn có)
- Ít occlusion, objects không biến mất
- Cost matrix: Diagonal dominance, low distances

⚠️ **Không chọn khi**:
- Detection bị đứt đoạn (gaps)
- Objects nhỏ, conf thấp
- Nhiều occlusion

### Khi nào chọn XYSR?

✅ **Chọn khi**:
- Balance giữa speed và accuracy
- Detection quality trung bình (conf 0.45-0.6)
- Có một số gaps nhỏ cần lấp
- Cost matrix: Good matches với occasional virtuals

⚠️ **Không chọn khi**:
- Cần track objects conf < 0.45
- Gaps quá dài (> 10 frames)

### Khi nào chọn TLUKF?

✅ **Chọn khi**:
- Detection quality thấp (conf 0.3-0.6 phổ biến)
- Nhiều gaps, occlusions
- Cần ID consistency tuyệt đối
- Accept trade-off: slower but more accurate
- Cost matrix: Mix of strong/weak/virtual, more complex

⚠️ **Không chọn khi**:
- Cần real-time processing
- Video chất lượng tốt, ít gaps

---

## 📊 Metrics Summary

| Metric | Good Range | Interpretation |
|--------|-----------|----------------|
| **Mean Distance (matched)** | < 0.3 | Matching quality |
| **Matches / Tracks ratio** | > 0.8 | Matching success rate |
| **Virtual / Total ratio** | 0.2-0.4 | Virtual box usage (XYSR/TLUKF) |
| **Weak / Total ratio** | 0.1-0.3 | Low-conf detection usage (TLUKF) |
| **Unmatched Tracks** | < 20% | Lost tracks ratio |

---

## 🚀 Quick Reference

```bash
# Basic usage
python visualize_similarity_matching.py \
    --video_path VIDEO.mp4 \
    --model_weights MODEL.pt

# Custom output
python visualize_similarity_matching.py \
    --video_path VIDEO.mp4 \
    --model_weights MODEL.pt \
    --output_dir my_analysis \
    --max_frames 100

# Full video processing
python visualize_similarity_matching.py \
    --video_path VIDEO.mp4 \
    --model_weights MODEL.pt \
    --max_frames 0
```

**Output**: `similarity_analysis/similarity_frame_XXXX.png` files

---

## 💡 Key Takeaways

1. **Cost Matrix = Heart of matching**: Giá trị thấp (xanh) = good match
2. **Blue boxes on frame = Strong tracks**: Real detections conf cao
3. **Orange boxes = Weak tracks**: TLUKF only, conf 0.3-0.6
4. **Gray boxes = Virtual tracks**: Predictions khi không có detection
5. **Blue lines = Successful matches**: Connect track → detection
6. **Distance Distribution**: Matched pairs nên tập trung bên trái (< 0.5)
7. **Summary Table**: So sánh 3 methods side-by-side

**Happy Analyzing! 🔬**
