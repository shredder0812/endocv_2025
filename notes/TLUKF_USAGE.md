# Transfer Learning Unscented Kalman Filter (TLUKF) - Hướng dẫn sử dụng

## Tổng quan

TLUKF là phương pháp tracking tiên tiến sử dụng **dual-tracker architecture** (Source + Primary) với **transfer learning** để cải thiện khả năng theo dõi đối tượng trong video nội soi y tế.

### Ưu điểm chính

1. **Chuyển động phi tuyến**: Sử dụng UKF thay vì Kalman Filter tuyến tính
2. **Transfer Learning**: Primary tracker học từ Source tracker khi mất detection
3. **Virtual Trajectory**: Box ảo được sinh ra từ predictions phi tuyến, không phải nội suy tuyến tính
4. **ID Consistency**: Duy trì ID nhất quán giữa box thật và box ảo

---

## Kiến trúc Dual-Tracker

### Source Tracker (Teacher)
- **Vai trò**: Học motion model "sạch" từ detections chất lượng cao
- **Cập nhật**: Chỉ với detections có `confidence >= 0.8`
- **Mục đích**: Cung cấp kiến thức đáng tin cậy cho Primary tracker

### Primary Tracker (Student)
- **Vai trò**: Tracking chính, xuất output cuối cùng
- **Cập nhật**: 
  - Với TẤT CẢ detections (kể cả confidence thấp)
  - Học từ Source tracker khi KHÔNG có detection (transfer learning)
- **Mục đích**: Kết hợp detections thực tế với kiến thức từ Source

---

## Cách hoạt động

### Frame có detection (time_since_update = 0)
```
1. Detection → Primary Tracker update
2. Nếu confidence >= 0.8 → Source Tracker update
3. Output: Box THẬT với confidence từ detection
```

### Frame KHÔNG có detection (time_since_update >= 1)
```
1. Kiểm tra Source tracker có dữ liệu mới (< 5 frames)?
2. Nếu có → Transfer Learning:
   - Primary học từ Source's prediction (eta_pred, P_eta)
   - Sequential Bayesian update
3. Output: Box ẢO với confidence = 0.3
```

---

## Virtual Boxes - Box ảo

### Khi nào xuất hiện?
- **CHỈ KHI** track không được match với detection ở frame hiện tại
- Fill gaps trong quá trình tracking
- Tương đương với "tracking low-confidence detections"

### Đặc điểm
- **Chuyển động**: Phi tuyến (UKF predictions), không phải linear interpolation
- **Confidence**: Cố định = 0.3 (để phân biệt với detections thật)
- **ID**: Giữ nguyên ID của track gốc
- **Màu sắc**: Xám (128, 128, 128) với border mỏng

### Visualization
```python
# Box thật
- Màu: Theo class (màu sáng)
- Border: 5px
- Label: "{class_name}, ID: {id}"
- Notes: "Tracking"

# Box ảo
- Màu: Xám (128, 128, 128)
- Border: 2px
- Label: "Virtual {class_name}, ID: {id}"
- Notes: "Virtual"
```

---

## Cách sử dụng

### 1. Chạy pipeline với TLUKF

```bash
python osnet_dcn_pipeline_tlukf_xysr.py --tracker_type tlukf
```

### 2. So sánh với XYSR

```bash
# XYSR (baseline)
python osnet_dcn_pipeline_tlukf_xysr.py --tracker_type xysr

# TLUKF (advanced)
python osnet_dcn_pipeline_tlukf_xysr.py --tracker_type tlukf
```

### 3. Tùy chỉnh output directory

```bash
python osnet_dcn_pipeline_tlukf_xysr.py \
    --tracker_type tlukf \
    --output_dir content/runs_tlukf_output
```

---

## Cấu hình Parameters

### High-confidence threshold
```python
high_conf_threshold = 0.8  # trong TrackTLUKF.__init__
```
- Detections >= 0.8: Cập nhật cả Source lẫn Primary
- Detections < 0.8: Chỉ cập nhật Primary

### Process Noise Q - Ổn định kích thước box
Theo paper TL-UKF, velocity của aspect ratio và height cần nhiễu CỰC THẤP:
```python
self.Q = np.diag([
    0.5, 0.5,        # Position (x, y) - có thể thay đổi
    1e-6, 1e-6,      # Aspect ratio & height - GẦN KHÔNG ĐỔI
    1.0, 1.0,        # Position velocities - cho phép di chuyển
    1e-8, 1e-8       # Size velocities - CỰC THẤP (box ổn định)
]) * dt
```
- **Nguyên tắc**: Kích thước box (a, h) thay đổi RẤT CHẬM
- **Lợi ích**: Tránh box ảo phóng to/thu nhỏ phi lý

### Static Scene Detection (Video Pause Handling)
```python
position_threshold = 1.0  # pixels
static_frame_count >= 3   # frames to detect pause
```
- **Phát hiện**: Position change < 1 pixel qua 3 frames
- **Xử lý**: Dampen velocities → box ảo giữ nguyên vị trí
- **Reset**: Khi có detection mới

### Source freshness window
```python
gap_since_hq = frame_id - last_high_quality_frame
if gap_since_hq > 5:  # trong apply_transfer_learning
    # Source quá cũ, bỏ qua transfer learning
```
- Source chỉ được tin cậy trong 5 frames
- Tránh sử dụng motion model cũ

### Virtual box confidence
```python
conf = 0.3  # trong StrongSortTLUKF.update
```
- Để phân biệt với detections thật (>= 0.6)

---

## Output Format

### Tracking result CSV
```csv
timestamp_hms,timestamp_hmsf,frame_idx,fps,object_cls,object_idx,object_id,notes,frame_height,frame_width,scale_height,scale_width,x1,y1,x2,y2,center_x,center_y
```

**Notes field**:
- `"Tracking"`: Box thật từ detection
- `"Virtual"`: Box ảo từ TLUKF prediction

### MOT format
```
<frame_id>,<track_id>,<x1>,<y1>,<width>,<height>,<conf>,-1,-1,-1
```

---

## Kiểm tra kết quả

### 1. Đếm số virtual boxes
```bash
# Trong file CSV
grep ",Virtual," tracking_result.csv | wc -l
```

### 2. Phân tích ID consistency
```python
import pandas as pd

df = pd.read_csv('tracking_result.csv')

# Tracks có cả real và virtual boxes
for track_id in df['object_id'].unique():
    track_data = df[df['object_id'] == track_id]
    real_count = (track_data['notes'] == 'Tracking').sum()
    virtual_count = (track_data['notes'] == 'Virtual').sum()
    print(f"Track {track_id}: {real_count} real, {virtual_count} virtual")
```

### 3. Visualize trajectories
```python
import cv2
import pandas as pd

df = pd.read_csv('tracking_result.csv')

for track_id in df['object_id'].unique():
    track_data = df[df['object_id'] == track_id].sort_values('frame_idx')
    
    # Plot trajectory
    for i in range(len(track_data) - 1):
        pt1 = (int(track_data.iloc[i]['center_x']), 
               int(track_data.iloc[i]['center_y']))
        pt2 = (int(track_data.iloc[i+1]['center_x']), 
               int(track_data.iloc[i+1]['center_y']))
        
        # Màu khác cho real vs virtual
        color = (0, 255, 0) if track_data.iloc[i]['notes'] == 'Tracking' else (128, 128, 128)
        cv2.line(img, pt1, pt2, color, 2)
```

---

## Troubleshooting

### Box ảo xuất hiện cùng box thật?
- **Không nên xảy ra** với implementation hiện tại
- Kiểm tra: `time_since_update` phải >= 1 cho virtual boxes

### ID không nhất quán?
- **ĐÃ ĐƯỢC SỬA**: Virtual boxes kế thừa ID từ track gốc
- ID được duy trì xuyên suốt lifetime của track (real → virtual → real)
- Kiểm tra CSV: Cùng `object_id` cho cả "Tracking" và "Virtual" notes

### Box ảo thay đổi kích thước phi lý?
- **ĐÃ ĐƯỢC SỬA**: Process noise Q đã được điều chỉnh theo TL-UKF paper
- Aspect ratio velocity (va): 1e-8 (cực thấp)
- Height velocity (vh): 1e-8 (cực thấp)
- **Kết quả**: Box ảo giữ kích thước ổn định, chỉ di chuyển vị trí

### Video pause - Box ảo "trôi đi"?
- **ĐÃ ĐƯỢC SỬA**: Static scene detection
- **Cơ chế**: 
  - Phát hiện scene tĩnh (position change < 1px qua 3 frames)
  - Tự động dampen velocities → box giữ nguyên vị trí
  - Reset khi có detection mới
- **Lợi ích**: Box ảo không "trôi" trong video pause

### Source tracker không cập nhật?
- Kiểm tra detections có `confidence >= 0.8`
- Xem log `last_high_quality_frame` trong debug

### Virtual boxes không xuất hiện?
- Kiểm tra Source tracker có dữ liệu mới (< 5 frames)
- Verify validation checks trong `apply_transfer_learning()`

---

## Technical Details

### UKF vs KF
- **KF**: Linear motion assumption (constant velocity)
- **UKF**: Non-linear motion via sigma points
- **Advantage**: Better handling of curved trajectories

### Transfer Learning Process
```python
# Sequential Bayesian Update
P_primary ← f(P_primary, P_source)  # Covariance fusion
x_primary ← weighted_fusion(x_primary, x_source)  # State fusion
```

### State Vector
```python
x = [x, y, a, h, vx, vy, va, vh]
# x, y: center position
# a: aspect ratio (width/height)
# h: height
# vx, vy, va, vh: velocities
```

---

## Performance Comparison

| Metric | XYSR | TLUKF |
|--------|------|-------|
| Motion Model | Linear | Non-linear |
| Gap Handling | Linear interpolation | UKF prediction |
| ID Switches | Baseline | Improved |
| Tracking Robustness | Good | Better |

---

## Liên hệ & Support

- **Issues**: Report tại GitHub repository
- **Documentation**: File này + code comments
- **Examples**: Xem `osnet_dcn_pipeline_tlukf_xysr.py`
