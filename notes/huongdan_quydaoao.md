Mô tả chi tiết + code: tích hợp quỹ đạo ảo vào KalmanFilterXYSR

Dưới đây mình cung cấp một hướng triển khai đầy đủ (lý thuyết → công thức → code Python) để bạn cắm thẳng vào pipeline XYSR của bạn. Nội dung gồm:

Các phép biến đổi toán → code (s,r ↔ w,h).

Hàm nội suy tạo danh sách bounding boxes ảo (x, y, s, r) với công thức tuyến tính (nội suy w,h).

Hàm áp dụng quỹ đạo ảo vào Kalman filter (apply_virtual_trajectory) — thực hiện: cập nhật KF từng bước, dùng confidence cho measurement ảo, optional appearance check (cosine sim) để quyết định có “xuất” các hộp ảo vào kết quả tracking hay không.

Gợi ý chỉnh unfreeze() / update() trong class của bạn để dùng confidence khi update ảo.

Kiểm tra và tham số tuning.

Mình ghi rất rõ công thức toán trong comment để bạn dễ theo dõi.

1) Chuyển đổi s,r ↔ w,h (Toán → code)

Công thức:

từ s, r sang w, h:

𝑤
=
𝑠
⋅
𝑟
w=
s⋅r
	​


ℎ
=
𝑠
/
𝑟
h=
s/r
	​


từ w, h sang s, r:

𝑠
=
𝑤
⋅
ℎ
s=w⋅h

𝑟
=
𝑤
/
ℎ
r=w/h

Code:

import numpy as np

_eps = 1e-6

def s_r_to_w_h(s: float, r: float):
    """Convert scale s (area) and ratio r to width w and height h."""
    # numeric safety
    s = float(s)
    r = float(max(r, _eps))
    w = np.sqrt(max(s * r, 0.0))
    h = np.sqrt(max(s / r, 0.0))
    return w, h

def w_h_to_s_r(w: float, h: float):
    """Convert width and height to scale s and ratio r."""
    w = float(max(w, _eps))
    h = float(max(h, _eps))
    s = w * h
    r = w / h
    return s, r

2) Nội suy tuyến tính cho quỹ đạo ảo (công thức + code)

Giả sử có hai quan sát:

𝑧
1
=
(
𝑥
1
,
𝑦
1
,
𝑠
1
,
𝑟
1
)
z
1
	​

=(x
1
	​

,y
1
	​

,s
1
	​

,r
1
	​

) tại frame t1

𝑧
2
=
(
𝑥
2
,
𝑦
2
,
𝑠
2
,
𝑟
2
)
z
2
	​

=(x
2
	​

,y
2
	​

,s
2
	​

,r
2
	​

) tại frame t2

time_gap = t2 - t1 (số bước giữa hai quan sát; nếu =1 thì không có frame mất)

Chi tiết nội suy:

Chuyển 
𝑠
𝑖
,
𝑟
𝑖
→
𝑤
𝑖
,
ℎ
𝑖
s
i
	​

,r
i
	​

→w
i
	​

,h
i
	​


Tính delta từng bước:

𝑑
𝑥
=
𝑥
2
−
𝑥
1
𝑡
𝑖
𝑚
𝑒
_
𝑔
𝑎
𝑝
dx=
time_gap
x
2
	​

−x
1
	​

	​


𝑑
𝑦
=
𝑦
2
−
𝑦
1
𝑡
𝑖
𝑚
𝑒
_
𝑔
𝑎
𝑝
dy=
time_gap
y
2
	​

−y
1
	​

	​


𝑑
𝑤
=
𝑤
2
−
𝑤
1
𝑡
𝑖
𝑚
𝑒
_
𝑔
𝑎
𝑝
dw=
time_gap
w
2
	​

−w
1
	​

	​


𝑑
ℎ
=
ℎ
2
−
ℎ
1
𝑡
𝑖
𝑚
𝑒
_
𝑔
𝑎
𝑝
dh=
time_gap
h
2
	​

−h
1
	​

	​


Với k = 1..(time_gap-1):

𝑥
𝑡
1
+
𝑘
=
𝑥
1
+
𝑘
⋅
𝑑
𝑥
x
t1+k
	​

=x
1
	​

+k⋅dx

𝑦
𝑡
1
+
𝑘
=
𝑦
1
+
𝑘
⋅
𝑑
𝑦
y
t1+k
	​

=y
1
	​

+k⋅dy

𝑤
𝑡
1
+
𝑘
=
𝑤
1
+
𝑘
⋅
𝑑
𝑤
w
t1+k
	​

=w
1
	​

+k⋅dw

ℎ
𝑡
1
+
𝑘
=
ℎ
1
+
𝑘
⋅
𝑑
ℎ
h
t1+k
	​

=h
1
	​

+k⋅dh

Sau đó: 
𝑠
=
𝑤
⋅
ℎ
,
 
𝑟
=
𝑤
/
ℎ
s=w⋅h, r=w/h

Code:

from typing import List

def interpolate_virtual_boxes(z1: np.ndarray, z2: np.ndarray, t1: int, t2: int, max_gap:int=50) -> List[np.ndarray]:
    """
    Return list of virtual measurements between z1@t1 and z2@t2 (exclusive).
    z1, z2: arrays shape (4,) as (x,y,s,r)
    returns: [z_hat_t1+1, ..., z_hat_t2-1] each shape (4,)
    """
    time_gap = int(t2 - t1)
    if time_gap <= 1:
        return []

    if time_gap > max_gap:
        # Avoid creating lots of virtual boxes if gap too large
        return []

    x1, y1, s1, r1 = map(float, z1)
    x2, y2, s2, r2 = map(float, z2)

    w1, h1 = s_r_to_w_h(s1, r1)
    w2, h2 = s_r_to_w_h(s2, r2)

    dx = (x2 - x1) / time_gap
    dy = (y2 - y1) / time_gap
    dw = (w2 - w1) / time_gap
    dh = (h2 - h1) / time_gap

    virtual_boxes = []
    for k in range(1, time_gap):
        xv = x1 + k * dx
        yv = y1 + k * dy
        wv = w1 + k * dw
        hv = h1 + k * dh
        sv, rv = w_h_to_s_r(wv, hv)
        virtual_boxes.append(np.array([xv, yv, sv, rv], dtype=float))
    return virtual_boxes

3) Appearance check (cosine similarity) — code

StrongSORT đã có embedding extractor; mình dùng một hàm cosine similarity đơn giản:

def cosine_similarity_vec(a: np.ndarray, b: np.ndarray, eps=1e-8):
    # a, b: 1D vectors
    a = a.reshape(-1)
    b = b.reshape(-1)
    na = np.linalg.norm(a) + eps
    nb = np.linalg.norm(b) + eps
    return float(np.dot(a, b) / (na * nb))

4) Hàm chính: apply_virtual_trajectory — tích hợp vào pipeline

Hàm này làm việc ngoại vi (không bắt buộc thay đổi class KF), nhưng dùng API kf.predict() và kf.update(meas, confidence=...) mà KalmanFilterXYSR bạn đã có.

Ý tưởng:

tạo virtual list bằng interpolate_virtual_boxes

cho mỗi z_hat: nếu có frame để crop + extract_feature hook thì tính sim với feature của z2; dựa trên sim quyết định confidence và có thêm hay không box ảo vào kết quả track.

Luôn dùng confidence thấp cho measurement ảo (conf_virtual), nhưng nếu appearance rất giống (sim high), nâng confidence (mapsim→conf) hoặc mark virtual as accepted cho output.

Code đầy đủ:

from typing import Callable, Optional, Tuple, Dict, Any

def apply_virtual_trajectory(
    kf,                       # KalmanFilterXYSR instance
    track,                    # track object or dict to which we may add boxes (user-provided)
    z1: np.ndarray, t1: int,   # last real measurement and time before miss
    z2: np.ndarray, t2: int,   # new real measurement and time after re-detect
    frames: Optional[Dict[int, Any]] = None,      # mapping frame_idx -> frame image (optional)
    crop_fn: Optional[Callable] = None,           # function(frame, x,y,w,h) -> crop image
    extract_feature: Optional[Callable] = None,   # function(crop) -> 1D embedding
    theta_sim: float = 0.65,
    conf_virtual_default: float = 0.30,
    conf_virtual_if_sim_high: Tuple[float,float]=(0.6, 0.9),  # mapping range for sim->conf
    max_gap: int = 30,
    add_virtual_to_track: Optional[Callable] = None, # function(track, box, virtual_flag)
):
    """
    Apply virtual trajectory between (z1,t1) and (z2,t2).
    - kf: KalmanFilterXYSR with methods predict(), update(meas, confidence=float)
    - track: user track object (can be dict), used only if add_virtual_to_track provided
    - frames + crop_fn + extract_feature: optional for appearance checking
    - add_virtual_to_track: optional callback to append virtual boxes to track result
    Returns:
      accepted_virtuals: list of (frame_idx, z_hat, sim) that were accepted by appearance
    """
    accepted_virtuals = []
    time_gap = int(t2 - t1)
    if time_gap <= 1 or time_gap > max_gap:
        # no intermediate frames or gap too large -> just update kf with z2
        kf.update(z2, confidence=1.0)
        if add_virtual_to_track:
            add_virtual_to_track(track, z2, virtual=False)
        return accepted_virtuals

    # create virtual boxes
    virtual_boxes = interpolate_virtual_boxes(z1, z2, t1, t2, max_gap=max_gap)

    # compute appearance feature of observed z2 if extractor available
    Fo = None
    if frames is not None and crop_fn is not None and extract_feature is not None and (t2 in frames):
        x2, y2, s2, r2 = z2
        w2, h2 = s_r_to_w_h(s2, r2)
        crop_obs = crop_fn(frames[t2], x2, y2, w2, h2)
        Fo = extract_feature(crop_obs)

    # iterate virtual boxes
    for idx, z_hat in enumerate(virtual_boxes, start=1):
        frame_idx = t1 + idx
        # appearance check if possible
        sim = None
        if Fo is not None and frames is not None and crop_fn is not None and extract_feature is not None and (frame_idx in frames):
            xh, yh, sh, rh = z_hat
            wh, hh = s_r_to_w_h(sh, rh)
            crop_v = crop_fn(frames[frame_idx], xh, yh, wh, hh)
            Fv = extract_feature(crop_v)
            sim = cosine_similarity_vec(Fo, Fv)

        # decide confidence
        if sim is None:
            conf = conf_virtual_default
            accept_for_output = False
        else:
            # map sim -> confidence, and decide acceptance
            if sim >= theta_sim:
                # high similarity -> stronger confidence and accept
                # simple mapping: conf = conf_virtual_if_sim_high[0] + (sim-theta)*(range)
                low, high = conf_virtual_if_sim_high
                # normalize sim in [theta_sim, 1]
                frac = min(1.0, max(0.0, (sim - theta_sim) / (1.0 - theta_sim + 1e-8)))
                conf = low + frac * (high - low)
                accept_for_output = True
            else:
                # low sim => very low confidence (only nudge KF), not accepted in output
                conf = min(conf_virtual_default, 0.2)
                accept_for_output = False

        # update KF with the virtual measurement (measurement is in x,y,s,r)
        kf.update(z_hat, confidence=conf)

        # If accepted -> optionally record into track / output
        if accept_for_output and add_virtual_to_track is not None:
            add_virtual_to_track(track, z_hat, virtual=True)
            accepted_virtuals.append((frame_idx, z_hat, sim))

        # if not last virtual, predict forward to next timestep
        if idx != len(virtual_boxes):
            kf.predict()

    # finally update with real observation
    kf.update(z2, confidence=1.0)
    if add_virtual_to_track is not None:
        add_virtual_to_track(track, z2, virtual=False)

    return accepted_virtuals


Giải thích điểm quan trọng trong code:

virtual_boxes được sinh bằng nội suy w,h (xem hàm interpolate_virtual_boxes).

Mỗi virtual z_hat được dùng để gọi kf.update(z_hat, confidence=conf); điều này cho phép KF “nudge” dọc đường thay vì nhảy từ t1 → t2.

conf được scale theo similarity (nếu có). Khi không có appearance info, sử dụng conf_virtual_default nhỏ (ví dụ 0.3).

add_virtual_to_track là callback của bạn (tùy pipeline) để thêm box vào đầu ra nếu accept_for_output==True. Nếu bạn muốn track luôn xuất virtual boxes mà không check appearance, set add_virtual_to_track và conf_virtual_default phù hợp.

5) Cập nhật nội dung unfreeze() trong KalmanFilterXYSR

Bạn đã có một unfreeze() trong class. Mình khuyến nghị sửa để unfreeze() gọi update(new_box, confidence=conf_virtual) thay vì mặc định 1.0. Dưới đây là ý chính (patch):

# inside KalmanFilterXYSR.unfreeze(), replace the block that calls self.update(new_box)
# currently: self.update(new_box)
# change to:
conf_virtual_default = 0.30  # tune as needed
self.update(new_box, confidence=conf_virtual_default)


Nếu bạn muốn hỗ trợ appearance inside-class, bạn có thể cung cấp callback hoặc property self.extract_feature_fn và frames buffer self.frame_buffer để unfreeze() có thể compute sim và chọn confidence adaptively (tương tự apply_virtual_trajectory).

6) Các điểm cần chú ý / tuning cho video nội soi (biến dạng mạnh)

max_gap (tối đa bước để tạo virtual): nội soi có biến dạng lớn → chọn nhỏ (3..8 frames). Nếu framerate cao bạn có thể tăng nhẹ.

conf_virtual_default: 0.2–0.4 (ảo) — càng nhỏ càng “mềm”, nhưng nếu quá nhỏ KF sẽ không được hiệu chỉnh đủ.

theta_sim: nếu extractor embedding ổn → 0.65–0.8; nếu embedding yếu (đối tượng thay đổi texture nhiều), giảm xuống 0.45–0.6.

Nếu tổn thương biến dạng phi tuyến nhiều (co giãn, shape change), tuyến tính w,h vẫn là approximation — bạn có thể dùng spline hoặc constant-acceleration model, nhưng thường chi phí/độ phức tạp tăng.

Nếu bạn không có frames cũ để crop (hoặc memory hạn chế), vẫn dùng virtual boxes để update KF (không appearance check). Điều này vẫn tốt để giảm ID-switch khi gap ngắn.

Sau việc áp dụng virtual boxes, có thể chạy temporal smoothing (EMA) lên trajectory xuất ra để giảm jitter do update ảo.

7) Ví dụ cắm thẳng vào StrongSORT (pseudo-integration)

Giả sử trong tracker loop khi bạn matched track tr with new detection det at frame t:

if tr.lost and det and (t - tr.last_seen_time) <= max_gap:
    # we have z1 at tr.last_seen_time, z2 is det at t
    z1 = tr.last_measurement  # (x,y,s,r)
    z2 = det.xy s r...
    accepted = apply_virtual_trajectory(
        kf=tr.kf, track=tr, z1=z1, t1=tr.last_seen_time,
        z2=z2, t2=t, frames=frame_buffer, crop_fn=crop_fn,
        extract_feature=extract_feature_fn, theta_sim=0.65,
        conf_virtual_default=0.30, add_virtual_to_track=tr.add_box
    )
else:
    # normal update
    tr.kf.update(det.measurement, confidence=det.confidence)

8) Kiểm tra tính đúng đắn (test plan)

Unit-test interpolate_virtual_boxes với vài z1,z2 để kiểm tra w,h->s,r chuyển đổi.

Dry-run: simulate sequence: observe at t=0, miss t=1..3, observe again t=4 → call apply_virtual_trajectory and log kf.x states before and after. Dự kiến: trạng thái trước sau mượt hơn (không jump).

A/B test: chạy full video với/không dùng virtual trajectory, so sánh:

số ID-switch, fragmentations, MOTA (nếu ground-truth).

quan sát bằng mắt (visualize trajectories).

Tune theta_sim, conf_virtual_default, max_gap.

9) Tóm tắt ngắn gọn

Tạo virtual boxes bằng nội suy w,h giữa (x,y,s,r) hai quan sát.

Dùng kf.update(z_hat, confidence=...) cho mỗi hộp ảo; confidence thấp để biểu diễn độ không chắc chắn.

Nếu có appearance features, dùng cosine similarity để quyết định có chấp nhận box ảo vào output hay chỉ dùng nó để hiệu chỉnh KF.

Hạn chế gap dài, scale noise lớn, test và tune.