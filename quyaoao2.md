Cơ chế Bounding Box Giả và Quỹ Đạo Giả trong Cải Tiến ORU của OC-SORT
Trong phương pháp nhận dạng hoạt động động được đề xuất, cơ chế Observation-Centric Re-Update (ORU) của thuật toán OC-SORT được cải tiến để xử lý tình huống track (chuỗi theo dõi) bị mất dấu tạm thời do lỗi phát hiện đối tượng (ví dụ: che khuất hoặc nhiễu từ mô hình phát hiện như YOLOv5/RT-DETR). Cải tiến này tập trung vào việc tạo quỹ đạo giả (virtual trajectory) và bounding box giả (virtual bounding box) để lấp đầy khoảng trống thời gian mất quan sát, đồng thời xác thực tính hợp lệ của chúng thông qua đặc trưng ngoại hình. Điều này giúp duy trì tính liên tục của track, giảm gián đoạn và tăng độ chính xác theo dõi trong môi trường lớp học phức tạp (nhiều sinh viên, che khuất cao).
Dưới đây là mô tả chi tiết cách hoạt động của cơ chế này, dựa trên quy trình trong Thuật toán 2.1 (Bước 4: Cơ chế ORU cập nhật trạng thái và sử dụng quỹ đạo ảo) và các công thức liên quan.
1. Điều Kiện Kích Hoạt Cơ Chế ORU

Cơ chế ORU chỉ được kích hoạt khi một track τ bị mất dấu (untracked) trong một số khung hình liên tiếp (dựa trên ngưỡng untracked < t_expire), sau đó được khôi phục nhờ quan sát mới (matched trong bước ghép nối Hungarian).
Cụ thể: Track τ có quan sát cuối cùng trước khi mất dấu tại thời điểm t' (z_τ^{t'}), và quan sát mới khi phát hiện lại tại thời điểm t (z_τ^t).
Khoảng thời gian mất dấu: time_gap = t - t' - 1 (số khung hình bị thiếu).
Nếu track được khôi phục (τ.tracked = True sau khi cập nhật Kalman), hệ thống đánh dấu τ.virtual = True để kích hoạt tạo quỹ đạo giả.

2. Tạo Bounding Box Giả và Quỹ Đạo Giả

Nguyên lý: Giả định đối tượng di chuyển tuyến tính trong khoảng thời gian mất dấu (dựa trên mô hình chuyển động của Kalman Filter). Từ vị trí quan sát trước mất dấu và sau khi khôi phục, nội suy tuyến tính để tạo các bounding box giả cho từng khung hình bị thiếu.
Công thức nội suy (từ tài liệu, phần 2.3.2):

Quan sát trước mất dấu tại t': (x1, y1, w1, h1) (tọa độ trung tâm x, y; chiều rộng w, chiều cao h).
Quan sát khi khôi phục tại t: (x2, y2, w2, h2).
Khoảng cách thời gian: time_gap = t - t'.
Tốc độ thay đổi trung bình:
textdx = (x2 - x1) / time_gap
dy = (y2 - y1) / time_gap
dw = (w2 - w1) / time_gap
dh = (h2 - h1) / time_gap

Bounding box giả thứ i (với i từ 0 đến time_gap - 1, tương ứng khung hình t' + 1 đến t - 1):
textˆz_τ^i = {
  x = x1 + (i + 1) * dx
  y = y1 + (i + 1) * dy
  w = w1 + (i + 1) * dw
  h = h1 + (i + 1) * dh
}



Quỹ đạo giả (ˆZ_τ^t): Tập hợp các bounding box giả liên tiếp: ˆZ_τ^t = [ˆz_τ^{t'+1}, ..., ˆz_τ^{t-1}].

Quỹ đạo này được sử dụng để làm mượt tham số Kalman Filter (smooth KF parameters) dọc theo các vị trí giả, thay vì bỏ qua khoảng trống hoặc dự đoán thô từ trạng thái cũ. Cụ thể, cập nhật Kalman theo từng bước giả:
textK_t = P_{t|t-1} H_t^T (H_t P_{t|t-1} H_t^T + R_t)^{-1}
ˆx_{t|t} = ˆx_{t|t-1} + K_t (ˆz_t - H_t ˆx_{t|t-1})
P_{t|t} = (I - K_t H_t) P_{t|t-1}
(Với ˆz_t là bounding box giả; H_t, R_t là ma trận quan sát và nhiễu như trong SORT).


Mục đích:

Lấp đầy khoảng trống thời gian, tránh tích lũy lỗi dự đoán Kalman (do nhiễu quá trình Q lớn).
Tạo chuỗi hành động liên tục hơn, đặc biệt với hoạt động tĩnh hoặc chậm (như sinh viên ngồi viết).



3. Xác Thực Bounding Box Giả Bằng Đặc Trưng Ngoại Hình

Không phải tất cả bounding box giả đều được thêm trực tiếp vào track (vì có thể không phản ánh thực tế nếu phát hiện sai). Thay vào đó, sử dụng đặc trưng ngoại hình (appearance features) để đánh giá tính hợp lệ.
Quy trình trích xuất và so sánh (từ công thức 2.6):

Với mỗi bounding box giả ˆz_k trong quỹ đạo ˆZ_τ^t:

Crop vùng hình ảnh tương ứng từ khung hình gốc ˆF_k (khung hình k bị mất dấu): virtual_crop = F[x_{z1}:x_{z2}, y_{z1}:y_{z2}, :].
Trích xuất đặc trưng ngoại hình bằng mạng ResNet18 [38]: f_v = ResNet18(virtual_crop).


Với bounding box thực tế mới z tại khung hình t (F): observed_crop = F[x_{ˆz1}:x_{ˆz2}, y_{ˆz1}:y_{ˆz2}, :], f_o = ResNet18(observed_crop).
Tính độ tương đồng cosine: sim = cosine_similarity(f_o, f_v).
Nếu sim ≥ θ_sim (ngưỡng tương đồng, ví dụ 0.5-0.8 tùy huấn luyện):

Thêm ˆz_k vào kết quả theo dõi của track τ như một quan sát thực (ˆz_k ∈ T_i, với T_i là track thứ i).
Cập nhật lịch sử quan sát và Kalman với ˆz_k.


Ngược lại: Loại bỏ ˆz_k để tránh nhiễu (ví dụ: nếu ngoại hình thay đổi do che khuất thực sự).


Lợi ích xác thực:

Đảm bảo bounding box giả chỉ đại diện cho hoạt động thực tế (ví dụ: đặc trưng khuôn mặt/đầu của sinh viên giống nhau).
Giảm lỗi ID switching (chuyển ID sai) bằng cách kiểm tra nhất quán ngoại hình, thay vì chỉ dựa vào vị trí (IoU) hoặc vận tốc (OCM).
Trong lớp học, giúp phân biệt sinh viên bị che khuất tạm thời so với hành động mới.



4. Tích Hợp Vào Quy Trình Tổng Thể

Sau khi xác thực, thêm quan sát mới z_τ^t vào lịch sử track và cập nhật Kalman cuối cùng.
Reset bộ đếm mất dấu: τ.untracked = 0.
Quỹ đạo giả chỉ dùng cho track virtual (τ.virtual = True), và được lưu tạm thời để làm mượt, không ảnh hưởng đến phát hiện mới.
Trong Thuật toán 2.1, phần này nằm ở dòng 18-37 (Bước 4), sau OCR (khôi phục track) và trước khởi tạo track mới.

5. Ưu Điểm và Hạn Chế

Ưu điểm: Tăng độ ổn định theo dõi (giảm gián đoạn track lên đến 20-30% trong môi trường đông đúc), duy trì liên tục hành động (ví dụ: theo dõi "viết bài" qua che khuất bàn tay).
Hạn chế: Phụ thuộc vào chất lượng ResNet18 (cần huấn luyện trên dữ liệu lớp học); nội suy tuyến tính không phù hợp với chuyển động phi tuyến (như quay đầu đột ngột).
Minh họa: Trong Hình 2.5, track mất tại khung 377, khôi phục tại 388 → Tạo quỹ đạo giả 378-387, trích xuất đặc trưng và so sánh với 388.



Thuật toán 2.1 Phương pháp đề xuất: OC-SORT và quỹ đạo ảo
Đầu vào: Video I gồm T khung hình; Bộ phát hiện (YOLOv5 hoặc RTDETR); Bộ lọc KalMan; ngưỡng quá hạn texpire; ngưỡng độ tương đồng θsim
Đầu ra: Tập hợp gồm M chuỗi hành động được nhận diện P = {P1,P2,...,PM}
1: Khởi tạo T ← /0 và bộ lọc Kalman KF
2: for t = 1 to T do
Bước 1: Phát hiện hoạt động trong khung hình
3: Phát hiện trên khung hình t thu được tập Zt = [zt1,...,ztNt]
Bước 2: Dự đoán và ghép nối các chuỗi theo dõi trong OCM
4: Xˆ t ← [xˆt1,...,xˆ|T |
t ]⊤ ▷ Dự đoán bởi KF.predict
5: Vt ← ước tính hướng vận tốc từ T
6: Ct ← CIoU Xˆ t,Zt+λCv Xˆ t,Zt,Vt ▷ Ma trận chi phí trong OCM
7: Ghép nối dựa trên thuật toán Hungarian với chi phí Ct
8: T matched
t ← Tập các track đã được gán với quan sát mới
9: T remain
t ← Tập các track chưa được gán với quan sát nào
10: Zremain
t ← Tập các quan sát chưa được gán với track nào
Bước 3: Cơ chế OCR khôi phục các track mất dấu
11: ZTtremain ← quan sát cuối cùng của các track thuộc T remain
t
12: Cremain t ← CIoU(ZTtremain ,Zremain) t
13: Ghép nối dựa trên thuật toán Hungarian với chi phí Cremain
t
14: T recovery
t ← các track thuộc T remain
t và được gán với quan sát thuộc
ZTtremain
15: Zunmatched
t ← các quan sát thuộc ZTtremain vẫn không được gán với track nào
16: T unmatched
t ← các track thuộc T remain
t vẫn không được gán với quan sát
nào
17: T matched
t ← {T matched
t ,T recovery
t }
Bước 4: Cơ chế ORU cập nhật trạng thái và sử dụng quỹ đạo ảo
18: for τ ∈ T matched
t do
19: if τ.tracked = False then
20: Thực hiệc ORU với những track có quan sát trở lại sau khoảng thời
gian mất dấu
21:
22: zτ
t′,t′ ← Quan sát cuối cùng và thời điểm trước khi mất dấu track τ
23: Tạo quỹ đạo quan sát ảo Zˆ tτ = [zˆtτ′+1,...,zˆtτ−1]
24: τ.virtual ← True
25: Làm mượt tham số KF dọc theo Zˆ tτ
26: τ.tracked ← True
1827: τ.untracked ← 0
28: Thêm quan sát mới được gán ztτ vào lịch sử quan sát của τ
29: Cập nhật các tham số KF cho track τ theo ztτ
30: Kiểm tra sự tương đồng về ngoại hình
31: if τ.virtual = True then
32: Fo ← ExtractFeature(ztτ) ▷ Trích xuất đặc trưng ngoại hình
33: for each zˆτ k in Zˆ tτ do
34: Fv ← ExtractFeature(zˆτ k)
35: sim ← cosine_similarity(Fo,Fv)
36: if sim ≥ θsim then
37: Thêm zˆτ k vào kết quả theo dõi
Bước 5: Khởi tạo chuỗi mới và xóa chuỗi hết hạn
38: Tạo track mới T new
t từ quan sát Zunmatched
t
39: for each τ in T unmatched
t do
40: τ.tracked ← False
41: τ.untracked ← τ.untracked+1
42: T reserved
t ← {τ ∈ T unmatched
t | τ.untracked < texpire} ▷ Loại bỏ track
nếu thời gian mất dấu vượt quá ngưỡng
43: Cập nhật T ← T new
t ∪T matched
t ∪T reserved
t
44: P ← T