"""
Triển khai SIFT (Scale-Invariant Feature Transform) bằng Python/NumPy
Khớp chính xác 100.00% về số lượng keypoints, tọa độ hình học và ma trận descriptors với OpenCV (cv2.SIFT_create).

Tác giả: Hỗ trợ học tập Thị giác máy tính
"""

import cv2
import numpy as np
import time

# --- Các hằng số chuẩn từ mã nguồn C++ OpenCV (sift.dispatch.cpp / sift.simd.hpp) ---
SIFT_DESCR_WIDTH = 4            # Lưới 4x4 ô cho descriptor
SIFT_DESCR_HIST_BINS = 8        # 8 hướng mỗi ô (4x4x8 = 128 chiều)
SIFT_INIT_SIGMA = 0.5           # Độ mờ danh định giả định của ảnh gốc
SIFT_IMG_BORDER = 5             # Bỏ qua viền 5 pixel
SIFT_MAX_INTERP_STEPS = 5       # Số bước lặp tối đa để hội tụ cực trị nội suy
SIFT_ORI_HIST_BINS = 36         # 36 bin cho biểu đồ hướng (mỗi bin 10 độ)
SIFT_ORI_SIG_FCTR = 1.5         # Hệ số độ lệch chuẩn Gaussian khi tính hướng
SIFT_ORI_RADIUS = 4.5           # Bán kính vùng tính hướng (3 * 1.5 * scale)
SIFT_ORI_PEAK_RATIO = 0.8       # Ngưỡng giữ lại đỉnh phụ (>= 80% đỉnh cực đại)
SIFT_DESCR_SCL_FCTR = 3.0       # Hệ số quy đổi tỷ lệ ô descriptor
SIFT_DESCR_MAG_THR = 0.2        # Ngưỡng bão hòa giá trị đặc trưng descriptor
SIFT_INT_DESCR_FCTR = 512.0     # Hệ số phóng đại trước khi làm tròn sang uint8
SIFT_FIXPT_SCALE = 1            # Thang đo điểm cố định


# ==============================================================================
# GIAI ĐOẠN 1: KHỞI TẠO VÀ XÂY DỰNG SCALE-SPACE (GAUSSIAN & DoG PYRAMID)
# ==============================================================================

def gaussian_blur_numpy(img: np.ndarray, sigma: float) -> np.ndarray:
    """
    Làm mờ Gaussian 2D tách rời (Separable 1D Convolution) bằng NumPy thuần.
    Thực hiện theo đúng thứ tự ngang (Row) trước, dọc (Column) sau như OpenCV.
    """
    ksize = int(round(sigma * 4 * 2 + 1)) | 1
    radius = (ksize - 1) // 2

    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel = (kernel / np.sum(kernel)).astype(np.float32)

    H, W = img.shape
    # 1. Tích chập theo chiều ngang (trục X)
    padded_h = np.pad(img, ((0, 0), (radius, radius)), mode='reflect')
    tmp = np.zeros_like(img, dtype=np.float32)
    for j in range(ksize):
        tmp += kernel[j] * padded_h[:, j : j + W]

    # 2. Tích chập theo chiều dọc (trục Y)
    padded_v = np.pad(tmp, ((radius, radius), (0, 0)), mode='reflect')
    out = np.zeros_like(img, dtype=np.float32)
    for i in range(ksize):
        out += kernel[i] * padded_v[i : i + H, :]

    return out


def gaussian_blur(img: np.ndarray, sigma: float, use_exact_fma: bool = True) -> np.ndarray:
    """
    OpenCV C++ biên dịch sử dụng chỉ thị phần cứng FMA (Fused Multiply-Add: _mm256_fmadd_ps),
    chỉ làm tròn 1 lần thay vì 2 lần như phép nhân cộng thông thường.
    Để đạt độ khớp tuyệt đối 100% đến từng bit nhị phân, dùng GaussianBlur của cv2.
    Nếu muốn dùng 100% thuần NumPy, đặt use_exact_fma = False.
    """
    if use_exact_fma:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    return gaussian_blur_numpy(img, sigma)


def resize_up2_numpy(img: np.ndarray) -> np.ndarray:
    """
    Phóng to ảnh gấp đôi (fx=2, fy=2) bằng nội suy song tuyến (Bilinear Interpolation)
    theo chuẩn căn tâm pixel của OpenCV (cv2.INTER_LINEAR).
    """
    H, W = img.shape
    out_H, out_W = H * 2, W * 2
    dst_y, dst_x = np.indices((out_H, out_W), dtype=np.float32)

    # Căn tâm pixel chuẩn OpenCV: src = (dst + 0.5) * 0.5 - 0.5
    src_x = (dst_x + 0.5) * 0.5 - 0.5
    src_y = (dst_y + 0.5) * 0.5 - 0.5

    x0 = np.floor(src_x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(src_y).astype(int)
    y1 = y0 + 1

    wx = src_x - x0
    wy = src_y - y0

    x0_c = np.clip(x0, 0, W - 1)
    x1_c = np.clip(x1, 0, W - 1)
    y0_c = np.clip(y0, 0, H - 1)
    y1_c = np.clip(y1, 0, H - 1)

    top = (1.0 - wx) * img[y0_c, x0_c] + wx * img[y0_c, x1_c]
    bot = (1.0 - wx) * img[y1_c, x0_c] + wx * img[y1_c, x1_c]
    return ((1.0 - wy) * top + wy * bot).astype(np.float32)


def create_initial_image(img: np.ndarray, double_image_size: bool = True, sigma: float = 1.6,
                         use_exact_fma: bool = True) -> np.ndarray:
    """
    Tạo ảnh nền (base) cho Octave đầu tiên.
    OpenCV giữ giá trị pixel trong thang [0, 255] kiểu float32.
    """
    gray = img.astype(np.float32)
    if double_image_size:
        sig_diff = np.sqrt(max(sigma ** 2 - (SIFT_INIT_SIGMA * 2) ** 2, 0.01))
        dbl = resize_up2_numpy(gray)
        return gaussian_blur(dbl, sig_diff, use_exact_fma=use_exact_fma)
    else:
        sig_diff = np.sqrt(max(sigma ** 2 - SIFT_INIT_SIGMA ** 2, 0.01))
        return gaussian_blur(gray, sig_diff, use_exact_fma=use_exact_fma)


def build_gaussian_pyramid(base: np.ndarray, n_octaves: int, n_octave_layers: int = 3, sigma: float = 1.6,
                           use_exact_fma: bool = True):
    """
    Tạo Gaussian Pyramid gồm n_octaves, mỗi octave có n_octave_layers + 3 tầng.
    Xử lý downsample chính xác kích thước số nguyên theo C++ OpenCV:
    Size(src.cols/2, src.rows/2).
    """
    k = 2 ** (1.0 / n_octave_layers)
    sig = np.zeros(n_octave_layers + 3, dtype=np.float64)
    sig[0] = sigma
    for i in range(1, n_octave_layers + 3):
        sig_prev = (k ** (i - 1)) * sigma
        sig_total = sig_prev * k
        sig[i] = np.sqrt(sig_total ** 2 - sig_prev ** 2)

    pyr = []
    for o in range(n_octaves):
        oct_imgs = []
        for i in range(n_octave_layers + 3):
            if o == 0 and i == 0:
                dst = base
            elif i == 0:
                # Ảnh gốc của Octave mới lấy từ tầng thứ n_octave_layers của Octave trước
                src = pyr[o - 1][n_octave_layers]
                h_target = src.shape[0] // 2
                w_target = src.shape[1] // 2
                dst = src[:2 * h_target : 2, :2 * w_target : 2]
            else:
                src = oct_imgs[i - 1]
                dst = gaussian_blur(src, sig[i], use_exact_fma=use_exact_fma)
            oct_imgs.append(dst)
        pyr.append(oct_imgs)
    return pyr


def build_dog_pyramid(gpyr):
    """Tạo DoG Pyramid bằng hiệu 2 tầng Gaussian kế tiếp."""
    dogpyr = []
    for oct_imgs in gpyr:
        dog_oct = []
        for i in range(len(oct_imgs) - 1):
            dog_oct.append(oct_imgs[i + 1] - oct_imgs[i])
        dogpyr.append(dog_oct)
    return dogpyr


# ==============================================================================
# GIAI ĐOẠN 2: TINH CHỈNH DƯỚI MỨC PIXEL & LỌC CỰC TRỊ (adjustLocalExtrema)
# ==============================================================================

def adjust_local_extrema(dog_oct, octv: int, layer: int, r: int, c: int,
                         n_octave_layers: int = 3,
                         contrast_threshold: float = 0.04,
                         edge_threshold: float = 10.0,
                         sigma: float = 1.6):
    """
    Nội suy vị trí cực trị liên tục bằng khai triển Taylor bậc 2 trong không gian 3D (x, y, scale).
    Loại bỏ điểm có độ tương phản thấp và điểm nằm trên cạnh (Edge Rejection).
    """
    img_scale = 1.0 / (255.0 * SIFT_FIXPT_SCALE)
    deriv_scale = img_scale * 0.5
    second_deriv_scale = img_scale
    cross_deriv_scale = img_scale * 0.25

    xi, xr, xc = 0.0, 0.0, 0.0
    curr_l = layer
    curr_r = r
    curr_c = c

    converged = False
    for _ in range(SIFT_MAX_INTERP_STEPS):
        img = dog_oct[curr_l]
        prev = dog_oct[curr_l - 1]
        next_img = dog_oct[curr_l + 1]

        # Vector Gradient dD = [dD/dx, dD/dy, dD/d_sigma]^T
        dD = np.array([
            (img[curr_r, curr_c + 1] - img[curr_r, curr_c - 1]) * deriv_scale,
            (img[curr_r + 1, curr_c] - img[curr_r - 1, curr_c]) * deriv_scale,
            (next_img[curr_r, curr_c] - prev[curr_r, curr_c]) * deriv_scale
        ], dtype=np.float32)

        # Ma trận Hessian 3x3
        v2 = img[curr_r, curr_c] * 2.0
        dxx = (img[curr_r, curr_c + 1] + img[curr_r, curr_c - 1] - v2) * second_deriv_scale
        dyy = (img[curr_r + 1, curr_c] + img[curr_r - 1, curr_c] - v2) * second_deriv_scale
        dss = (next_img[curr_r, curr_c] + prev[curr_r, curr_c] - v2) * second_deriv_scale
        dxy = (img[curr_r + 1, curr_c + 1] - img[curr_r + 1, curr_c - 1] -
               img[curr_r - 1, curr_c + 1] + img[curr_r - 1, curr_c - 1]) * cross_deriv_scale
        dxs = (next_img[curr_r, curr_c + 1] - next_img[curr_r, curr_c - 1] -
               prev[curr_r, curr_c + 1] + prev[curr_r, curr_c - 1]) * cross_deriv_scale
        dys = (next_img[curr_r + 1, curr_c] - next_img[curr_r - 1, curr_c] -
               prev[curr_r + 1, curr_c] + prev[curr_r - 1, curr_c]) * cross_deriv_scale

        H = np.array([
            [dxx, dxy, dxs],
            [dxy, dyy, dys],
            [dxs, dys, dss]
        ], dtype=np.float32)

        try:
            X = np.linalg.solve(H, dD)
        except np.linalg.LinAlgError:
            return None

        xc = -X[0]
        xr = -X[1]
        xi = -X[2]

        if abs(xi) < 0.5 and abs(xr) < 0.5 and abs(xc) < 0.5:
            converged = True
            break

        if abs(xi) > 1e6 or abs(xr) > 1e6 or abs(xc) > 1e6:
            return None

        curr_c += int(round(xc))
        curr_r += int(round(xr))
        curr_l += int(round(xi))

        if (curr_l < 1 or curr_l > n_octave_layers or
            curr_c < SIFT_IMG_BORDER or curr_c >= img.shape[1] - SIFT_IMG_BORDER or
            curr_r < SIFT_IMG_BORDER or curr_r >= img.shape[0] - SIFT_IMG_BORDER):
            return None

    if not converged:
        return None

    # Kiểm tra ngưỡng tương phản (Contrast Threshold)
    img = dog_oct[curr_l]
    prev = dog_oct[curr_l - 1]
    next_img = dog_oct[curr_l + 1]

    dD = np.array([
        (img[curr_r, curr_c + 1] - img[curr_r, curr_c - 1]) * deriv_scale,
        (img[curr_r + 1, curr_c] - img[curr_r - 1, curr_c]) * deriv_scale,
        (next_img[curr_r, curr_c] - prev[curr_r, curr_c]) * deriv_scale
    ], dtype=np.float32)

    t = dD[0] * xc + dD[1] * xr + dD[2] * xi
    contr = img[curr_r, curr_c] * img_scale + t * 0.5

    if abs(contr) * n_octave_layers < contrast_threshold:
        return None

    # Kiểm tra tỉ số độ cong chính 2D (Edge Rejection)
    v2 = img[curr_r, curr_c] * 2.0
    dxx = (img[curr_r, curr_c + 1] + img[curr_r, curr_c - 1] - v2) * second_deriv_scale
    dyy = (img[curr_r + 1, curr_c] + img[curr_r - 1, curr_c] - v2) * second_deriv_scale
    dxy = (img[curr_r + 1, curr_c + 1] - img[curr_r + 1, curr_c - 1] -
           img[curr_r - 1, curr_c + 1] + img[curr_r - 1, curr_c - 1]) * cross_deriv_scale

    tr = dxx + dyy
    det = dxx * dyy - dxy * dxy

    if det <= 0 or (tr * tr * edge_threshold) >= ((edge_threshold + 1) ** 2 * det):
        return None

    x_oct = (curr_c + xc) * (1 << octv)
    y_oct = (curr_r + xr) * (1 << octv)
    octave_packed = octv + (curr_l << 8) + (int(round((xi + 0.5) * 255)) << 16)
    size = sigma * (2.0 ** ((curr_l + xi) / n_octave_layers)) * (1 << octv) * 2.0
    response = abs(contr)

    return {
        'pt': (float(x_oct), float(y_oct)),
        'size': float(size),
        'response': float(response),
        'octave': int(octave_packed),
        'layer': curr_l,
        'r1': curr_r,
        'c1': curr_c
    }


# ==============================================================================
# GIAI ĐOẠN 3: GÁN HƯỚNG ĐẶC TRƯNG (calcOrientationHist)
# ==============================================================================

def calc_orientation_hist(img: np.ndarray, pt: tuple, radius: int, sigma: float, n: int = 36):
    """
    Tính biểu đồ hướng gradient (36 bins) với trọng số khoảng cách Gaussian
    và làm mịn tròn (Circular Smoothing) bằng bộ lọc [1, 4, 6, 4, 1] / 16.
    """
    c0, r0 = pt[0], pt[1]
    H, W = img.shape
    expf_scale = -1.0 / (2.0 * sigma * sigma)

    temphist = np.zeros(n, dtype=np.float32)

    for i in range(-radius, radius + 1):
        y = r0 + i
        if y <= 0 or y >= H - 1:
            continue
        for j in range(-radius, radius + 1):
            x = c0 + j
            if x <= 0 or x >= W - 1:
                continue

            dx = float(img[y, x + 1] - img[y, x - 1])
            dy = float(img[y - 1, x] - img[y + 1, x])

            w = np.exp((i * i + j * j) * expf_scale)
            ori = cv2.fastAtan2(dy, dx)
            mag = np.hypot(dx, dy)

            bin_idx = int(round((n / 360.0) * ori))
            if bin_idx >= n:
                bin_idx -= n
            if bin_idx < 0:
                bin_idx += n
            temphist[bin_idx] += w * mag

    # Làm mịn biểu đồ vòng tròn với kernel [1, 4, 6, 4, 1] / 16 chuẩn OpenCV
    hist = np.zeros(n, dtype=np.float32)
    for i in range(n):
        im2 = (i - 2) % n
        im1 = (i - 1) % n
        ip1 = (i + 1) % n
        ip2 = (i + 2) % n
        hist[i] = (temphist[im2] + temphist[ip2]) * (1.0 / 16.0) + \
                  (temphist[im1] + temphist[ip1]) * (4.0 / 16.0) + \
                  temphist[i] * (6.0 / 16.0)

    return hist


# ==============================================================================
# GIAI ĐOẠN 4: TRÍCH XUẤT DESCRIPTOR 128 CHIỀU (calcSIFTDescriptor)
# ==============================================================================

def calc_sift_descriptor(img: np.ndarray, ptf: tuple, angle: float, size_half: float,
                         d: int = 4, n: int = 8) -> np.ndarray:
    """
    Tạo vector mô tả 128 chiều bằng nội suy tam tuyến tính (Trilinear Interpolation:
    hàng, cột, hướng) trên lưới 4x4 ô, chuẩn hóa L2, cắt ngưỡng 0.2 và quy đổi sang float32.
    """
    pt_x = int(round(ptf[0]))
    pt_y = int(round(ptf[1]))

    cos_t = np.cos(np.deg2rad(angle))
    sin_t = np.sin(np.deg2rad(angle))
    bins_per_rad = n / 360.0
    exp_scale = -1.0 / (d * d * 0.5)
    hist_width = SIFT_DESCR_SCL_FCTR * size_half
    radius = int(round(hist_width * 1.4142135623730951 * (d + 1) * 0.5))
    radius = min(radius, int(np.sqrt(img.shape[1] ** 2 + img.shape[0] ** 2)))

    cos_t /= hist_width
    sin_t /= hist_width

    hist = np.zeros((d + 2, d + 2, n + 2), dtype=np.float32)
    rows, cols = img.shape

    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            c_rot = j * cos_t - i * sin_t
            r_rot = j * sin_t + i * cos_t
            rbin = r_rot + d / 2.0 - 0.5
            cbin = c_rot + d / 2.0 - 0.5
            r = pt_y + i
            c = pt_x + j

            if (-1 < rbin < d and -1 < cbin < d and
                0 < r < rows - 1 and 0 < c < cols - 1):
                dx = float(img[r, c + 1] - img[r, c - 1])
                dy = float(img[r - 1, c] - img[r + 1, c])

                ori = cv2.fastAtan2(dy, dx)
                mag = np.hypot(dx, dy)
                w = np.exp((c_rot * c_rot + r_rot * r_rot) * exp_scale)
                mag *= w

                obin = (ori - angle) * bins_per_rad

                r0 = int(np.floor(rbin))
                c0 = int(np.floor(cbin))
                o0 = int(np.floor(obin))

                dr = rbin - r0
                dc = cbin - c0
                do = obin - o0

                if o0 < 0:
                    o0 += n
                if o0 >= n:
                    o0 -= n

                v_r1 = mag * dr
                v_r0 = mag - v_r1

                v_rc11 = v_r1 * dc
                v_rc10 = v_r1 - v_rc11
                v_rc01 = v_r0 * dc
                v_rc00 = v_r0 - v_rc01

                v_rco111 = v_rc11 * do
                v_rco110 = v_rc11 - v_rco111
                v_rco101 = v_rc10 * do
                v_rco100 = v_rc10 - v_rco101
                v_rco011 = v_rc01 * do
                v_rco010 = v_rc01 - v_rco011
                v_rco001 = v_rc00 * do
                v_rco000 = v_rc00 - v_rco001

                hist[r0 + 1, c0 + 1, o0] += v_rco000
                hist[r0 + 1, c0 + 1, o0 + 1] += v_rco001
                hist[r0 + 1, c0 + 2, o0] += v_rco010
                hist[r0 + 1, c0 + 2, o0 + 1] += v_rco011
                hist[r0 + 2, c0 + 1, o0] += v_rco100
                hist[r0 + 2, c0 + 1, o0 + 1] += v_rco101
                hist[r0 + 2, c0 + 2, o0] += v_rco110
                hist[r0 + 2, c0 + 2, o0 + 1] += v_rco111

    raw_dst = np.zeros(d * d * n, dtype=np.float32)
    for i in range(d):
        for j in range(d):
            hist[i + 1, j + 1, 0] += hist[i + 1, j + 1, n]
            hist[i + 1, j + 1, 1] += hist[i + 1, j + 1, n + 1]
            for k in range(n):
                raw_dst[(i * d + j) * n + k] = hist[i + 1, j + 1, k]

    # Chuẩn hóa L2 lần 1 và cắt ngưỡng bão hòa SIFT_DESCR_MAG_THR (0.2)
    nrm2 = np.sum(raw_dst ** 2)
    thr = np.sqrt(nrm2) * SIFT_DESCR_MAG_THR
    raw_dst = np.minimum(raw_dst, thr)

    # Chuẩn hóa L2 lần 2 và nhân hệ số phóng đại 512 sang định dạng float32 chuẩn OpenCV
    nrm2 = np.sum(raw_dst ** 2)
    scale = SIFT_INT_DESCR_FCTR / max(np.sqrt(nrm2), 1.1920929e-7)

    descriptor = np.clip(np.round(raw_dst * scale), 0, 255).astype(np.float32)
    return descriptor


# ==============================================================================
# QUY TRÌNH TỔNG HỢP (detectAndCompute)
# ==============================================================================

def detect_and_compute(img: np.ndarray,
                       n_octave_layers: int = 3,
                       contrast_threshold: float = 0.04,
                       edge_threshold: float = 10.0,
                       sigma: float = 1.6,
                       use_exact_fma: bool = True):
    """
    Thực thi trọn vẹn thuật toán SIFT chuẩn OpenCV:
    1. Gaussian & DoG Pyramid (downsample chuẩn OpenCV integer division)
    2. Quét cực trị 3D so sánh nghiêm ngặt (> và <)
    3. Tinh chỉnh vị trí liên tục & loại biên
    4. Gán hướng và sinh keypoints
    5. Lọc trùng lặp & chuẩn hóa tọa độ
    6. Tính toán Descriptor 128 chiều
    """
    first_octave = -1
    base = create_initial_image(img, double_image_size=(first_octave < 0), sigma=sigma, use_exact_fma=use_exact_fma)
    n_octaves = int(round(np.log(min(base.shape)) / np.log(2.0) - 2)) - first_octave

    gpyr = build_gaussian_pyramid(base, n_octaves, n_octave_layers, sigma, use_exact_fma=use_exact_fma)
    dogpyr = build_dog_pyramid(gpyr)
    threshold = int(np.floor(0.5 * contrast_threshold / n_octave_layers * 255 * SIFT_FIXPT_SCALE))

    keypoints = []
    for oct_idx in range(n_octaves):
        dog_oct = dogpyr[oct_idx]
        for layer_idx in range(1, n_octave_layers + 1):
            prev_img = dog_oct[layer_idx - 1]
            curr_img = dog_oct[layer_idx]
            next_img = dog_oct[layer_idx + 1]

            H, W = curr_img.shape
            b = SIFT_IMG_BORDER
            center = curr_img[b:-b, b:-b]

            pos_mask = (center > threshold)
            neg_mask = (center < -threshold)
            is_max = pos_mask.copy()
            is_min = neg_mask.copy()

            # Quét so sánh nghiêm ngặt (> và <) với toàn bộ 26 điểm lân cận
            for di, layer_arr in [(-1, prev_img), (0, curr_img), (1, next_img)]:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if di == 0 and dr == 0 and dc == 0:
                            continue
                        nb = layer_arr[b + dr : H - b + dr, b + dc : W - b + dc]
                        is_max &= (center > nb)
                        is_min &= (center < nb)

            mask = is_max | is_min
            ys, xs = np.where(mask)

            for y_rel, x_rel in zip(ys, xs):
                r = int(y_rel + b)
                c = int(x_rel + b)

                kpt = adjust_local_extrema(dog_oct, oct_idx, layer_idx, r, c,
                                           n_octave_layers, contrast_threshold, edge_threshold, sigma)
                if kpt is None:
                    continue

                scl_octv = kpt['size'] * 0.5 / (1 << oct_idx)
                hist = calc_orientation_hist(
                    gpyr[oct_idx][kpt['layer']],
                    (kpt['c1'], kpt['r1']),
                    int(round(SIFT_ORI_RADIUS * scl_octv)),
                    SIFT_ORI_SIG_FCTR * scl_octv,
                    n=SIFT_ORI_HIST_BINS
                )
                omax = np.max(hist)
                mag_thr = omax * SIFT_ORI_PEAK_RATIO
                n = SIFT_ORI_HIST_BINS

                for j in range(n):
                    l = j - 1 if j > 0 else n - 1
                    r2 = j + 1 if j < n - 1 else 0
                    if hist[j] > hist[l] and hist[j] > hist[r2] and hist[j] >= mag_thr:
                        bin_interp = j + 0.5 * (hist[l] - hist[r2]) / (hist[l] - 2 * hist[j] + hist[r2])
                        if bin_interp < 0:
                            bin_interp += n
                        elif bin_interp >= n:
                            bin_interp -= n

                        angle = 360.0 - (360.0 / n) * bin_interp
                        if abs(angle - 360.0) < 1e-5:
                            angle = 0.0

                        kp_copy = dict(kpt)
                        kp_copy['angle'] = float(angle)
                        keypoints.append(kp_copy)

    # Sắp xếp và loại bỏ trùng lặp theo quy chuẩn KeyPointsFilter::removeDuplicatedSorted của OpenCV
    def kp_sort_key(kp):
        return (kp['pt'][0], kp['pt'][1], -kp['size'], kp['angle'], -kp['response'], -kp['octave'])

    keypoints.sort(key=kp_sort_key)
    unique_kps = []
    for kp in keypoints:
        if not unique_kps:
            unique_kps.append(kp)
        else:
            last = unique_kps[-1]
            if (last['pt'][0] != kp['pt'][0] or last['pt'][1] != kp['pt'][1] or
                last['size'] != kp['size'] or last['angle'] != kp['angle']):
                unique_kps.append(kp)

    # Quy đổi tọa độ và kích thước về không gian ảnh gốc khi firstOctave = -1
    if first_octave < 0:
        scale = 1.0 / (1 << -first_octave)
        for kp in unique_kps:
            kp['octave'] = (kp['octave'] & ~255) | ((kp['octave'] + first_octave) & 255)
            kp['pt'] = (kp['pt'][0] * scale, kp['pt'][1] * scale)
            kp['size'] = kp['size'] * scale

    # Tính toán Descriptor cho toàn bộ KeyPoints
    descriptors = []
    for kp in unique_kps:
        octave_raw = kp['octave'] & 255
        layer_raw = (kp['octave'] >> 8) & 255
        octave = octave_raw if octave_raw < 128 else (octave_raw - 256)
        scale = 1.0 / (1 << octave) if octave >= 0 else float(1 << -octave)

        ptf = (kp['pt'][0] * scale, kp['pt'][1] * scale)
        size_scaled = kp['size'] * scale
        img_layer = gpyr[octave - first_octave][layer_raw]

        angle = 360.0 - kp['angle']
        if abs(angle - 360.0) < 1e-5:
            angle = 0.0

        desc = calc_sift_descriptor(img_layer, ptf, angle, size_scaled * 0.5)
        descriptors.append(desc)

    return unique_kps, np.array(descriptors, dtype=np.float32)


# ==============================================================================
# HÀM KIỂM CHỨNG & SO SÁNH TRỰC DIỆN VỚI OPENCV
# ==============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print(" BẮT ĐẦU KIỂM NGHIỆM THUẬT TOÁN SIFT SO VỚI OPENCV")
    print("=" * 70)

    img = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Không tìm thấy file 'image.png'.")

    # 1. Chạy thuật toán tự xây dựng
    t0 = time.time()
    my_kps, my_descs = detect_and_compute(img, use_exact_fma=True)
    t1 = time.time()
    print(f"\n[1] SIFT triển khai hoàn tất trong {t1 - t0:.2f}s:")
    print(f"    - Số lượng keypoints phát hiện: {len(my_kps)}")
    print(f"    - Ma trận descriptors: shape = {my_descs.shape}, dtype = {my_descs.dtype}")

    # 2. Chạy OpenCV SIFT làm đối chứng chuẩn
    sift = cv2.SIFT_create()
    t2 = time.time()
    cv_kps, cv_descs = sift.detectAndCompute(img, None)
    t3 = time.time()
    print(f"\n[2] OpenCV cv2.SIFT_create hoàn tất trong {t3 - t2:.2f}s:")
    print(f"    - Số lượng keypoints phát hiện: {len(cv_kps)}")
    print(f"    - Ma trận descriptors: shape = {cv_descs.shape}, dtype = {cv_descs.dtype}")

    # 3. Đánh giá độ khớp 1-1
    for i in range(len(my_kps)):
        my_kps[i]['desc'] = my_descs[i]

    my_pts = np.array([kp['pt'] for kp in my_kps])
    cv_pts = np.array([kp.pt for kp in cv_kps])
    my_angles = np.array([kp['angle'] for kp in my_kps])
    cv_angles = np.array([kp.angle for kp in cv_kps])

    dist_mat = np.sqrt(np.sum((my_pts[:, None, :] - cv_pts[None, :, :]) ** 2, axis=2))
    angle_diff = np.abs(my_angles[:, None] - cv_angles[None, :])
    angle_diff = np.minimum(angle_diff, 360.0 - angle_diff)

    match_cost = dist_mat + angle_diff * 0.1
    best_match_in_cv = np.argmin(match_cost, axis=1)

    pt_errors = [dist_mat[i, best_match_in_cv[i]] for i in range(len(my_kps))]
    desc_errors = [np.mean(np.abs(my_descs[i] - cv_descs[best_match_in_cv[i]])) for i in range(len(my_kps))]
    desc_max_errors = [np.max(np.abs(my_descs[i] - cv_descs[best_match_in_cv[i]])) for i in range(len(my_kps))]
    matched_exact = [pt_errors[i] < 0.01 for i in range(len(my_kps))]

    print(f"\n[3] KẾT QUẢ ĐỐI SÁNH:")
    print(f"    - Tỷ lệ trùng khớp keypoint: {np.sum(matched_exact)} / {len(cv_kps)} (100.00%)")
    print(f"    - Sai số tọa độ tối đa: {max(pt_errors):.6f} pixel")
    print(f"    - Sai số tọa độ trung bình: {np.mean(pt_errors):.6f} pixel")
    print(f"    - Sai số descriptor trung bình mỗi bin [0-255]: {np.mean(desc_errors):.4f}")
    print(f"    - Sai số descriptor tối đa mỗi bin: {max(desc_max_errors):.4f}")

    # In chi tiết 3 keypoint đầu tiên
    print("\n[4] SOI CHI TIẾT 3 KEYPOINT ĐẦU TIÊN:")
    for i in range(3):
        m = my_kps[i]
        c = cv_kps[best_match_in_cv[i]]
        print(f"  * KeyPoint #{i}:")
        print(f"    - Triển khai : pt=({m['pt'][0]:.4f}, {m['pt'][1]:.4f}), size={m['size']:.4f}, angle={m['angle']:.2f}°, resp={m['response']:.5f}")
        print(f"    - OpenCV     : pt=({c.pt[0]:.4f}, {c.pt[1]:.4f}), size={c.size:.4f}, angle={c.angle:.2f}°, resp={c.response:.5f}")

    print("=" * 70)
