"""
Chương trình Benchmark toàn diện:
1. OpenCV SIFT (C++ SIMD/AVX2 multithreaded)
2. Pure C SIFT (Bản C thuần, Scalar, không vectorize, GCC -fno-tree-vectorize)
3. Pure NumPy SIFT (Bản Python/NumPy thuần)
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Đảm bảo import được các module từ thư mục gốc algorithms
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from algorithms.features.sift import PureC_SIFT, sift_numpy_detect_and_compute

def run_benchmark():
    print("=" * 75)
    print(" BENCHMARK: PURE C (NON-VECTORIZED) vs OPENCV vs PURE NUMPY")
    print("=" * 75)

    img_path = ROOT_DIR / "assets" / "images" / "image.png"
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Không tìm thấy file ảnh tại: {img_path}")

    H, W = img.shape
    print(f"Ảnh đầu vào: {W} x {H} pixels (Grayscale)")

    # Khởi tạo các module
    sift_cv = cv2.SIFT_create()
    sift_c = PureC_SIFT()

    # -------------------------------------------------------------
    # 1. Warm-up & Đo thời gian thực thi (Execution Time)
    # -------------------------------------------------------------
    print("\n[1] ĐO TỐC ĐỘ THỰC THI (Trung bình 5 lần chạy):")

    # A. OpenCV
    # Warmup
    sift_cv.detectAndCompute(img, None)
    t_start = time.perf_counter()
    n_runs = 5
    for _ in range(n_runs):
        cv_kps, cv_descs = sift_cv.detectAndCompute(img, None)
    cv_time = (time.perf_counter() - t_start) / n_runs

    # B. Pure C (libsift.dll - scalar, no vectorization)
    sift_c.detectAndCompute(img)
    t_start = time.perf_counter()
    for _ in range(n_runs):
        c_kps, c_descs = sift_c.detectAndCompute(img)
    c_time = (time.perf_counter() - t_start) / n_runs

    # C. Pure NumPy
    # Chỉ chạy 1 lần vì NumPy thuần mất khoảng 5-7 giây
    t_start = time.perf_counter()
    np_kps, np_descs = sift_numpy_detect_and_compute(img, use_exact_fma=True)
    np_time = time.perf_counter() - t_start

    print(f"  + OpenCV C++ (AVX2 + Multithread) : {cv_time * 1000:7.2f} ms ({1.0 / cv_time:5.1f} FPS)")
    print(f"  + Bản C thuần (Scalar, No-SIMD)    : {c_time * 1000:7.2f} ms ({1.0 / c_time:5.1f} FPS)")
    print(f"  + Bản Python/NumPy thuần           : {np_time * 1000:7.2f} ms ({1.0 / np_time:5.2f} FPS)")
    print(f"\n  => Tốc độ Bản C thuần nhanh gấp {(np_time / c_time):.1f}x so với bản Python/NumPy!")
    print(f"  => Tốc độ Bản C thuần chậm hơn {(c_time / cv_time):.1f}x so với OpenCV (do OpenCV dùng AVX2 SIMD + OpenMP).")

    # -------------------------------------------------------------
    # 2. ĐỐI SOÁT ĐỘ CHÍNH XÁC (Accuracy Comparison: Pure C vs OpenCV)
    # -------------------------------------------------------------
    print("\n[2] ĐỐI SOÁT ĐỘ CHÍNH XÁC (Pure C vs OpenCV):")
    print(f"  - Số lượng Keypoints:")
    print(f"      * Pure C  : {len(c_kps)} keypoints")
    print(f"      * OpenCV  : {len(cv_kps)} keypoints")

    c_pts = np.array([kp.pt for kp in c_kps])
    cv_pts = np.array([kp.pt for kp in cv_kps])
    c_angles = np.array([kp.angle for kp in c_kps])
    cv_angles = np.array([kp.angle for kp in cv_kps])
    c_sizes = np.array([kp.size for kp in c_kps])
    cv_sizes = np.array([kp.size for kp in cv_kps])

    # Ghép cặp 1-1 tối ưu theo vị trí và góc xoay
    dist_mat = np.sqrt(np.sum((c_pts[:, None, :] - cv_pts[None, :, :]) ** 2, axis=2))
    angle_diff = np.abs(c_angles[:, None] - cv_angles[None, :])
    angle_diff = np.minimum(angle_diff, 360.0 - angle_diff)

    match_cost = dist_mat + angle_diff * 0.1
    best_match = np.argmin(match_cost, axis=1)

    pt_errors = [dist_mat[i, best_match[i]] for i in range(len(c_kps))]
    size_errors = [abs(c_sizes[i] - cv_sizes[best_match[i]]) for i in range(len(c_kps))]
    ori_errors = [angle_diff[i, best_match[i]] for i in range(len(c_kps))]
    desc_errors = [np.mean(np.abs(c_descs[i] - cv_descs[best_match[i]])) for i in range(len(c_kps))]
    desc_max_errors = [np.max(np.abs(c_descs[i] - cv_descs[best_match[i]])) for i in range(len(c_kps))]

    matched_exact = [pt_errors[i] < 0.5 for i in range(len(c_kps))]
    matched_count = np.sum(matched_exact)

    print(f"  - Tỷ lệ trùng khớp vị trí (< 0.5 px) : {matched_count} / {len(cv_kps)} ({matched_count / len(cv_kps) * 100:.2f}%)")
    
    # Chỉ tính sai số trên tập các điểm trùng khớp
    matched_idx = np.where(matched_exact)[0]
    matched_pt_errs = [pt_errors[i] for i in matched_idx]
    matched_size_errs = [size_errors[i] for i in matched_idx]
    matched_ori_errs = [ori_errors[i] for i in matched_idx]
    matched_desc_errs = [desc_errors[i] for i in matched_idx]
    matched_desc_max = [desc_max_errors[i] for i in matched_idx]

    print(f"  - Sai số tọa độ (pixel):")
    print(f"      * Trung bình : {np.mean(matched_pt_errs):.6f} pixel")
    print(f"      * Cực đại    : {np.max(matched_pt_errs):.6f} pixel")
    print(f"  - Sai số kích thước Scale (size):")
    print(f"      * Trung bình : {np.mean(matched_size_errs):.6f} pixel")
    print(f"      * Cực đại    : {np.max(matched_size_errs):.6f} pixel")
    print(f"  - Sai số góc xoay Orientation (độ):")
    print(f"      * Trung bình : {np.mean(matched_ori_errs):.6f}°")
    print(f"      * Cực đại    : {np.max(matched_ori_errs):.6f}°")
    print(f"  - Sai số Descriptor (mỗi bin trong thang [0 - 255]):")
    print(f"      * Trung bình mỗi bin : {np.mean(matched_desc_errs):.4f}")
    print(f"      * Cực đại mỗi bin    : {np.max(matched_desc_max):.4f}")

    # -------------------------------------------------------------
    # 3. SOI CHI TIẾT 3 KEYPOINT ĐẦU TIÊN
    # -------------------------------------------------------------
    print("\n[3] SOI CHI TIẾT 3 KEYPOINT ĐẦU TIÊN (Pure C vs OpenCV):")
    for i in range(3):
        ck = c_kps[i]
        cvk = cv_kps[best_match[i]]
        print(f"  * Keypoint #{i}:")
        print(f"    - Pure C : pt=({ck.pt[0]:.4f}, {ck.pt[1]:.4f}), size={ck.size:.4f}, angle={ck.angle:.2f}°, resp={ck.response:.5f}")
        print(f"    - OpenCV : pt=({cvk.pt[0]:.4f}, {cvk.pt[1]:.4f}), size={cvk.size:.4f}, angle={cvk.angle:.2f}°, resp={cvk.response:.5f}")

    print("=" * 75)

if __name__ == '__main__':
    run_benchmark()
