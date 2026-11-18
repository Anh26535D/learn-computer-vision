"""
Computer Vision Playground - Main Entrypoint
Ví dụ minh họa cách sử dụng các thuật toán đã triển khai (bắt đầu với SIFT).
"""

import sys
from pathlib import Path
import cv2

# Đảm bảo import được module algorithms
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from algorithms.features.sift import PureC_SIFT

def demo_sift():
    print("=" * 60)
    print(" HỌC THỊ GIÁC MÁY TÍNH - DEMO THUẬT TOÁN SIFT THUẦN C")
    print("=" * 60)

    img_path = ROOT_DIR / "assets" / "images" / "image.png"
    if not img_path.exists():
        print(f"[!] Không tìm thấy ảnh tại: {img_path}")
        return

    img_color = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

    print(f"[*] Đang xử lý ảnh: {img_path.name} ({gray.shape[1]}x{gray.shape[0]})...")
    
    # 1. Khởi tạo SIFT C thuần
    sift_c = PureC_SIFT()
    
    # 2. Trích xuất đặc trưng
    kps, descs = sift_c.detectAndCompute(gray)
    print(f"[✓] Đã phát hiện {len(kps)} keypoints!")
    print(f"[✓] Ma trận Descriptors: shape={descs.shape}, dtype={descs.dtype}")

    # 3. Vẽ keypoints phong phú (hiển thị kích thước scale và hướng xoay)
    out_img = cv2.drawKeypoints(
        img_color, kps, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        color=(0, 255, 0)
    )

    out_path = ROOT_DIR / "assets" / "outputs" / "sift_keypoints_visualized.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out_img)
    print(f"[✓] Đã lưu ảnh minh họa keypoints tại: {out_path.relative_to(ROOT_DIR)}")
    print("=" * 60)

if __name__ == '__main__':
    demo_sift()
