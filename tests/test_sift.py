"""
Unit Tests cho module SIFT (Pure C & Pure NumPy) sử dụng unittest tiêu chuẩn
"""

import sys
import unittest
from pathlib import Path
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from algorithms.features.sift import PureC_SIFT

class TestSIFT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        img_path = ROOT_DIR / "assets" / "images" / "image.png"
        if not img_path.exists():
            cls.img = np.zeros((200, 200), dtype=np.uint8)
            cv2.circle(cls.img, (100, 100), 40, 255, -1)
            cv2.rectangle(cls.img, (30, 30), (80, 80), 180, -1)
        else:
            cls.img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

    def test_pure_c_sift_runs(self):
        sift = PureC_SIFT()
        kps, descs = sift.detectAndCompute(self.img)
        self.assertGreater(len(kps), 0, "Không phát hiện được keypoint nào")
        self.assertEqual(descs.shape[0], len(kps), "Số lượng descriptor không khớp với keypoints")
        self.assertEqual(descs.shape[1], 128, "Descriptor không đúng 128 chiều")

    def test_pure_c_vs_opencv(self):
        sift_c = PureC_SIFT()
        sift_cv = cv2.SIFT_create()

        c_kps, c_descs = sift_c.detectAndCompute(self.img)
        cv_kps, cv_descs = sift_cv.detectAndCompute(self.img, None)

        # Đảm bảo số lượng keypoints lệch không quá 1 điểm (do FMA)
        self.assertLessEqual(abs(len(c_kps) - len(cv_kps)), 1)

        # Kiểm tra tỷ lệ trùng khớp vị trí < 0.5 pixel
        c_pts = np.array([kp.pt for kp in c_kps])
        cv_pts = np.array([kp.pt for kp in cv_kps])

        dist_mat = np.sqrt(np.sum((c_pts[:, None, :] - cv_pts[None, :, :]) ** 2, axis=2))
        min_dist = np.min(dist_mat, axis=1)
        matched = np.sum(min_dist < 0.5)

        match_rate = matched / len(cv_kps)
        self.assertGreaterEqual(match_rate, 0.995, f"Tỷ lệ trùng khớp vị trí quá thấp: {match_rate * 100:.2f}%")

if __name__ == '__main__':
    unittest.main()
