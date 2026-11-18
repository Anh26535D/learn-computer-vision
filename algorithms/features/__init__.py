"""
Feature Detection & Description Package
Chứa các thuật toán trích xuất đặc trưng điểm: SIFT, (tương lai: ORB, Harris, FAST, BRIEF,...)
"""

from .sift import PureC_SIFT, sift_numpy_detect_and_compute

__all__ = [
    "PureC_SIFT",
    "sift_numpy_detect_and_compute",
]
