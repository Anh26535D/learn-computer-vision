"""
SIFT (Scale-Invariant Feature Transform) Module
Cung cấp hai bản triển khai:
1. PureC_SIFT: Bản C thuần tối ưu hiệu năng cao (scalar, không SIMD vectorization), gọi qua ctypes.
2. SiftNumPy: Bản triển khai thuần Python/NumPy phục vụ mục đích nghiên cứu, học tập từng bước.
"""

from .sift_wrapper import PureC_SIFT
from .sift_numpy import detect_and_compute as sift_numpy_detect_and_compute

__all__ = [
    "PureC_SIFT",
    "sift_numpy_detect_and_compute",
]
