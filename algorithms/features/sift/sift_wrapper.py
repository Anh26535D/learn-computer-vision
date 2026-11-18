"""
Python ctypes wrapper cho thư viện C thuần SIFT (libsift.dll)
Không vectorize, cung cấp API tương thích với OpenCV cv2.SIFT_create().
"""

import os
import ctypes
import numpy as np
import cv2

class CSiftKeypoint(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("size", ctypes.c_float),
        ("angle", ctypes.c_float),
        ("response", ctypes.c_float),
        ("octave", ctypes.c_int),
        ("class_id", ctypes.c_int)
    ]

class PureC_SIFT:
    def __init__(self, dll_path: str = None):
        if dll_path is None:
            candidates = [
                os.path.join(os.path.dirname(__file__), "lib", "libsift.dll"),
                os.path.join(os.path.dirname(__file__), "libsift.dll"),
                os.path.join(os.path.dirname(__file__), "lib", "libsift.so"),
                os.path.join(os.path.dirname(__file__), "libsift.so"),
            ]
            for p in candidates:
                if os.path.exists(p):
                    dll_path = p
                    break
        if dll_path is None or not os.path.exists(dll_path):
            raise FileNotFoundError(
                f"Không tìm thấy thư viện SIFT C (libsift.dll / libsift.so). "
                f"Vui lòng chạy 'python scripts/build_c.py' để biên dịch mã nguồn C."
            )
        
        self.lib = ctypes.CDLL(dll_path)
        
        # Prototype:
        # int sift_detect_and_compute(
        #     const unsigned char* img_data, int width, int height,
        #     int n_octave_layers, float contrast_threshold, float edge_threshold, float sigma,
        #     SiftKeypoint** out_kps, float** out_descs, int* out_count
        # );
        self.lib.sift_detect_and_compute.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.POINTER(ctypes.POINTER(CSiftKeypoint)),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
            ctypes.POINTER(ctypes.c_int)
        ]
        self.lib.sift_detect_and_compute.restype = ctypes.c_int
        
        # void sift_free_results(SiftKeypoint* kps, float* descs);
        self.lib.sift_free_results.argtypes = [
            ctypes.POINTER(CSiftKeypoint),
            ctypes.POINTER(ctypes.c_float)
        ]
        self.lib.sift_free_results.restype = None

    def detectAndCompute(self, img: np.ndarray, mask=None,
                         n_octave_layers: int = 3,
                         contrast_threshold: float = 0.04,
                         edge_threshold: float = 10.0,
                         sigma: float = 1.6,
                         return_cv_keypoints: bool = True):
        """
        Trích xuất SIFT KeyPoints và Descriptors bằng thư viện C thuần (scalar).
        Tương thích API của cv2.SIFT_create().detectAndCompute()
        """
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        img = np.ascontiguousarray(img, dtype=np.uint8)
        height, width = img.shape
        
        c_kps = ctypes.POINTER(CSiftKeypoint)()
        c_descs = ctypes.POINTER(ctypes.c_float)()
        count = ctypes.c_int(0)
        
        img_ptr = img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        
        ret = self.lib.sift_detect_and_compute(
            img_ptr, width, height,
            n_octave_layers,
            ctypes.c_float(contrast_threshold),
            ctypes.c_float(edge_threshold),
            ctypes.c_float(sigma),
            ctypes.byref(c_kps),
            ctypes.byref(c_descs),
            ctypes.byref(count)
        )
        
        if ret != 0:
            raise RuntimeError(f"sift_detect_and_compute trả về lỗi mã: {ret}")
        
        n = count.value
        if n == 0:
            return [], np.empty((0, 128), dtype=np.float32)
        
        # Đọc descriptors: copy dữ liệu từ C pointer sang numpy array
        desc_raw = np.ctypeslib.as_array(c_descs, shape=(n, 128)).copy()
        
        # Đọc keypoints
        keypoints = []
        for i in range(n):
            kp = c_kps[i]
            if return_cv_keypoints:
                cv_kp = cv2.KeyPoint(
                    x=float(kp.x),
                    y=float(kp.y),
                    size=float(kp.size),
                    angle=float(kp.angle),
                    response=float(kp.response),
                    octave=int(kp.octave),
                    class_id=int(kp.class_id)
                )
                keypoints.append(cv_kp)
            else:
                keypoints.append({
                    'pt': (float(kp.x), float(kp.y)),
                    'size': float(kp.size),
                    'angle': float(kp.angle),
                    'response': float(kp.response),
                    'octave': int(kp.octave)
                })
        
        # Giải phóng bộ nhớ heap phía C
        self.lib.sift_free_results(c_kps, c_descs)
        
        return keypoints, desc_raw
