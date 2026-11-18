# Computer Vision From Scratch: Pure C & Pure NumPy

Dự án nghiên cứu, tự triển khai từ đầu (from scratch) các thuật toán Thị giác máy tính (Computer Vision) kinh điển bằng **C thuần (Pure C99, Scalar, không vectorize)** và **NumPy thuần**, với tiêu chí đối soát kết quả chính xác 100% so với **OpenCV**.

---

## 📁 Cấu trúc thư mục dự án (Architecture)

Dự án được phân cấp rõ ràng theo từng nhóm chuyên đề thuật toán để dễ dàng mở rộng trong tương lai:

```text
learn_computer_vision/
├── assets/                          # Dữ liệu ảnh và kết quả minh họa
│   ├── images/                      # Ảnh mẫu đầu vào (image.png,...)
│   └── outputs/                     # Ảnh xuất kết quả visualize, debug pyramid
│
├── algorithms/                      # Triển khai thuật toán theo từng chuyên mục
│   ├── features/                    # Phát hiện & mô tả điểm đặc trưng (Features)
│   │   ├── sift/                    # Thuật toán SIFT (Scale-Invariant Feature Transform)
│   │   │   ├── src/                 # Mã nguồn C thuần (Scalar, No-SIMD)
│   │   │   │   ├── sift_c.c
│   │   │   │   └── sift_c.h
│   │   │   ├── lib/                 # Thư viện liên kết động đã biên dịch
│   │   │   │   └── libsift.dll      (hoặc libsift.so trên Linux)
│   │   │   ├── sift_wrapper.py      # Python ctypes wrapper (API tương thích OpenCV)
│   │   │   └── sift_numpy.py        # Bản NumPy thuần (phục vụ học tập từng bước)
│   │   │
│   │   ├── harris/                  # (Dự kiến mở rộng: Harris Corner Detector)
│   │   ├── fast/                    # (Dự kiến mở rộng: FAST Corner Detector)
│   │   └── orb/                     # (Dự kiến mở rộng: ORB - Oriented FAST & Rotated BRIEF)
│   │
│   ├── filters/                     # (Dự kiến mở rộng: Gaussian, Bilateral, Sobel...)
│   └── edges/                       # (Dự kiến mở rộng: Canny Edge Detector, Hough Transform...)
│
├── benchmarks/                      # Đo đạc hiệu năng và đối soát độ chính xác
│   └── benchmark_sift.py            # Benchmark SIFT: OpenCV vs Pure C vs Pure NumPy
│
├── scripts/                         # Tiện ích xây dựng (Build tools)
│   └── build_c.py                   # Script tự động tìm GCC/Clang và build toàn bộ DLL/so
│
├── tests/                           # Unit tests kiểm nghiệm tự động
│   └── test_sift.py                 # Kiểm tra độ chính xác và hồi quy
│
├── main.py                          # Demo chạy thử nghiệm nhanh
└── pyproject.toml
```

---

## 🚀 Hướng dẫn khởi chạy nhanh

### 1. Chạy demo trực tiếp
```bash
python main.py
```
Lệnh trên sẽ đọc `assets/images/image.png`, sử dụng thư viện **Pure C SIFT** để trích xuất toàn bộ Keypoints & Descriptors, sau đó vẽ kết quả trực quan vào `assets/outputs/sift_keypoints_visualized.png`.

### 2. Chạy Benchmark so sánh tốc độ & độ chính xác
```bash
python benchmarks/benchmark_sift.py
```

### 3. Chạy Unit Tests tự động
```bash
python tests/test_sift.py
```

### 4. Biên dịch lại các module C thuần
Nếu bạn chỉnh sửa mã nguồn C (`sift_c.c` hoặc thêm thuật toán mới):
```bash
python scripts/build_c.py
```
> Script sẽ tự động dùng GCC với cờ tối ưu `-O3 -fno-tree-vectorize` (tắt SIMD phần cứng, chỉ dùng scalar loops chuẩn).

---

## 📊 Kết quả Benchmark: Thuật toán SIFT

Thử nghiệm trên ảnh thực tế `assets/images/image.png` ($474 \times 315$):

### 1. Tốc độ thực thi (Speed Benchmark)

| Phiên bản | Thời gian trung bình | FPS | Tăng tốc (Speedup) |
| :--- | :---: | :---: | :---: |
| **OpenCV SIFT** *(AVX2 + Multithread)* | **57.01 ms** | **17.5** | Baseline C++ tối ưu phần cứng |
| **Bản C thuần** *(Scalar, No-SIMD, Single-thread)* | **409.87 ms** | **2.4** | **Nhanh gấp ~50x so với NumPy** |
| **Bản Python / NumPy thuần** | **20,261.49 ms** | **0.05** | Bản mẫu nghiên cứu |

### 2. Độ chính xác so với OpenCV (Accuracy Benchmark)

* **Tỷ lệ trùng khớp vị trí keypoint ($< 0.5$ px):** **1051 / 1051 (100.00%)**
* **Sai số tọa độ trung bình $(\Delta x, \Delta y)$:** **0.000030 pixel**
* **Sai số kích thước Scale trung bình:** **0.000016 pixel**
* **Sai số góc xoay Orientation trung bình:** **0.001279°**
* **Sai số ma trận Descriptor (mỗi bin $[0 - 255]$):** **0.0028** (trùng khớp tuyệt đối khi làm tròn sang số nguyên).

---

## 🛠 Hướng dẫn mở rộng thuật toán mới

Để thêm một thuật toán mới (ví dụ `Harris Corner` hoặc `ORB`):

1. **Tạo thư mục thuật toán:**
   Tạo thư mục mới trong nhánh tương ứng, ví dụ `algorithms/features/harris/`.
2. **Cấu trúc bên trong:**
   - `src/harris.c`, `src/harris.h`: Triển khai C thuần (scalar).
   - `harris_wrapper.py`: Wrapper `ctypes` để gọi từ Python.
   - `harris_numpy.py`: Bản triển khai thuần NumPy để đối chứng giải thuật.
3. **Khai báo trong `scripts/build_c.py`:**
   Thêm thông tin module vào danh sách `MODULES` trong `scripts/build_c.py`:
   ```python
   {
       "name": "Harris",
       "src_dir": PROJECT_ROOT / "algorithms" / "features" / "harris" / "src",
       "sources": ["harris.c"],
       "out_dir": PROJECT_ROOT / "algorithms" / "features" / "harris" / "lib",
       "lib_name": "libharris",
       "cflags": ["-O3", "-fno-tree-vectorize", "-Wall"]
   }
   ```
4. **Tạo Benchmark và Test:**
   - Viết bài đo trong `benchmarks/benchmark_harris.py`.
   - Viết test trong `tests/test_harris.py`.
