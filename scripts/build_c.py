"""
Script tự động biên dịch toàn bộ các module C thuần trong dự án:
- Tìm trình biên dịch GCC / Clang / MSVC có sẵn trên máy.
- Biên dịch C source thành DLL / Shared Library đặt trong thư mục lib/ của từng module.
- Mặc định áp dụng cờ -O3 -fno-tree-vectorize (không dùng SIMD vectorization theo đúng thiết kế học tập).
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Cấu hình danh sách các module C cần biên dịch
MODULES = [
    {
        "name": "SIFT",
        "src_dir": PROJECT_ROOT / "algorithms" / "features" / "sift" / "src",
        "sources": ["sift_c.c"],
        "out_dir": PROJECT_ROOT / "algorithms" / "features" / "sift" / "lib",
        "lib_name": "libsift",
        # Cờ biên dịch: -fno-tree-vectorize để đảm bảo chỉ dùng scalar loop thuần
        "cflags": ["-O3", "-fno-tree-vectorize", "-Wall"]
    }
]

def find_compiler():
    """Tìm trình biên dịch C khả dụng."""
    for cc in ["gcc", "clang"]:
        if shutil.which(cc):
            return cc
    return None

def build_modules():
    cc = find_compiler()
    if not cc:
        print("[LỖI] Không tìm thấy trình biên dịch GCC hoặc Clang trên máy của bạn!")
        print("      Vui lòng cài đặt MinGW/MSYS2 hoặc GCC để biên dịch các module C.")
        sys.exit(1)

    print(f"[+] Tìm thấy trình biên dịch: {cc} ({shutil.which(cc)})")
    
    is_windows = sys.platform.startswith("win")
    ext = ".dll" if is_windows else (".dylib" if sys.platform == "darwin" else ".so")

    for mod in MODULES:
        print(f"\n[*] Đang biên dịch module: {mod['name']}...")
        src_dir = mod["src_dir"]
        out_dir = mod["out_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        
        out_file = out_dir / f"{mod['lib_name']}{ext}"
        src_files = [str(src_dir / s) for s in mod["sources"]]

        cmd = [
            cc,
            *mod["cflags"],
            "-shared",
            "-o", str(out_file),
            *src_files,
            "-lm"
        ]

        print(f"    Lệnh: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"[!] Thất bại khi biên dịch {mod['name']}:")
            print(result.stderr)
            sys.exit(1)
        else:
            print(f"[✓] Đã tạo thành công: {out_file.relative_to(PROJECT_ROOT)} ({out_file.stat().st_size:,} bytes)")

    print("\n[✓] Hoàn tất biên dịch toàn bộ các module C!")

if __name__ == "__main__":
    build_modules()
