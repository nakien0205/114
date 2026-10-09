import sys
from pathlib import Path
import torch
import torch.nn as nn

# Thêm thư mục gốc vào path để import src
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import ultralytics.nn.modules as modules
import ultralytics.nn.tasks as tasks
from ultralytics import YOLO

# 1. Import và đăng ký DWTBackboneL3P2 từ repo của nhóm
from src.modules.dwt.backbone import DWTBackboneL3P2

setattr(modules, "DWTBackboneL3P2", DWTBackboneL3P2)
setattr(tasks, "DWTBackboneL3P2", DWTBackboneL3P2)


# 2. Định nghĩa / Đăng ký class Index (dùng để tách tensor P2, P3, P4, P5)
# 2. Định nghĩa class Index tương thích mọi cách viết trong YAML
class Index(nn.Module):
    """Trích xuất tensor từ tuple trả về của DWT Backbone (P2, P3, P4, P5).
    Tương thích cả [idx] hoặc [channels, idx].
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 1:
            # Nếu YAML viết dạng: Index, [0]
            self.idx = int(args[0])
        elif len(args) >= 2:
            # Nếu YAML viết dạng: Index, [32, 0]
            self.idx = int(args[1])
        elif "idx" in kwargs:
            self.idx = int(kwargs["idx"])
        else:
            self.idx = 0

    def forward(self, x):
        return x[self.idx]

setattr(modules, "Index", Index)
setattr(tasks, "Index", Index)


# 3. Định nghĩa / Đăng ký SpatialRelationalCoAttention (SRC)
class SpatialRelationalCoAttention(nn.Module):
    def __init__(self, c1, num_heads=8, sr_ratio=8):
        super().__init__()
        self.c1 = c1
        self.num_heads = num_heads
        self.head_dim = c1 // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_conv = nn.Conv2d(c1, c1, kernel_size=1, bias=False)
        self.k_conv = nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False)
        self.v_conv = nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(c1, c1, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.BatchNorm2d(c1)

        self.proj = nn.Conv2d(c1, c1, kernel_size=1)
        self.act = nn.SiLU()

    def forward(self, x):
        B, C, H, W = x.shape
        # Query
        q = self.q_conv(x).reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)

        # Key & Value với Spatial Reduction
        if self.sr_ratio > 1:
            x_sr = self.norm(self.sr(x))
            _, _, H_sr, W_sr = x_sr.shape
            k = self.k_conv(x_sr).reshape(B, self.num_heads, self.head_dim, H_sr * W_sr).permute(0, 1, 3, 2)
            v = self.v_conv(x_sr).reshape(B, self.num_heads, self.head_dim, H_sr * W_sr).permute(0, 1, 3, 2)
        else:
            k = self.k_conv(x).reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
            v = self.v_conv(x).reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)

        # Scaled Dot-Product Attention
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn = attn.permute(0, 1, 3, 2).reshape(B, C, H, W)

        return x + self.act(self.proj(attn))

setattr(modules, "SpatialRelationalCoAttention", SpatialRelationalCoAttention)
setattr(tasks, "SpatialRelationalCoAttention", SpatialRelationalCoAttention)


# 4. Kiểm tra nạp file YAML kiến trúc
if __name__ == "__main__":
    yaml_path = "models/yolo26n-p2-dwt-src.yaml"
    print(f"--- Đang khởi tạo mô hình từ {yaml_path} ---")
    
    model = YOLO(yaml_path)
    model.info(detailed=True)
    
    print("\n✅ THÀNH CÔNG: Mô hình YOLO26n-P2 kết hợp DWT + SRC đã khởi tạo chuẩn xác!")