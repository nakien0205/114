### 🚀 Cải tiến: Module Spatial-Relational Co-Attention (SRC) cho nhánh P2

#### 1. Đóng góp kỹ thuật
- Tích hợp module **SpatialRelationalCoAttention** vào tầng P2 (`yolo26n-p2-src.yaml`)[cite: 5].
- Ràng buộc tương quan không gian đồng xuất hiện giữa khói và lửa (Physical Co-occurrence Prior)[cite: 1, 5].
- Tối ưu bộ nhớ bằng spatial reduction (`sr_ratio=8`)[cite: 5], giảm thiểu nguy cơ OOM và duy trì tốc độ thời gian thực[cite: 5].

#### 2. Kết quả thực nghiệm (50 Epochs trên GPU T4)
- **mAP50:** **67.5%** (Tăng **+1.6%** so với baseline YOLO26n-P2 là 65.90%)
- **Precision:** **69.3%** (Tăng mạnh, dập tắt các trường hợp báo động giả)
- **Recall:** **61.4%**
- **mAP50-95:** **36.8%**
- **Độ trễ (Inference Latency):** **6.8 ms** (~147 FPS)
- **Kích thước mô hình:** **2.44M params** | **7.8 GFLOPs**

#### 3. Tài nguyên bàn giao
- Cấu hình kiến trúc: `yolo26n-p2-src.yaml`[cite: 5]
- Mã nguồn module: `SpatialRelationalCoAttention`[cite: 5]
- Checkpoint đã huấn luyện: `best.pt` (5.7MB)
- Biểu đồ và nhật ký: `results.csv`, `results.png`, `BoxPR_curve.png`, `confusion_matrix_normalized.png`
