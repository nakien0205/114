# Change Log
### 20h00 6/9/2026 - Hân Đồng
- Train và so sánh 3 model: `yolov8n`, `yolo11n`, `yolo26n` bằng dataset "DFire"
- Eval `yolo11n` và `yolo26n` bằng dataset "Indoor Fire and Smoke" và "DFS"
- Kết quả:

Baseline comparison:

| Model | Dataset | Precision | Recall | mAP50 | mAP50-95 | inference_ms | fps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| yolov8n | DFire | 0.63705 | 0.55563 | 0.47418 | 0.24156 | 3.584 | 211.32 |
| yolo11n | DFire | 0.714 | 0.674 | 0.61 | 0.342 | 3.66 | 208.75 |
| yolo26n | DFire | **0.72933** | **0.67796** | **0.62782** | **0.36487** | 3.552 | 234.02 |

Domain adaptation test:

| Model | Dataset | Precision | Recall | mAP50 | mAP50-95 | inference_ms | fps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| yolo11n | Indoor | 0.45202 | 0.29706 | 0.24102 | 0.11635 | 5.896 | 132.04 |
| yolo26n | Indoor | 0.50927 | 0.25554 | 0.2097 | 0.10709 | 5.301 | 150.39 |

Domain adaptation test:

| Model | Dataset | Precision | Recall | mAP50 | mAP50-95 | inference_ms | fps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| yolo11n | DFS | 0.3784 | 0.31269 | 0.20877 | 0.08435 | 4.821 | 160.96 |
| yolo26n | DFS | 0.41136 | 0.31734 | 0.22778 | 0.09403 | 4.928 | 170.98 |


---

### 19h00 28/8/2026 - Hân Đồng

- Để các tất cả dataset vào `datasets/`, chia folder manifests/ và splits/ theo từng dataset.
- Chỉnh sửa audit, scanner, isolator, partitioner, manifest và verifier để có thể sử dụng cho các bất kì dataset. Sau khi chạy dataset pipeline, tự động tạo folder manifests/ và splits/ bên trong folder dataset tương ứng.

Ví dụ:
```
datasets/                   # Đã được thêm vào .gitignore
├──DFire/
│   ├── train/
│   ├── test/ 
│   ├── valid/ 
│   ├── manifest/           # Tự động tạo sau khi chạy dataset pipeline
│   └── splits/             # Tự động tạo sau khi chạy dataset pipeline
│
├──Indoor_Fire_and_Smoke/
│   ├── train/
│   ├── test/ 
│   ├── valid/ 
│   ├── manifest/           # Tự động tạo sau khi chạy dataset pipeline
│   └── splits/             # Tự động tạo sau khi chạy dataset pipeline
│
```
 
- Thêm `config.yaml` để cấu hình dùng chung cho dự án. Xóa các đường dẫn tuyệt đối để không phụ thuộc máy local.
- Cập nhật code CLI, train, evaluate và inference để phù hợp với những thay đổi ở `src/fire_audit/`.