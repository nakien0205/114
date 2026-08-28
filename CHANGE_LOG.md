# Change Log
last update: 19h00 28/8/2026

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