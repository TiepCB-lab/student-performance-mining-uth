# Dự án Khai phá Dữ liệu: Dự đoán Kết quả Học tập Sinh viên (UTH)

Dự án này xây dựng khung Backend API nhằm phục vụ cho bài toán phân tích và dự đoán điểm số cuối kỳ (G3) của sinh viên dựa trên tập dữ liệu học tập và thói quen sinh hoạt từ file [Student_Performance.csv](data/raw/Student_Performance.csv).

Dự án được thiết kế theo mô hình **Pluggable Architecture** (Kiến trúc cắm rút), cho phép mỗi thành viên trong nhóm tự huấn luyện mô hình của mình độc lập (như Random Forest, SVM, XGBoost, Mạng Nơ-ron...) rồi tích hợp trực tiếp vào cấu trúc backend đã dựng sẵn mà không cần thay đổi code lõi của hệ thống.

---

## 📂 Cấu trúc thư mục dự án

```text
student-performance-mining-uth/
├── backend/          # Backend API (Python + FastAPI)
│   ├── app/          # Mã nguồn chính của ứng dụng
│   │   ├── core/     # Chứa cấu hình lõi (config.py)
│   │   ├── routes/   # Các API endpoints (/predict, /models)
│   │   ├── schemas/  # Pydantic schemas dữ liệu sinh viên
│   │   └── services/ # Logic tiền xử lý và nạp/dự đoán của mô hình
│   └── run.py        # File khởi động nhanh Backend
│
├── frontend/         # Thư mục chứa mã nguồn giao diện Web
├── data/             # Thư mục chứa tập dữ liệu
│   └── raw/          # Lưu trữ tập dữ liệu thô ban đầu (Student_Performance.csv)
├── models/           # Thư mục lưu trữ các file mô hình đã train (.pkl, .joblib...)
├── notebooks/        # Thư mục lưu trữ Jupyter Notebook để EDA/Thử nghiệm nháp
├── training/         # Luồng huấn luyện & đánh giá mô hình dạng module sản xuất
│   └── train.py
├── reports/          # Thư mục lưu trữ biểu đồ và báo cáo đánh giá mô hình
│   ├── charts/
│   └── metrics/
└── README.md         # Hướng dẫn tổng quan (File này)
```

---

## 🚀 Hướng dẫn khởi chạy nhanh Backend API

Để chạy API phục vụ dự đoán trên máy cục bộ, hãy làm theo hướng dẫn dưới đây:

### Yêu cầu
- Python 3.8+ (khuyên dùng Python 3.10 hoặc 3.11).

### Các bước khởi chạy:
--Khởi chạy backend
1. Mở một terminal tại thư mục `backend/`.
2. Khởi tạo môi trường ảo (khuyên dùng):
   ```bash
   python -m venv venv
   # Kích hoạt trên Windows:
   .\venv\Scripts\activate
   # Hoặc trên macOS/Linux:
   source venv/bin/activate
   ```
3. Cài đặt các thư viện:
   ```bash
   pip install -r requirements.txt
   ```
4. Khởi chạy server:
   ```bash
   python run.py
   ```
   API sẽ chạy tại địa chỉ [http://127.0.0.1:8000](http://127.0.0.1:8000).
   Bạn có thể xem tài liệu tự động và thử nghiệm trực tiếp tại: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

-- Khởi chạy frontend
⚠️ Lưu ý: Đảm bảo bạn vẫn đang giữ Terminal của Backend chạy ngầm. Hãy mở một Terminal mới để thực hiện các bước sau:

1. Di chuyển vào thư mục frontend:
Bash
cd frontend

2. Cài đặt các gói thư viện Node.js:
Bash
npm install

3. Khởi chạy giao diện:
Bash
npm run dev
💡 Giao diện web sẽ hoạt động tại địa chỉ hiển thị trên Terminal (thường là http://localhost:5173). Dữ liệu nhập từ web sẽ gọi thẳng vào Backend API ở cổng 8000.

---

## 🤝 Hướng dẫn làm việc nhóm & Tích hợp mô hình

Khi một thành viên hoàn thành huấn luyện mô hình của mình trong Jupyter Notebook hoặc script training:
1. Lưu mô hình dưới dạng file (ví dụ: `random_forest.pkl`, `svm.pkl`...).
2. Bỏ file đó vào thư mục `models/` ở gốc dự án.
3. Mở file dịch vụ tương ứng tại `backend/app/services/models/{your_model}.py` và dán logic dự đoán/tiền xử lý cụ thể của bạn vào hàm `predict`.
4. Khởi động server, API sẽ tự động chuyển trạng thái của mô hình từ **Giả lập (Mock/Placeholder)** sang **Thật (Active)**.
*Đọc chi tiết hướng dẫn lập trình và tích hợp tại [backend/README.md](backend/README.md).*