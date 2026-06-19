# Dự án Khai phá Dữ liệu: Dự đoán Kết quả Học tập Sinh viên (UTH)

Dự án này xây dựng hệ thống phục vụ cho bài toán phân tích và dự đoán điểm số cuối kỳ (G3) của sinh viên dựa trên tập dữ liệu học tập và thói quen sinh hoạt từ file [student_data.csv](data/student_data.csv).

Dự án được thiết kế theo mô hình **Pluggable Architecture** (Kiến trúc cắm rút), cho phép mỗi thành viên trong nhóm tự huấn luyện mô hình của mình độc lập (như Random Forest, SVM, XGBoost, Mạng Nơ-ron...) rồi tích hợp trực tiếp vào cấu trúc backend đã dựng sẵn mà không cần thay đổi code lõi của hệ thống.

---

## 📂 Cấu trúc thư mục

```text
student-performance-mining-uth/
├── backend/          # Backend API (Python + FastAPI)
│   ├── app/          # Mã nguồn chính của ứng dụng
│   │   ├── config.py # Cấu hình môi trường (Host, Port, CORS...)
│   │   ├── main.py   # File khởi tạo FastAPI server chính
│   │   ├── models/   # Nơi các thành viên lưu file mô hình đã train (.pkl, .joblib...)
│   │   ├── routers/  # Các API endpoints (/predict, /models)
│   │   ├── schemas/  # Pydantic schemas chuẩn hóa cấu trúc dữ liệu sinh viên
│   │   └── services/ # Logic tiền xử lý chung và nạp/dự đoán của các mô hình
│   ├── run.py        # File khởi động nhanh Backend
│   └── README.md     # Hướng dẫn chi tiết chạy & tích hợp mô hình cho Backend
│
├── frontend/         # Giao diện Web (ReactJS + Vite + Tailwind CSS)
│   ├── src/          # Mã nguồn chứa giao diện chính (App.jsx)
│   ├── package.json  # Danh sách các thư viện Node.js
│   └── tailwind.config.js # Cấu hình CSS
│
├── data/             # Thư mục chứa tập dữ liệu gốc (student_data.csv)
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

Khi một thành viên hoàn thành huấn luyện mô hình của mình trong Jupyter Notebook:
1. Lưu mô hình dưới dạng file (ví dụ: `random_forest.pkl`, `svm.pkl`...).
2. Bỏ file đó vào thư mục `backend/app/saved_models/`.
3. Mở file dịch vụ tương ứng tại `backend/app/services/models/{your_model}.py` và dán logic dự đoán/tiền xử lý cụ thể của bạn vào hàm `predict`.
4. Khởi động server, API sẽ tự động chuyển trạng thái của mô hình từ **Giả lập (Mock/Placeholder)** sang **Thật (Active)**.
*Đọc chi tiết hướng dẫn lập trình và tích hợp tại [backend/README.md](backend/README.md).*