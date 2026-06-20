# Dự án Khai phá Dữ liệu: Dự đoán Kết quả Học tập Sinh viên (UTH)

Dự án này xây dựng hệ thống phân tích và dự đoán xếp hạng học lực (A–F) của sinh viên dựa trên tập dữ liệu học tập và thói quen sinh hoạt từ file [Student_Performance.csv](data/raw/Student_Performance.csv).

Dự án được thiết kế theo mô hình **Pluggable Architecture** (Kiến trúc cắm rút), cho phép mỗi thành viên trong nhóm tự huấn luyện mô hình của mình độc lập (như Random Forest, SVM, XGBoost, Mạng Nơ-ron...) rồi tích hợp trực tiếp vào cấu trúc backend đã dựng sẵn mà không cần thay đổi code lõi của hệ thống.

---

## 📂 Cấu trúc thư mục dự án

```text
student-performance-mining-uth/
├── backend/                # Backend API (Python + FastAPI)
│   ├── app/                # Mã nguồn chính của ứng dụng
│   │   ├── core/           # Chứa cấu hình lõi (config.py)
│   │   ├── routes/         # Các API endpoints (/predict, /models)
│   │   ├── schemas/        # Pydantic schemas dữ liệu sinh viên
│   │   └── services/       # Logic tiền xử lý và nạp/dự đoán của mô hình
│   ├── run.py              # File khởi động nhanh Backend
│   └── README.md           # Hướng dẫn chi tiết phát triển Backend
│
├── frontend/               # Giao diện Web (ReactJS + Vite + Tailwind CSS)
│   ├── src/                # Mã nguồn giao diện chính (App.jsx)
│   ├── package.json        # Danh sách các thư viện Node.js
│   └── tailwind.config.js  # Cấu hình Tailwind CSS
│
├── data/                   # Thư mục chứa tập dữ liệu
│   ├── raw/                # Dữ liệu thô ban đầu (Student_Performance.csv)
│   └── processed/          # Dữ liệu đã qua tiền xử lý
│
├── models/                 # File mô hình đã train (.pkl, .joblib...)
│
├── notebooks/              # Jupyter Notebooks theo pipeline MLOps
│   ├── 01_data_understanding.ipynb   # Tìm hiểu dữ liệu
│   ├── 02_eda.ipynb                  # Phân tích khám phá (EDA)
│   ├── 03_preprocessing.ipynb        # Tiền xử lý dữ liệu
│   ├── models/                       # Thư mục chứa notebook huấn luyện
│   │   └── 04_train_random_forest.ipynb
│   └── 05_model_comparison.ipynb     # So sánh hiệu năng các mô hình
│
├── training/               # Script huấn luyện dạng production
│   └── train.py
│
├── reports/                # Biểu đồ và báo cáo đánh giá mô hình
│   ├── charts/
│   └── metrics/
│
├── .gitignore              # Cấu hình Git bỏ qua file cache/model nặng
├── requirements.txt        # Quản lý thư viện chung của toàn bộ dự án
└── README.md               # Hướng dẫn tổng quan (File này)
```

---

## 🚀 Hướng dẫn khởi chạy

### Yêu cầu
- Python 3.8+ (khuyên dùng Python 3.10 hoặc 3.11).
- Node.js 18+ (cho Frontend).

### 1. Khởi chạy Backend

1. Mở một terminal tại thư mục gốc dự án.
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
   cd backend
   python run.py
   ```
   API sẽ chạy tại địa chỉ [http://127.0.0.1:8000](http://127.0.0.1:8000).
   Bạn có thể xem tài liệu tự động và thử nghiệm trực tiếp tại: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### 2. Khởi chạy Frontend

> ⚠️ **Lưu ý:** Đảm bảo bạn vẫn đang giữ Terminal của Backend chạy ngầm. Hãy mở một Terminal **mới** để thực hiện các bước sau.

1. Di chuyển vào thư mục frontend:
   ```bash
   cd frontend
   ```
2. Cài đặt các gói thư viện Node.js:
   ```bash
   npm install
   ```
3. Khởi chạy giao diện:
   ```bash
   npm run dev
   ```

> 💡 Giao diện web sẽ hoạt động tại địa chỉ hiển thị trên Terminal (thường là http://localhost:5173). Dữ liệu nhập từ web sẽ gọi thẳng vào Backend API ở cổng 8000.

---

## 🤝 Hướng dẫn làm việc nhóm & Tích hợp mô hình

Khi một thành viên hoàn thành huấn luyện mô hình của mình trong Jupyter Notebook hoặc script training:
1. Lưu mô hình dưới dạng file (ví dụ: `random_forest.pkl`, `svm.pkl`...).
2. Bỏ file đó vào thư mục `models/` ở gốc dự án.
3. Mở file dịch vụ tương ứng tại `backend/app/services/models/{your_model}.py` và dán logic dự đoán/tiền xử lý cụ thể của bạn vào hàm `predict`.
4. Khởi động server, API sẽ tự động chuyển trạng thái của mô hình từ **Giả lập (Mock/Placeholder)** sang **Thật (Active)**.

*Đọc chi tiết hướng dẫn lập trình và tích hợp tại [backend/README.md](backend/README.md).*