# Hướng dẫn Phát triển Backend & Tích hợp Mô hình ML

Thư mục này chứa toàn bộ mã nguồn Backend của dự án dự đoán học lực sinh viên, được phát triển bằng **FastAPI**. Backend đã được thiết kế theo dạng module độc lập giúp từng thành viên trong nhóm tích hợp mô hình của mình vào hệ thống chung một cách dễ dàng và nhanh chóng.

---

## 1. Cấu trúc thư mục Backend cần lưu ý

- `app/core/`: Chứa cấu hình môi trường lõi (config.py).
- `app/routes/`: Chứa các API endpoints (/predict, /models).
- `app/schemas/student.py`: Nơi định nghĩa các Schema dữ liệu. Lớp `StudentFeatures` quy định đúng các thuộc tính đầu vào của sinh viên giống hệt tập dữ liệu `Student_Performance.csv`.
- `app/services/models/`: Chứa file xử lý dự đoán của từng mô hình (ví dụ: Random Forest, SVM, Neural Network).
- `app/services/preprocessor.py`: Cung cấp sẵn class `DataPreprocessor` để tự động hóa việc mã hóa dữ liệu chữ sang dữ liệu số trước khi đưa vào mô hình.

---

## 2. Hướng dẫn cài đặt và chạy thử

### Yêu cầu hệ thống
- Python 3.8+ (khuyên dùng Python 3.10 hoặc 3.11).

### Các bước cài đặt:
1. Mở terminal tại thư mục `backend/`
2. Tạo môi trường ảo (khuyên dùng):
   ```bash
   python -m venv venv
   # Kích hoạt trên Windows:
   .\venv\Scripts\activate
   # Hoặc kích hoạt trên macOS/Linux:
   source venv/bin/activate
   ```
3. Cài đặt các thư viện phụ thuộc:
   ```bash
   pip install -r requirements.txt
   ```
4. Khởi động server:
   ```bash
   python run.py
   ```
   Server sẽ chạy tại địa chỉ: [http://127.0.0.1:8000](http://127.0.0.1:8000)
   Bạn có thể xem tài liệu tự động và thử nghiệm API tại: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 3. Hướng dẫn tích hợp mô hình ML của bạn (Dành cho các thành viên)

Khi bạn đã hoàn thành việc huấn luyện mô hình của mình trong Jupyter Notebook và lưu mô hình, hãy làm theo 3 bước sau để triển khai:

### Bước 1: Lưu file model đã train
Bỏ file model của bạn vào thư mục `models/` ở gốc dự án.
- Ví dụ nếu bạn train Random Forest: đặt tên file là `random_forest.pkl`.

### Bước 2: Sửa đổi file service tương ứng trong `app/services/models/`
Tìm file tương ứng với mô hình của bạn (ví dụ: `app/services/models/random_forest.py` hoặc tạo file mới kế thừa từ `BaseModelService`).

Mã nguồn mẫu đã được viết sẵn khung. Bạn chỉ cần sửa đổi logic trong hàm `predict(self, features: StudentFeatures)`:
```python
def predict(self, features: StudentFeatures) -> dict:
    raw_data = features.dict()
    
    # 1. Nếu mô hình thật đã được tải thành công từ file
    if self.model is not None:
        # 2. Tiền xử lý dữ liệu thô sang Pandas DataFrame
        # DataPreprocessor sẽ tự động mã hóa nhị phân, Ordinal, và One-Hot...
        X = DataPreprocessor.preprocess_to_dataframe(raw_data) # Trả về DataFrame (1, 17)
        
        # 3. Thực hiện dự đoán (nhãn chữ xếp hạng A-F)
        prediction_val = self.model.predict(X)[0]
        grade_str = str(prediction_val).upper()
        
        # 4. Trả về đúng định dạng yêu cầu
        return {
            "prediction": grade_str,
            "probability": self.model.predict_proba(X)[0].tolist() if hasattr(self.model, "predict_proba") else None,
            "model_name": self.model_name,
            "display_name": self.display_name,
            "success": True,
            "message": "Prediction using actual model."
        }
```

*Lưu ý:* Nếu bạn cần quy trình tiền xử lý riêng biệt (ví dụ: chuẩn hóa tỉ lệ `StandardScaler`, điền khuyết đặc biệt), bạn có thể import và áp dụng trực tiếp trong hàm `predict` này.

### Bước 3: Kiểm tra trên Giao diện API Swagger
Sau khi lưu code, uvicorn sẽ tự động reload lại server. Truy cập [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs):
1. Gọi API `GET /api/models` để kiểm tra mô hình của bạn có chuyển trạng thái từ `placeholder` sang `active` hay chưa.
2. Gọi API `POST /api/predict/{model_name}` để test thử đầu ra với dữ liệu sinh viên mẫu.
