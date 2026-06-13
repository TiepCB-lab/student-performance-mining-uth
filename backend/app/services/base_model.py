import os
import joblib
from abc import ABC, abstractmethod
from app.schemas.student import StudentFeatures

class BaseModelService(ABC):
    def __init__(self, model_name: str, display_name: str, model_path: str = None, description: str = ""):
        self.model_name = model_name
        self.display_name = display_name
        self.model_path = model_path
        self.description = description
        self.model = None
        self.status = "placeholder"
        
        # Thử load model nếu có đường dẫn
        if self.model_path:
            self.load_model()

    def load_model(self):
        """
        Tải mô hình đã lưu từ ổ đĩa.
        Các thành viên có thể ghi đè hàm này nếu cần cách load đặc biệt (ví dụ: keras .h5, tensorflow, pytorch).
        """
        if not self.model_path:
            print(f"[{self.display_name}] No model path configured. Running in mock mode.")
            self.status = "placeholder"
            return

        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                self.status = "active"
                print(f"[{self.display_name}] Loaded model successfully from {self.model_path}")
            except Exception as e:
                print(f"[{self.display_name}] ERROR loading model from {self.model_path}: {str(e)}")
                self.status = "error"
        else:
            print(f"[{self.display_name}] Model file not found at {self.model_path}. Running in mock mode.")
            self.status = "placeholder"

    @abstractmethod
    def predict(self, features: StudentFeatures) -> dict:
        """
        Nhận đầu vào là StudentFeatures và trả về kết quả dự đoán.
        Cần được triển khai chi tiết ở lớp con.
        
        Trả về định dạng:
        {
            "prediction": float/int,
            "probability": list[float] (tùy chọn),
            "model_name": str,
            "display_name": str,
            "success": bool,
            "message": str (tùy chọn)
        }
        """
        pass
