import os
import numpy as np
from app.services.base_model import BaseModelService
from app.services.preprocessor import DataPreprocessor
from app.schemas.student import StudentFeatures

class RandomForestService(BaseModelService):
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.abspath(os.path.join(current_dir, "..", "..", "saved_models", "random_forest.pkl"))
        super().__init__(
            model_name="random_forest",
            display_name="Random Forest",
            model_path=model_path,
            description="Mô hình Random Forest phân lớp / hồi quy kết quả học tập."
        )

    def predict(self, features: StudentFeatures) -> dict:
        raw_data = features.dict()
        
        # Nếu đã tải được mô hình thật
        if self.model is not None:
            try:
                # Tiền xử lý thành định dạng mảng NumPy (1, N)
                X = DataPreprocessor.preprocess_to_numpy(raw_data)
                
                # Gọi hàm predict của Scikit-Learn
                prediction_val = self.model.predict(X)[0]
                
                # Thử lấy xác suất nếu mô hình hỗ trợ phân lớp
                probability_list = None
                if hasattr(self.model, "predict_proba"):
                    try:
                        probability_list = self.model.predict_proba(X)[0].tolist()
                    except:
                        pass
                
                return {
                    "prediction": float(prediction_val),
                    "probability": probability_list,
                    "model_name": self.model_name,
                    "display_name": self.display_name,
                    "success": True,
                    "message": "Dự đoán bằng mô hình Random Forest thật."
                }
            except Exception as e:
                return {
                    "prediction": 0.0,
                    "model_name": self.model_name,
                    "display_name": self.display_name,
                    "success": False,
                    "message": f"Lỗi dự đoán mô hình thật: {str(e)}"
                }
        
        # Chế độ giả lập (Mock/Placeholder) nếu chưa có model file
        # Giả lập logic dự đoán G3 dựa trên các đặc trưng chính
        g1 = raw_data.get("G1") if raw_data.get("G1") is not None else 10
        g2 = raw_data.get("G2") if raw_data.get("G2") is not None else 10
        studytime = raw_data.get("studytime", 2)
        failures = raw_data.get("failures", 0)
        absences = raw_data.get("absences", 0)
        
        # Tính toán điểm giả lập G3 hợp lý
        mock_g3 = (g1 + g2) / 2.0 + (studytime * 0.5) - (failures * 1.5) - (absences * 0.05)
        mock_g3 = max(0.0, min(20.0, mock_g3)) # Giới hạn trong khoảng [0, 20]
        
        # Giả lập xác suất đỗ (G3 >= 10)
        pass_prob = 1.0 / (1.0 + np.exp(-(mock_g3 - 10.0)))
        fail_prob = 1.0 - pass_prob

        return {
            "prediction": round(mock_g3, 2),
            "probability": [round(fail_prob, 4), round(pass_prob, 4)],
            "model_name": self.model_name,
            "display_name": self.display_name,
            "success": True,
            "message": "Dự đoán ở chế độ giả lập (Chờ tích hợp file model .pkl)."
        }
