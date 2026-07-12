import os
import numpy as np
from app.services.base_model import BaseModelService
from app.services.preprocessor import DataPreprocessor
from app.schemas.student import StudentFeatures

class RandomForestService(BaseModelService):
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.abspath(os.path.join(current_dir, "..", "..", "..", "..", "models", "random_forest_model.pkl"))
        super().__init__(
            model_name="random_forest",
            display_name="Random Forest",
            model_path=model_path,
            description="Random Forest classification model for student performance grading (A-F)."
        )

    def predict(self, features: StudentFeatures) -> dict:
        raw_data = features.dict()
        
        # Nếu đã tải được mô hình thật từ file
        if self.model is not None:
            try:
                # Tiền xử lý dữ liệu thô sang Pandas DataFrame (1, 17)
                X = DataPreprocessor.preprocess_to_dataframe(raw_data)
                
                # Gọi hàm predict của mô hình thật (trả về nhãn chữ dạng 'a', 'b'...)
                prediction_val = self.model.predict(X)[0]
                grade_str = str(prediction_val).upper()
                
                # Lấy xác suất phân lớp
                probability_list = None
                if hasattr(self.model, "predict_proba"):
                    try:
                        probability_list = self.model.predict_proba(X)[0].tolist()
                        probability_list = [round(p, 4) for p in probability_list]
                    except:
                        pass
                
                return {
                    "prediction": grade_str,
                    "probability": probability_list,
                    "model_name": self.model_name,
                    "display_name": self.display_name,
                    "success": True,
                    "message": "Prediction using actual Random Forest model."
                }
            except Exception as e:
                return {
                    "prediction": "F",
                    "model_name": self.model_name,
                    "display_name": self.display_name,
                    "success": False,
                    "message": f"Error predicting with actual model: {str(e)}"
                }
        
        # Chế độ giả lập (Mock/Placeholder) nếu chưa có model file
        study_hours = raw_data.get("study_hours") if raw_data.get("study_hours") is not None else 4.0
        attendance = raw_data.get("attendance_percentage") if raw_data.get("attendance_percentage") is not None else 80.0
        
        # Tính toán điểm giả lập đơn giản
        mock_score = study_hours * 5.0 + attendance * 0.7
        mock_score = min(100.0, max(0.0, mock_score))
        
        # Áp dụng ngưỡng phân loại xếp hạng A-F
        if mock_score >= 85:
            mock_grade = "A"
            prob = [0.70, 0.20, 0.05, 0.03, 0.01, 0.01]
        elif mock_score >= 70:
            mock_grade = "B"
            prob = [0.15, 0.60, 0.15, 0.05, 0.03, 0.02]
        elif mock_score >= 55:
            mock_grade = "C"
            prob = [0.05, 0.15, 0.60, 0.15, 0.03, 0.02]
        elif mock_score >= 40:
            mock_grade = "D"
            prob = [0.02, 0.03, 0.15, 0.60, 0.15, 0.05]
        elif mock_score >= 25:
            mock_grade = "E"
            prob = [0.01, 0.02, 0.05, 0.15, 0.60, 0.17]
        else:
            mock_grade = "F"
            prob = [0.01, 0.01, 0.02, 0.06, 0.20, 0.70]
            
        return {
            "prediction": mock_grade,
            "probability": prob,
            "model_name": self.model_name,
            "display_name": self.display_name,
            "success": True,
            "message": "Prediction in mock mode (Awaiting integrated model .pkl file)."
        }
