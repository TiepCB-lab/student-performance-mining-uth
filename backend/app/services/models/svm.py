import os
import numpy as np
from app.services.base_model import BaseModelService
from app.services.preprocessor import DataPreprocessor
from app.schemas.student import StudentFeatures

class SVMService(BaseModelService):
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.abspath(os.path.join(current_dir, "..", "..", "..", "..", "models", "svm.pkl"))
        super().__init__(
            model_name="svm",
            display_name="Support Vector Machine (SVM)",
            model_path=model_path,
            description="SVM classification model for student performance grading (A-F)."
        )

    def predict(self, features: StudentFeatures) -> dict:
        raw_data = features.dict()
        
        # Nếu đã tải được mô hình thật
        if self.model is not None:
            try:
                # Tiền xử lý dữ liệu thô sang Pandas DataFrame (1, 17)
                X = DataPreprocessor.preprocess_to_dataframe(raw_data)
                prediction_val = self.model.predict(X)[0]
                grade_str = str(prediction_val).upper()
                
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
                    "message": "Prediction using actual SVM model."
                }
            except Exception as e:
                return {
                    "prediction": "F",
                    "model_name": self.model_name,
                    "display_name": self.display_name,
                    "success": False,
                    "message": f"Error predicting with actual model: {str(e)}"
                }
        
        # Giả lập logic dự đoán của SVM (khác một chút so với RF)
        study_hours = raw_data.get("study_hours") if raw_data.get("study_hours") is not None else 4.0
        attendance = raw_data.get("attendance_percentage") if raw_data.get("attendance_percentage") is not None else 80.0
        extra_activities = raw_data.get("extra_activities", "no")
        
        # Điểm giả lập SVM nhạy hơn với hoạt động ngoại khóa
        activity_bonus = 3.0 if extra_activities == "yes" else 0.0
        mock_score = study_hours * 4.5 + attendance * 0.72 + activity_bonus
        mock_score = min(100.0, max(0.0, mock_score))
        
        if mock_score >= 85:
            mock_grade = "A"
            prob = [0.65, 0.22, 0.07, 0.03, 0.02, 0.01]
        elif mock_score >= 70:
            mock_grade = "B"
            prob = [0.18, 0.58, 0.16, 0.05, 0.02, 0.01]
        elif mock_score >= 55:
            mock_grade = "C"
            prob = [0.06, 0.14, 0.62, 0.13, 0.03, 0.02]
        elif mock_score >= 40:
            mock_grade = "D"
            prob = [0.03, 0.04, 0.13, 0.62, 0.14, 0.04]
        elif mock_score >= 25:
            mock_grade = "E"
            prob = [0.01, 0.02, 0.04, 0.14, 0.62, 0.17]
        else:
            mock_grade = "F"
            prob = [0.01, 0.01, 0.01, 0.05, 0.22, 0.70]
            
        return {
            "prediction": mock_grade,
            "probability": prob,
            "model_name": self.model_name,
            "display_name": self.display_name,
            "success": True,
            "message": "Prediction in mock mode (Awaiting integrated model .pkl file)."
        }
