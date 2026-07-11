import os
import pandas as pd
from app.services.base_model import BaseModelService
from app.schemas.student import StudentFeatures

class XGBoostService(BaseModelService):
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.abspath(os.path.join(current_dir, "..", "..", "..", "..", "models", "xgboost_model.pkl"))
        super().__init__(
            model_name="xgboost",
            display_name="XGBoost",
            model_path=model_path,
            description="Extreme Gradient Boosting (XGBoost) model pipeline for student performance grading."
        )

    def predict(self, features: StudentFeatures) -> dict:
        raw_data = features.dict()
        
        # Nếu đã tải được mô hình thật từ file
        if self.model is not None:
            try:
                # Tạo raw DataFrame từ dict thô
                raw_df = pd.DataFrame([raw_data])
                
                # Thứ tự các cột đặc trưng mà pipeline huấn luyện của leader mong đợi
                expected_cols = [
                    'age', 'gender', 'school_type', 'parent_education', 'study_hours',
                    'attendance_percentage', 'internet_access', 'travel_time',
                    'extra_activities', 'study_method', 'math_score', 'science_score', 'english_score'
                ]
                X = raw_df[expected_cols]
                
                # Dự đoán qua pipeline (sẽ tự động chạy ColumnTransformer)
                prediction_idx = self.model.predict(X)[0]
                
                # Giải mã nhãn: LabelEncoder xếp 'a'-'f' tương ứng 0-5
                classes = ['A', 'B', 'C', 'D', 'E', 'F']
                grade_str = classes[int(prediction_idx)]
                
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
                    "message": "Prediction using actual XGBoost model pipeline."
                }
            except Exception as e:
                return {
                    "prediction": "F",
                    "model_name": self.model_name,
                    "display_name": self.display_name,
                    "success": False,
                    "message": f"Error predicting with actual XGBoost model: {str(e)}"
                }
        
        # Chế độ giả lập (Mock/Placeholder) nếu chưa có model file
        study_hours = raw_data.get("study_hours") if raw_data.get("study_hours") is not None else 4.0
        attendance = raw_data.get("attendance_percentage") if raw_data.get("attendance_percentage") is not None else 80.0
        
        mock_score = study_hours * 5.2 + attendance * 0.68
        mock_score = min(100.0, max(0.0, mock_score))
        
        if mock_score >= 85:
            mock_grade = "A"
            prob = [0.72, 0.18, 0.06, 0.02, 0.01, 0.01]
        elif mock_score >= 70:
            mock_grade = "B"
            prob = [0.12, 0.62, 0.16, 0.06, 0.02, 0.02]
        elif mock_score >= 55:
            mock_grade = "C"
            prob = [0.04, 0.16, 0.62, 0.14, 0.02, 0.02]
        elif mock_score >= 40:
            mock_grade = "D"
            prob = [0.01, 0.04, 0.16, 0.62, 0.13, 0.04]
        elif mock_score >= 25:
            mock_grade = "E"
            prob = [0.01, 0.01, 0.06, 0.14, 0.62, 0.16]
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
