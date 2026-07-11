import os
import numpy as np
import pandas as pd
from app.services.base_model import BaseModelService
from app.services.preprocessor import DataPreprocessor
from app.schemas.student import StudentFeatures

class LightGBMService(BaseModelService):
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.abspath(os.path.join(current_dir, "..", "..", "..", "..", "models", "lightgbm_model.pkl"))
        super().__init__(
            model_name="lightgbm",
            display_name="LightGBM",
            model_path=model_path,
            description="Light Gradient Boosting Machine (LightGBM) model for student performance grading."
        )

    def predict(self, features: StudentFeatures) -> dict:
        raw_data = features.dict()
        
        # Nếu đã tải được mô hình thật từ file
        if self.model is not None:
            try:
                # Tiền xử lý bằng DataPreprocessor chuẩn của dự án (26 đặc trưng)
                X = DataPreprocessor.preprocess_to_dataframe(raw_data)
                
                # Bổ sung 5 đặc trưng thống kê điểm số môn học giống như LightGBM_model.py
                subject_scores = [
                    raw_data.get("math_score") or 0.0, 
                    raw_data.get("science_score") or 0.0, 
                    raw_data.get("english_score") or 0.0
                ]
                
                X['subject_score_mean'] = np.mean(subject_scores)
                X['subject_score_min'] = np.min(subject_scores)
                X['subject_score_max'] = np.max(subject_scores)
                X['subject_score_range'] = X['subject_score_max'] - X['subject_score_min']
                X['subject_score_std'] = np.std(subject_scores, ddof=1) if len(subject_scores) > 1 else 0.0
                
                # Sắp xếp các cột đặc trưng chính xác theo trật tự 31 đặc trưng lúc train
                expected_cols = [
                    "age", "study_hours", "attendance_percentage",
                    "math_score", "science_score", "english_score",
                    "school_type", "internet_access", "extra_activities",
                    "parent_education",
                    "gender_male", "gender_female", "gender_other",
                    "travel_time_<15 min", "travel_time_15-30 min", "travel_time_30-60 min", "travel_time_>60 min",
                    "study_method_notes", "study_method_textbook", "study_method_group study",
                    "study_method_coaching", "study_method_mixed", "study_method_online videos",
                    "study_hours_x_attendance", "study_hours_squared", "attendance_squared",
                    "subject_score_mean", "subject_score_min", "subject_score_max", "subject_score_range", "subject_score_std"
                ]
                X = X[expected_cols]
                
                # Dự đoán bằng mô hình LightGBM thật
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
                    "message": "Prediction using actual LightGBM model."
                }
            except Exception as e:
                return {
                    "prediction": "F",
                    "model_name": self.model_name,
                    "display_name": self.display_name,
                    "success": False,
                    "message": f"Error predicting with actual LightGBM model: {str(e)}"
                }
        
        # Chế độ giả lập (Mock/Placeholder) nếu chưa có model file
        study_hours = raw_data.get("study_hours") if raw_data.get("study_hours") is not None else 4.0
        attendance = raw_data.get("attendance_percentage") if raw_data.get("attendance_percentage") is not None else 80.0
        
        mock_score = study_hours * 5.1 + attendance * 0.72
        mock_score = min(100.0, max(0.0, mock_score))
        
        if mock_score >= 85:
            mock_grade = "A"
            prob = [0.75, 0.17, 0.05, 0.01, 0.01, 0.01]
        elif mock_score >= 70:
            mock_grade = "B"
            prob = [0.14, 0.64, 0.14, 0.04, 0.02, 0.02]
        elif mock_score >= 55:
            mock_grade = "C"
            prob = [0.03, 0.15, 0.65, 0.13, 0.02, 0.02]
        elif mock_score >= 40:
            mock_grade = "D"
            prob = [0.01, 0.03, 0.15, 0.64, 0.13, 0.04]
        elif mock_score >= 25:
            mock_grade = "E"
            prob = [0.01, 0.01, 0.05, 0.13, 0.65, 0.15]
        else:
            mock_grade = "F"
            prob = [0.01, 0.01, 0.01, 0.04, 0.20, 0.73]
            
        return {
            "prediction": mock_grade,
            "probability": prob,
            "model_name": self.model_name,
            "display_name": self.display_name,
            "success": True,
            "message": "Prediction in mock mode (Awaiting integrated model .pkl file)."
        }
