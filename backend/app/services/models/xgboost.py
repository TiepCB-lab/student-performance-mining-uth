import os
import numpy as np
from app.services.base_model import BaseModelService
from app.services.preprocessor import DataPreprocessor
from app.schemas.student import StudentFeatures

class XGBoostService(BaseModelService):
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.abspath(os.path.join(current_dir, "..", "..", "saved_models", "xgboost.pkl"))
        super().__init__(
            model_name="xgboost",
            display_name="XGBoost",
            model_path=model_path,
            description="Mô hình tăng cường độ dốc XGBoost tối ưu hóa độ chính xác dự đoán."
        )

    def predict(self, features: StudentFeatures) -> dict:
        raw_data = features.dict()
        
        # Nếu đã tải được mô hình thật
        if self.model is not None:
            try:
                X = DataPreprocessor.preprocess_to_numpy(raw_data)
                prediction_val = self.model.predict(X)[0]
                
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
                    "message": "Dự đoán bằng mô hình XGBoost thật."
                }
            except Exception as e:
                return {
                    "prediction": 0.0,
                    "model_name": self.model_name,
                    "display_name": self.display_name,
                    "success": False,
                    "message": f"Lỗi dự đoán mô hình thật: {str(e)}"
                }
        
        # Giả lập logic dự đoán của XGBoost
        g1 = raw_data.get("G1") if raw_data.get("G1") is not None else 10
        g2 = raw_data.get("G2") if raw_data.get("G2") is not None else 10
        studytime = raw_data.get("studytime", 2)
        failures = raw_data.get("failures", 0)
        higher = raw_data.get("higher", "yes")
        
        # XGBoost mock: Kỳ vọng cao hơn nếu học sinh có ý định học đại học (higher == yes)
        higher_bonus = 1.2 if higher == "yes" else -0.5
        mock_g3 = (0.35 * g1 + 0.65 * g2) + (studytime * 0.4) - (failures * 1.8) + higher_bonus
        mock_g3 = max(0.0, min(20.0, mock_g3))
        
        # Xác suất đỗ
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
