import os
import numpy as np
from app.services.base_model import BaseModelService
from app.services.preprocessor import DataPreprocessor
from app.schemas.student import StudentFeatures

class NeuralNetworkService(BaseModelService):
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.abspath(os.path.join(current_dir, "..", "..", "saved_models", "neural_network.pkl"))
        super().__init__(
            model_name="neural_network",
            display_name="Multi-Layer Perceptron (Neural Network)",
            model_path=model_path,
            description="Mô hình mạng nơ-ron nhân tạo đa tầng học sâu các mối quan hệ phi tuyến."
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
                    "message": "Dự đoán bằng mô hình Neural Network thật."
                }
            except Exception as e:
                return {
                    "prediction": 0.0,
                    "model_name": self.model_name,
                    "display_name": self.display_name,
                    "success": False,
                    "message": f"Lỗi dự đoán mô hình thật: {str(e)}"
                }
        
        # Giả lập logic dự đoán của Neural Network
        g1 = raw_data.get("G1") if raw_data.get("G1") is not None else 10
        g2 = raw_data.get("G2") if raw_data.get("G2") is not None else 10
        studytime = raw_data.get("studytime", 2)
        failures = raw_data.get("failures", 0)
        schoolsup = raw_data.get("schoolsup", "no")
        famsup = raw_data.get("famsup", "no")
        
        # Neural Network mock: Kết hợp phi tuyến phức tạp hơn một chút
        support_factor = 0.5 if schoolsup == "yes" else 0.0
        family_support = 0.3 if famsup == "yes" else 0.0
        
        # Phi tuyến: dùng bình phương nhẹ hoặc tích chéo
        base_grade = (g1 * g2) ** 0.5 if g1 > 0 and g2 > 0 else (g1 + g2) / 2.0
        mock_g3 = base_grade + (studytime * 0.4) - (failures * 2.0) + support_factor + family_support
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
