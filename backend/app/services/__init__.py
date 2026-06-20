from typing import Dict, List, Optional
from app.services.base_model import BaseModelService
from app.services.models.random_forest import RandomForestService
from app.services.models.svm import SVMService
from app.services.models.neural_network import NeuralNetworkService

# Khởi tạo instance của tất cả các mô hình dịch vụ
# Khi server khởi động, các lớp này sẽ tự động load file model thực tế nếu có
services: Dict[str, BaseModelService] = {
    "random_forest": RandomForestService(),
    "svm": SVMService(),
    "neural_network": NeuralNetworkService()
}

def get_model_service(model_name: str) -> Optional[BaseModelService]:
    """
    Lấy service tương ứng với tên model.
    """
    return services.get(model_name)

def list_model_services() -> List[BaseModelService]:
    """
    Lấy danh sách tất cả các services mô hình đã đăng ký.
    """
    return list(services.values())
