from fastapi import APIRouter
from typing import List
from app.schemas.student import ModelInfo
from app.services import list_model_services

router = APIRouter()

@router.get("", response_model=List[ModelInfo], summary="Lấy danh sách các mô hình")
def get_models():
    """
    Trả về danh sách tất cả các mô hình đã đăng ký trong hệ thống
    kèm theo trạng thái hoạt động (chế độ giả lập 'placeholder' hoặc mô hình thực tế 'active').
    """
    result = []
    for service in list_model_services():
        result.append(
            ModelInfo(
                model_name=service.model_name,
                display_name=service.display_name,
                status=service.status,
                description=service.description
            )
        )
    return result
