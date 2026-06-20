from fastapi import APIRouter, HTTPException, Path
from app.schemas.student import StudentFeatures, PredictionResult
from app.services import get_model_service, list_model_services
from typing import Dict

router = APIRouter()

@router.post("/all", response_model=Dict[str, PredictionResult], summary="Dự đoán bằng tất cả mô hình")
def predict_all(features: StudentFeatures):
    """
    Chạy dự đoán song song/đồng thời trên tất cả các mô hình đang có trong hệ thống
    và trả về danh sách so sánh kết quả.
    """
    results = {}
    for service in list_model_services():
        try:
            res = service.predict(features)
            results[service.model_name] = PredictionResult(**res)
        except Exception as e:
            results[service.model_name] = PredictionResult(
                prediction="F",
                model_name=service.model_name,
                display_name=service.display_name,
                success=False,
                message=f"System error: {str(e)}"
            )
    return results

@router.post("/{model_name}", response_model=PredictionResult, summary="Dự đoán theo mô hình cụ thể")
def predict_model(
    features: StudentFeatures,
    model_name: str = Path(..., description="Tên định danh của mô hình (ví dụ: random_forest, svm, neural_network)")
):
    """
    Thực hiện dự đoán điểm G3 hoặc trạng thái học tập của sinh viên dựa trên mô hình được chọn.
    """
    service = get_model_service(model_name)
    if not service:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' does not exist or has not been registered in the system."
        )
    
    try:
        res = service.predict(features)
        return PredictionResult(**res)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during the processing of model {model_name}: {str(e)}"
        )
