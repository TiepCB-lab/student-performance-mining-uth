import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.routers import models, predict

# Đảm bảo thư mục lưu trữ model tồn tại
os.makedirs(os.path.join(os.path.dirname(__file__), "saved_models"), exist_ok=True)

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    description="Backend API phục vụ dự báo học lực sinh viên dựa trên kỹ thuật khai phá dữ liệu."
)

# Cấu hình CORS Middleware để Frontend gọi API thuận tiện
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đăng ký các API Routers
app.include_router(models.router, prefix=f"{settings.API_PREFIX}/models", tags=["Models Management"])
app.include_router(predict.router, prefix=f"{settings.API_PREFIX}/predict", tags=["Predictions Engine"])

@app.get("/", tags=["Root"])
def root():
    """
    Endpoint chào mừng và kiểm tra trạng thái hoạt động của Server.
    """
    return {
        "message": f"Chào mừng bạn đến với API {settings.PROJECT_NAME}",
        "version": settings.PROJECT_VERSION,
        "swagger_docs": "/docs"
    }
