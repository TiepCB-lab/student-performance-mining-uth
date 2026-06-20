from pydantic import BaseModel, Field
from typing import Optional, List

class StudentFeatures(BaseModel):
    gender: str = Field(..., description="Giới tính học sinh (male, female, other)")
    school_type: str = Field(..., description="Loại trường học (public, private)")
    age: int = Field(..., description="Tuổi học sinh (14 - 22)")
    parent_education: str = Field(..., description="Trình độ học vấn của cha mẹ (high school, diploma, graduate, post graduate, phd, no formal)")
    study_hours: float = Field(..., description="Số giờ tự học mỗi ngày (giờ)")
    attendance_percentage: float = Field(..., description="Tỷ lệ chuyên cần (%)")
    internet_access: str = Field(..., description="Có kết nối internet ở nhà không? (yes hoặc no)")
    travel_time: str = Field(..., description="Thời gian đi từ nhà đến trường (<15 min, 15-30 min, 30-60 min, >60 min)")
    extra_activities: str = Field(..., description="Có tham gia hoạt động ngoại khóa không? (yes hoặc no)")
    study_method: str = Field(..., description="Phương pháp tự học (notes, textbook, group study, coaching, mixed, online videos)")
    
    math_score: Optional[float] = Field(None, description="Điểm Toán (0 - 100, tùy chọn)")
    science_score: Optional[float] = Field(None, description="Điểm Khoa học (0 - 100, tùy chọn)")
    english_score: Optional[float] = Field(None, description="Điểm Tiếng Anh (0 - 100, tùy chọn)")
    overall_score: Optional[float] = Field(None, description="Điểm tổng kết (0 - 100, tùy chọn)")

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "female",
                "school_type": "public",
                "age": 16,
                "parent_education": "graduate",
                "study_hours": 4.5,
                "attendance_percentage": 92.5,
                "internet_access": "yes",
                "travel_time": "15-30 min",
                "extra_activities": "yes",
                "study_method": "notes",
                "math_score": 75.0,
                "science_score": 82.0,
                "english_score": 78.0,
                "overall_score": 78.3
            }
        }

class PredictionResult(BaseModel):
    prediction: str = Field(..., description="Kết quả dự đoán xếp hạng học lực (A, B, C, D, E, F)")
    probability: Optional[List[float]] = Field(None, description="Danh sách xác suất cho từng lớp học lực [A, B, C, D, E, F]")
    model_name: str = Field(..., description="Mã nhận dạng mô hình")
    display_name: str = Field(..., description="Tên mô hình hiển thị")
    success: bool = Field(True, description="Trạng thái thực thi thành công")
    message: Optional[str] = Field(None, description="Thông điệp thông báo hoặc lỗi")

class ModelInfo(BaseModel):
    model_name: str = Field(..., description="Mã nhận dạng mô hình")
    display_name: str = Field(..., description="Tên hiển thị của mô hình")
    status: str = Field(..., description="Trạng thái của mô hình (active: đã load, placeholder: mô hình giả lập)")
    description: str = Field(..., description="Mô tả ngắn gọn về thuật toán/mô hình")
