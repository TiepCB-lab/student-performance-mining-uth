from pydantic import BaseModel, Field
from typing import Optional, List

class StudentFeatures(BaseModel):
    school: str = Field(..., description="GP (Gabriel Pereira) hoặc MS (Mousinho da Silveira)")
    sex: str = Field(..., description="F (Nữ) hoặc M (Nam)")
    age: int = Field(..., description="Tuổi học sinh (15 - 22)")
    address: str = Field(..., description="U (Đô thị / Urban) hoặc R (Nông thôn / Rural)")
    famsize: str = Field(..., description="Quy mô gia đình: LE3 (<= 3 người) hoặc GT3 (> 3 người)")
    Pstatus: str = Field(..., description="Tình trạng sống chung của bố mẹ: T (Sống chung) hoặc A (Ly thân)")
    Medu: int = Field(..., description="Trình độ học vấn của mẹ (0: Không đi học, 1: Hết tiểu học, 2: Hết cấp 2, 3: Hết cấp 3, 4: Đại học/Cao đẳng)")
    Fedu: int = Field(..., description="Trình độ học vấn của bố (0: Không đi học, 1: Hết tiểu học, 2: Hết cấp 2, 3: Hết cấp 3, 4: Đại học/Cao đẳng)")
    Mjob: str = Field(..., description="Nghề nghiệp của mẹ (teacher, health, services, at_home, other)")
    Fjob: str = Field(..., description="Nghề nghiệp của bố (teacher, health, services, at_home, other)")
    reason: str = Field(..., description="Lý do chọn trường (home: gần nhà, reputation: danh tiếng, course: sở thích khóa học, other: lý do khác)")
    guardian: str = Field(..., description="Người giám hộ (mother, father, other)")
    traveltime: int = Field(..., description="Thời gian đi từ nhà đến trường (1: <15 phút, 2: 15-30 phút, 3: 30 phút - 1 tiếng, 4: >1 tiếng)")
    studytime: int = Field(..., description="Thời gian tự học hàng tuần (1: <2 tiếng, 2: 2-5 tiếng, 3: 5-10 tiếng, 4: >10 tiếng)")
    failures: int = Field(..., description="Số lần trượt môn trong quá khứ (0 - 3, nếu >=4 thì ghi 3)")
    schoolsup: str = Field(..., description="Có nhận hỗ trợ học tập từ nhà trường không? (yes hoặc no)")
    famsup: str = Field(..., description="Có nhận hỗ trợ học tập từ gia đình không? (yes hoặc no)")
    paid: str = Field(..., description="Có học thêm các lớp ngoài trường (có trả phí) không? (yes hoặc no)")
    activities: str = Field(..., description="Có tham gia hoạt động ngoại khóa không? (yes hoặc no)")
    nursery: str = Field(..., description="Có từng học trường mầm non không? (yes hoặc no)")
    higher: str = Field(..., description="Có muốn học lên đại học không? (yes hoặc no)")
    internet: str = Field(..., description="Có kết nối internet ở nhà không? (yes hoặc no)")
    romantic: str = Field(..., description="Có đang trong mối quan hệ yêu đương không? (yes hoặc no)")
    famrel: int = Field(..., description="Chất lượng mối quan hệ gia đình (1: Rất tệ, 5: Rất tốt)")
    freetime: int = Field(..., description="Thời gian rảnh rỗi sau giờ học (1: Rất ít, 5: Rất nhiều)")
    goout: int = Field(..., description="Tần suất đi chơi với bạn bè (1: Rất ít, 5: Rất nhiều)")
    Dalc: int = Field(..., description="Tần suất uống rượu trong tuần (1: Rất ít, 5: Rất nhiều)")
    Walc: int = Field(..., description="Tần suất uống rượu cuối tuần (1: Rất ít, 5: Rất nhiều)")
    health: int = Field(..., description="Tình trạng sức khỏe hiện tại (1: Rất tệ, 5: Rất tốt)")
    absences: int = Field(..., description="Số buổi nghỉ học (0 - 93)")
    G1: Optional[int] = Field(None, description="Điểm kỳ 1 (0 - 20, tùy chọn)")
    G2: Optional[int] = Field(None, description="Điểm kỳ 2 (0 - 20, tùy chọn)")

    class Config:
        json_schema_extra = {
            "example": {
                "school": "GP",
                "sex": "F",
                "age": 18,
                "address": "U",
                "famsize": "GT3",
                "Pstatus": "A",
                "Medu": 4,
                "Fedu": 4,
                "Mjob": "at_home",
                "Fjob": "teacher",
                "reason": "course",
                "guardian": "mother",
                "traveltime": 2,
                "studytime": 2,
                "failures": 0,
                "schoolsup": "yes",
                "famsup": "no",
                "paid": "no",
                "activities": "no",
                "nursery": "yes",
                "higher": "yes",
                "internet": "no",
                "romantic": "no",
                "famrel": 4,
                "freetime": 3,
                "goout": 4,
                "Dalc": 1,
                "Walc": 1,
                "health": 3,
                "absences": 6,
                "G1": 5,
                "G2": 6
            }
        }

class PredictionResult(BaseModel):
    prediction: float = Field(..., description="Kết quả dự đoán (điểm G3 hoặc phân lớp 0/1)")
    probability: Optional[List[float]] = Field(None, description="Xác suất cho từng lớp (nếu là bài toán phân lớp)")
    model_name: str = Field(..., description="Mã nhận dạng mô hình")
    display_name: str = Field(..., description="Tên mô hình hiển thị")
    success: bool = Field(True, description="Trạng thái thực thi thành công")
    message: Optional[str] = Field(None, description="Thông điệp thông báo hoặc lỗi")

class ModelInfo(BaseModel):
    model_name: str = Field(..., description="Mã nhận dạng mô hình")
    display_name: str = Field(..., description="Tên hiển thị của mô hình")
    status: str = Field(..., description="Trạng thái của mô hình (active: đã load, placeholder: mô hình giả lập)")
    description: str = Field(..., description="Mô tả ngắn gọn về thuật toán/mô hình")
