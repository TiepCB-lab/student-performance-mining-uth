import pandas as pd
import numpy as np
from typing import Dict, Any, List

class DataPreprocessor:
    # 1. Ánh xạ nhị phân (0/1) cho các biến phân loại chỉ có 2 giá trị
    BINARY_MAPS = {
        "school_type": {"public": 0, "private": 1},
        "internet_access": {"no": 0, "yes": 1},
        "extra_activities": {"no": 0, "yes": 1}
    }

    # 2. Ánh xạ thứ tự (Ordinal Map) cho biến parent_education để giữ tính tăng tiến
    ORDINAL_MAPS = {
        "parent_education": {
            "no formal": 0,
            "high school": 1,
            "diploma": 2,
            "graduate": 3,
            "post graduate": 4,
            "phd": 5
        }
    }

    # 3. Các danh mục định danh (Nominal) cần được One-Hot Encoding (3 danh mục trở lên)
    # parent_education đã được chuyển sang Ordinal
    ONE_HOT_CATEGORIES = {
        "gender": ["male", "female", "other"],
        "travel_time": ["<15 min", "15-30 min", "30-60 min", ">60 min"],
        "study_method": ["notes", "textbook", "group study", "coaching", "mixed", "online videos"]
    }

    # 4. Các cột cơ sở dạng số (numerical features)
    BASE_COLUMNS = [
        "age", "study_hours", "attendance_percentage"
    ]
    #bổ sung 3 cột điểm thành phần
    SCORE_COLUMNS = [
        "math_score", "science_score", "english_score"
    ]

    # 5. Các cột nhị phân (Binary features) - cần được đưa vào feature vector
    BINARY_COLUMNS = ["school_type", "internet_access", "extra_activities"]

    # 6. Các cột tương tác (Interaction features) được tính từ features gốc
    INTERACTION_COLUMNS = [
        "study_hours_x_attendance",
        "study_hours_squared",
        "attendance_squared"
    ]

    @classmethod
    def get_feature_names(cls) -> List[str]:
        """
        Trả về danh sách đầy đủ tất cả các cột đặc trưng sau khi xử lý (tổng cộng 26 cột).
        Thứ tự các cột này sẽ luôn cố định để nạp vào mô hình ML.
        Bao gồm: 3 numerical + 3 score + 3 binary + 1 ordinal + 13 one-hot + 3 interaction = 26 cột.
        """
        columns = list(cls.BASE_COLUMNS)
        
        columns.extend(cls.SCORE_COLUMNS)
        # Thêm các cột nhị phân
        columns.extend(cls.BINARY_COLUMNS)
        # Thêm ordinal
        columns.append("parent_education")
        # Thêm one-hot
        for col, cats in cls.ONE_HOT_CATEGORIES.items():
            for cat in cats:
                columns.append(f"{col}_{cat}")
        # Thêm interaction features
        columns.extend(cls.INTERACTION_COLUMNS)
        return columns

    @classmethod
    def preprocess_to_dict(cls, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Xử lý dữ liệu thô từ API thành dữ liệu số (dict) đã mã hóa nhị phân, thứ tự & One-Hot.
        """
        processed = {}
        
        # 1. Xử lý các cột số (numerical features)
        for col in cls.BASE_COLUMNS:
            val = raw_data.get(col)
            if val is None:
                processed[col] = 0.0
            else:
                if col in ["study_hours", "attendance_percentage"]:
                    processed[col] = float(val)
                else:
                    processed[col] = int(val)
        
        # 1b. Xử lý 3 cột điểm thành phần 
        # Validate khoảng giá trị 0-100 để tránh input lỗi từ client làm sai lệch dự đoán.
        for col in cls.SCORE_COLUMNS:
            val = raw_data.get(col)
            if val is None:
                processed[col] = 0.0
            else:
                score = float(val)
                processed[col] = min(max(score, 0.0), 100.0)

        # 2. Xử lý các cột nhị phân (Binary Encoding)
        for col in cls.BINARY_COLUMNS:
            val = raw_data.get(col)
            processed[col] = cls.BINARY_MAPS[col].get(str(val).lower().strip(), 0)
                        
        # 3. Xử lý mã hóa thứ tự (Ordinal Encoding)
        for col, mapping in cls.ORDINAL_MAPS.items():
            val = raw_data.get(col)
            # Chuẩn hóa chuỗi trước khi tra cứu
            processed[col] = mapping.get(str(val).lower().strip(), 0)
                    
        # 4. Xử lý One-Hot Encoding cho các biến Nominal còn lại
        for col, categories in cls.ONE_HOT_CATEGORIES.items():
            current_val = raw_data.get(col)
            for cat in categories:
                # Tạo cột mới dạng {tên_cột}_{danh_mục} có giá trị là 1 hoặc 0
                processed[f"{col}_{cat}"] = 1 if str(current_val) == cat else 0

        # 5. Tạo Interaction Features (đặc trưng tương tác)
        study_hours = processed.get("study_hours", 0.0)
        attendance = processed.get("attendance_percentage", 0.0)
        processed["study_hours_x_attendance"] = study_hours * attendance
        processed["study_hours_squared"] = study_hours ** 2
        processed["attendance_squared"] = attendance ** 2
                
        return processed

    @classmethod
    def preprocess_to_numpy(cls, raw_data: Dict[str, Any]) -> np.ndarray:
        """
        Chuyển đổi dữ liệu học sinh thành mảng NumPy 2D (1, 23) sẵn sàng dự đoán.
        """
        processed_dict = cls.preprocess_to_dict(raw_data)
        ordered_cols = cls.get_feature_names()
        
        row = [processed_dict.get(col, 0) for col in ordered_cols]
        return np.array([row])

    @classmethod
    def preprocess_to_dataframe(cls, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Chuyển đổi dữ liệu học sinh thành Pandas DataFrame với cấu trúc cột chuẩn (26 cột).
        """
        processed_dict = cls.preprocess_to_dict(raw_data)
        df = pd.DataFrame([processed_dict])
        ordered_cols = cls.get_feature_names()
        
        # Đảm bảo các cột theo đúng thứ tự
        return df[ordered_cols]
