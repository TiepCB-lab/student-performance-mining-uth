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

    @classmethod
    def get_feature_names(cls) -> List[str]:
        """
        Trả về danh sách đầy đủ tất cả các cột đặc trưng sau khi xử lý (tổng cộng 17 cột).
        Thứ tự các cột này sẽ luôn cố định để nạp vào mô hình ML.
        """
        columns = list(cls.BASE_COLUMNS)
        columns.append("parent_education")
        for col, cats in cls.ONE_HOT_CATEGORIES.items():
            for cat in cats:
                columns.append(f"{col}_{cat}")
        return columns

    @classmethod
    def preprocess_to_dict(cls, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Xử lý dữ liệu thô từ API thành dữ liệu số (dict) đã mã hóa nhị phân, thứ tự & One-Hot.
        """
        processed = {}
        
        # 1. Xử lý các cột số và nhị phân
        for col in cls.BASE_COLUMNS:
            val = raw_data.get(col)
            if col in cls.BINARY_MAPS:
                # Mã hóa nhị phân 0/1
                processed[col] = cls.BINARY_MAPS[col].get(str(val), 0)
            else:
                # Giá trị số nguyên hoặc float
                if val is None:
                    processed[col] = 0.0
                else:
                    if col in ["study_hours", "attendance_percentage"]:
                        processed[col] = float(val)
                    else:
                        processed[col] = int(val)
                        
        # 2. Xử lý mã hóa thứ tự (Ordinal Encoding)
        for col, mapping in cls.ORDINAL_MAPS.items():
            val = raw_data.get(col)
            # Chuẩn hóa chuỗi trước khi tra cứu
            processed[col] = mapping.get(str(val).lower().strip(), 0)
                    
        # 3. Xử lý One-Hot Encoding cho các biến Nominal còn lại
        for col, categories in cls.ONE_HOT_CATEGORIES.items():
            current_val = raw_data.get(col)
            for cat in categories:
                # Tạo cột mới dạng {tên_cột}_{danh_mục} có giá trị là 1 hoặc 0
                processed[f"{col}_{cat}"] = 1 if str(current_val) == cat else 0
                
        return processed

    @classmethod
    def preprocess_to_numpy(cls, raw_data: Dict[str, Any]) -> np.ndarray:
        """
        Chuyển đổi dữ liệu học sinh thành mảng NumPy 2D (1, 17) sẵn sàng dự đoán.
        """
        processed_dict = cls.preprocess_to_dict(raw_data)
        ordered_cols = cls.get_feature_names()
        
        row = [processed_dict.get(col, 0) for col in ordered_cols]
        return np.array([row])

    @classmethod
    def preprocess_to_dataframe(cls, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Chuyển đổi dữ liệu học sinh thành Pandas DataFrame với cấu trúc cột chuẩn (17 cột).
        """
        processed_dict = cls.preprocess_to_dict(raw_data)
        df = pd.DataFrame([processed_dict])
        ordered_cols = cls.get_feature_names()
        
        # Đảm bảo các cột theo đúng thứ tự
        return df[ordered_cols]
