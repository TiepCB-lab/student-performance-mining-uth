import pandas as pd
import numpy as np
from typing import Dict, Any, List

class DataPreprocessor:
    # 1. Ánh xạ nhị phân (0/1) cho các biến phân loại chỉ có 2 giá trị
    BINARY_MAPS = {
        "school": {"GP": 0, "MS": 1},
        "sex": {"F": 0, "M": 1},
        "address": {"U": 0, "R": 1},
        "famsize": {"LE3": 0, "GT3": 1},
        "Pstatus": {"T": 0, "A": 1},
        "schoolsup": {"no": 0, "yes": 1},
        "famsup": {"no": 0, "yes": 1},
        "paid": {"no": 0, "yes": 1},
        "activities": {"no": 0, "yes": 1},
        "nursery": {"no": 0, "yes": 1},
        "higher": {"no": 0, "yes": 1},
        "internet": {"no": 0, "yes": 1},
        "romantic": {"no": 0, "yes": 1}
    }

    # 2. Các danh mục định danh (Nominal) cần được One-Hot Encoding (3 danh mục trở lên)
    # Để tránh việc thuật toán gán thứ tự sai lệch (ví dụ: teacher < health < services)
    ONE_HOT_CATEGORIES = {
        "Mjob": ["teacher", "health", "services", "at_home", "other"],
        "Fjob": ["teacher", "health", "services", "at_home", "other"],
        "reason": ["home", "reputation", "course", "other"],
        "guardian": ["mother", "father", "other"]
    }

    # 3. Các cột cơ sở (Không bao gồm các cột Nominal đã được mã hóa One-Hot)
    BASE_COLUMNS = [
        "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
        "traveltime", "studytime", "failures", "schoolsup", "famsup", "paid",
        "activities", "nursery", "higher", "internet", "romantic", "famrel",
        "freetime", "goout", "Dalc", "Walc", "health", "absences", "G1", "G2"
    ]

    @classmethod
    def get_feature_names(cls) -> List[str]:
        """
        Trả về danh sách đầy đủ tất cả các cột đặc trưng sau khi đã thực hiện One-Hot Encoding (tổng 45 cột).
        Thứ tự các cột này sẽ luôn cố định để nạp vào mô hình ML.
        """
        columns = list(cls.BASE_COLUMNS)
        for col, cats in cls.ONE_HOT_CATEGORIES.items():
            for cat in cats:
                columns.append(f"{col}_{cat}")
        return columns

    @classmethod
    def preprocess_to_dict(cls, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Xử lý dữ liệu thô từ API thành dữ liệu số (dict) đã mã hóa nhị phân & One-Hot.
        """
        processed = {}
        
        # 1. Xử lý các cột cơ sở (số nguyên hoặc nhị phân)
        for col in cls.BASE_COLUMNS:
            val = raw_data.get(col)
            if col in cls.BINARY_MAPS:
                # Mã hóa nhị phân 0/1
                processed[col] = cls.BINARY_MAPS[col].get(str(val), 0)
            else:
                # Giá trị số nguyên
                if val is None:
                    processed[col] = 0  # Giá trị mặc định cho G1/G2 nếu bị thiếu
                else:
                    processed[col] = int(val)
                    
        # 2. Xử lý One-Hot Encoding cho các biến Nominal
        for col, categories in cls.ONE_HOT_CATEGORIES.items():
            current_val = raw_data.get(col, "other")
            for cat in categories:
                # Tạo cột mới dạng {tên_cột}_{danh_mục} có giá trị là 1 hoặc 0
                processed[f"{col}_{cat}"] = 1 if str(current_val) == cat else 0
                
        return processed

    @classmethod
    def preprocess_to_numpy(cls, raw_data: Dict[str, Any]) -> np.ndarray:
        """
        Chuyển đổi dữ liệu học sinh thành mảng NumPy 2D (1, 45) sẵn sàng dự đoán.
        """
        processed_dict = cls.preprocess_to_dict(raw_data)
        ordered_cols = cls.get_feature_names()
        
        row = [processed_dict.get(col, 0) for col in ordered_cols]
        return np.array([row])

    @classmethod
    def preprocess_to_dataframe(cls, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Chuyển đổi dữ liệu học sinh thành Pandas DataFrame với cấu trúc cột chuẩn (45 cột).
        """
        processed_dict = cls.preprocess_to_dict(raw_data)
        df = pd.DataFrame([processed_dict])
        ordered_cols = cls.get_feature_names()
        
        # Đảm bảo các cột theo đúng thứ tự
        return df[ordered_cols]
