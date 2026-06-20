import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Thiết lập mã hóa UTF-8 cho console Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Thêm thư mục backend vào hệ thống PATH
sys.path.append(os.path.abspath('../backend'))
from app.services.preprocessor import DataPreprocessor

# 1. Đọc dữ liệu
data_path = '../data/raw/Student_Performance.csv'
if not os.path.exists(data_path):
    print(f"Data file not found at {data_path}")
    sys.exit(1)

df_raw = pd.read_csv(data_path)
df_raw.columns = df_raw.columns.str.strip()
print(f"Loaded dataset: {df_raw.shape[0]} rows.")

# 2. Tiền xử lý dữ liệu bằng DataPreprocessor của backend
print("Preprocessing features using backend DataPreprocessor...")
processed_records = []
for _, row in df_raw.iterrows():
    raw_dict = row.to_dict()
    processed_dict = DataPreprocessor.preprocess_to_dict(raw_dict)
    processed_records.append(processed_dict)

df_processed = pd.DataFrame(processed_records)
feature_cols = DataPreprocessor.get_feature_names()
X = df_processed[feature_cols]

# Nhãn mục tiêu (A-F)
y = df_raw['final_grade']

# 3. Phân chia Train/Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Huấn luyện mô hình Random Forest Classifier (Đa lớp)
print("Training Multi-class Random Forest Classifier on 6 academic performance levels...")
model = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 5. Đánh giá
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n[RANDOM FOREST] Accuracy: {acc*100:.2f}%")
print("\nClassification Report:")
target_labels = sorted(y.unique())
print(classification_report(y_test, y_pred, target_names=target_labels))

# 6. Lưu mô hình vào backend
save_dir = '../models'
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, 'random_forest.pkl')
joblib.dump(model, model_path)
print(f"Multi-class model saved successfully to: {model_path}!")
