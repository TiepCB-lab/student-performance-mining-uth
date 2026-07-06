import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score
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

print(f"Feature count: {len(feature_cols)}")
print(f"Features: {feature_cols}")

# Nhãn mục tiêu (A-F)
y = df_raw['final_grade']

print(f"\nTarget distribution:")
print(y.value_counts().sort_index())

# 3. Phân chia Train/Test set (SỬ DỤNG STRATIFY để giữ phân bố lớp đồng đều)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain set: {X_train.shape[0]} rows, Test set: {X_test.shape[0]} rows")

# 4. Hyperparameter Tuning với RandomizedSearchCV
print("\n" + "="*60)
print("Hyperparameter Tuning with RandomizedSearchCV...")
print("="*60)

param_distributions = {
    'n_estimators': [100, 200, 300],
    'max_depth': [8, 12, 15, 20],
    'min_samples_split': [5, 10, 15, 20],
    'min_samples_leaf': [2, 4, 8, 12],
    'max_features': ['sqrt', 'log2'],
}

base_model = RandomForestClassifier(
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# StratifiedKFold đảm bảo mỗi fold giữ đúng tỷ lệ phân bố lớp
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_distributions,
    n_iter=30,  # Thử 30 tổ hợp ngẫu nhiên
    cv=cv,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

search.fit(X_train, y_train)

print(f"\nBest Parameters: {search.best_params_}")
print(f"Best CV Accuracy: {search.best_score_*100:.2f}%")

# 5. Đánh giá mô hình tốt nhất trên Test set
model = search.best_estimator_
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n{'='*60}")
print(f"[RANDOM FOREST] Test Accuracy: {acc*100:.2f}%")
print(f"{'='*60}")
print("\nClassification Report:")
target_labels = sorted(y.unique())
print(classification_report(y_test, y_pred, target_names=target_labels))

# 6. Cross-Validation trên toàn bộ dữ liệu để đánh giá ổn định
print("Cross-Validation on full dataset (5-fold)...")
cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
print(f"CV Scores: {[f'{s*100:.2f}%' for s in cv_scores]}")
print(f"Mean CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*200:.2f}%)")

# 7. Feature Importance
print("\nTop 10 most important features:")
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
for i in range(min(10, len(feature_cols))):
    print(f"  {i+1}. {feature_cols[indices[i]]} ({importances[indices[i]]:.4f})")

# 8. Lưu mô hình vào backend
save_dir = '../models'
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, 'random_forest.pkl')
joblib.dump(model, model_path)
print(f"\nMulti-class model saved successfully to: {model_path}!")
print(f"Model uses {len(feature_cols)} features (including binary + interaction features).")
