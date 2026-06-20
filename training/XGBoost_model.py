# XGBoost model for Student_Performance.csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import xgboost as xgb
import joblib

# 1. Load dữ liệu
data = pd.read_csv('../data/raw/Student_Performance.csv')
feature_cols = ['age', 'gender', 'school_type', 'parent_education', 'study_hours',
                'attendance_percentage', 'internet_access', 'travel_time',
                'extra_activities', 'study_method']
X_features = data[feature_cols]
target = data['final_grade']

# 2. Xử lý nhãn (Target Encoding)
# XGBoost yêu cầu target phải là số (0, 1, 2, 3...)
label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)

# 3. Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(X_features, target_encoded, test_size=0.2, random_state=42)

# 4. Pipeline với XGBoost
cat_cols = X_features.select_dtypes(include=['object']).columns
num_cols = X_features.select_dtypes(exclude=['object']).columns

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# Lưu ý: use_label_encoder=False và objective='multi:softmax' cho bài toán nhiều lớp
clf = Pipeline([
    ('pre', preprocessor),
    ('model', xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5))
])

from sklearn.model_selection import GridSearchCV

# Grid search parameters
param_grid = {
    'model__max_depth': [3, 5, 7],
    'model__n_estimators': [100, 200],
    'model__learning_rate': [0.05, 0.1]
}
search_cv = GridSearchCV(clf, param_grid, cv=3, scoring='f1_macro', verbose=1)
# 5. Training
search_cv.fit(X_train, y_train)

# Mô hình được chọn (từ GridSearchCV)
selected_model = search_cv.best_estimator_

# In ra tham số được chọn
print("Tham số được chọn:", search_cv.best_params_)

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Dự đoán trên tập test bằng mô hình được chọn
y_pred = selected_model.predict(X_test)

# 2. In Classification Report
print("\n--- CHI TIẾT CÁC CHỈ SỐ MÔ HÌNH (CLASSIFICATION REPORT) ---")
# le.classes_ là danh sách các nhãn gốc (a, b, c, d, e, f) đã được encode
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 3. In Confusion Matrix (Dạng text)
print("\n--- MA TRẬN NHẦM LẪN (CONFUSION MATRIX) ---")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# (Tùy chọn) 4. Vẽ Confusion Matrix để dễ nhìn hơn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.title('Ma trận nhầm lẫn của mô hình XGBoost')
plt.show()

# Lưu mô hình được chọn
joblib.dump(selected_model, '../models/xgboost_model.pkl')