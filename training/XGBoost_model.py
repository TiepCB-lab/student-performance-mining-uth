import os
import sys
import pandas as pd
import re
import numpy as np
import joblib
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    learning_curve,
    StratifiedKFold,
    cross_val_score
)

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)

# Thiết lập mã hóa UTF-8 cho console Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Thêm thư mục backend vào hệ thống PATH
sys.path.append(os.path.abspath('../backend'))
from app.services.preprocessor import DataPreprocessor

# 1. Load dữ liệu
data_path = '../data/raw/Student_Performance.csv'
if not os.path.exists(data_path):
    print(f"❌ Data file not found at {data_path}")
    sys.exit(1)

df_raw = pd.read_csv(data_path)
df_raw.columns = df_raw.columns.str.strip()
print(f"✅ Loaded dataset: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

# 2. Tiền xử lý dữ liệu bằng DataPreprocessor từ backend
print("🔄 Preprocessing features using backend DataPreprocessor...")
processed_records = []
for _, row in df_raw.iterrows():
    processed_records.append(DataPreprocessor.preprocess_to_dict(row.to_dict()))

df_processed = pd.DataFrame(processed_records)
feature_cols_original = DataPreprocessor.get_feature_names()

# Sanitize feature names to be XGBoost-safe (no <, >, [, ], or special chars)
sanitized = []
seen = {}
for col in feature_cols_original:
    s = re.sub(r'[^0-9a-zA-Z_]', '_', col)
    if s in seen:
        seen[s] += 1
        s = f"{s}_{seen[s]}"
    else:
        seen[s] = 0
    sanitized.append(s)

mapping = dict(zip(feature_cols_original, sanitized))
df_processed = df_processed.rename(columns=mapping)
feature_cols = sanitized
X_features = df_processed[feature_cols]

print(f"📊 Feature count: {len(feature_cols)}")
print(f"📊 Feature names: {feature_cols}")

# Nhãn mục tiêu (A-F)
target = df_raw['final_grade']

print(f"📊 Target distribution:")
print(target.value_counts().sort_index())
print()

# Encode target labels to integers for XGBoost
label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)
num_class = len(label_encoder.classes_)

# 3. Train/Test Split (sử dụng stratify để giữ phân bố lớp)
X_train, X_test, y_train, y_test = train_test_split(
    X_features,
    target_encoded,
    test_size=0.2,
    random_state=42,
    stratify=target_encoded
)
print(f"📊 Train set: {X_train.shape[0]} rows, Test set: {X_test.shape[0]} rows\n")

# 5. XGBoost Model
clf_xgb = xgb.XGBClassifier(
    objective='multi:softprob',
    eval_metric='mlogloss',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)

# 6. Randomized Search
print("="*60)
print("Hyperparameter Tuning with RandomizedSearchCV...")
print("="*60 + "\n")

param_grid = {
    'max_depth': [3, 4, 5, 6],
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.03, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [3, 5, 10, 15],
    'reg_alpha': [0, 0.1, 0.5, 1],
    'reg_lambda': [1, 2, 5, 10],
    'gamma': [0, 0.1, 0.5]
}

search_cv = RandomizedSearchCV(
    estimator=clf_xgb,
    param_distributions=param_grid,
    n_iter=30,
    cv=5,
    scoring='f1_macro',
    random_state=42,
    verbose=1,
    n_jobs=-1
)

print("⏳ Training model...")

search_cv.fit(X_train, y_train)

selected_model = search_cv.best_estimator_

print("\nBest Parameters:")
print(search_cv.best_params_)

print("\nBest CV Score (F1-Macro):")
print(f"{search_cv.best_score_:.4f}")

# 7. Prediction
y_pred = selected_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\n{'='*60}")
print(f"[XGBOOST] Test Accuracy: {acc*100:.2f}%")
print(f"{'='*60}")

print("\nClassification Report:")

target_labels = list(label_encoder.classes_)
print(classification_report(y_test, y_pred, target_names=target_labels))

# 8. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=target_labels,
    yticklabels=target_labels
)

plt.xlabel("Predicted")

plt.ylabel("True")

plt.title("Confusion Matrix - XGBoost")

plt.tight_layout()

plt.show()

# 9. Kiểm tra Overfitting
y_train_pred = selected_model.predict(X_train)

y_test_pred = selected_model.predict(X_test)

f1_train = f1_score(

    y_train,

    y_train_pred,

    average='macro'

)

f1_test = f1_score(

    y_test,

    y_test_pred,

    average='macro'

)

print("\n==============================")

print("Overfitting Check")

print(f"Macro-F1 Train : {f1_train:.4f}")

print(f"Macro-F1 Test  : {f1_test:.4f}")

print(f"Difference      : {f1_train-f1_test:.4f}")

if f1_train - f1_test > 0.10:

    print("⚠️ Overfitting detected")

elif f1_test < 0.45:

    print("⚠️ Underfitting detected")

else:

    print("✅ Model is stable")

# 10. Cross-Validation trên toàn bộ dữ liệu
# Lưu ý: selected_model đã được chọn (tune hyperparameter) dựa trên X_train,
# nên số liệu CV này chủ yếu phản ánh ĐỘ ỔN ĐỊNH (variance) qua các fold,
# KHÔNG phải một ước lượng generalization độc lập với dữ liệu train.
print("\n" + "="*60)
print("Cross-Validation on full dataset (5-fold)...")
print("="*60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(selected_model, X_features, target_encoded, cv=cv, scoring='f1_macro', n_jobs=-1)
print(f"CV Scores: {[f'{s:.4f}' for s in cv_scores]}")
print(f"Mean CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})\n")

# 11. Feature Importance
print("="*60)
print("Top 10 Most Important Features")
print("="*60)
try:
    importances = selected_model.feature_importances_
    feature_names = feature_cols
    indices = np.argsort(importances)[::-1][:10]
    for i, idx in enumerate(indices):
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
except Exception as e:
    print(f"⚠️  Could not extract feature importance: {e}")

print()
    
# 12. Learning Curve
print("⏳ Generating Learning Curve...")

train_sizes, train_scores, val_scores = learning_curve(
    estimator=selected_model,
    X=X_train,
    y=y_train,
    cv=5,
    scoring='f1_macro',
    train_sizes=np.linspace(0.1,1.0,10),
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)

train_std = train_scores.std(axis=1)

val_mean = val_scores.mean(axis=1)

val_std = val_scores.std(axis=1)

plt.figure(figsize=(9,5))

plt.plot(

    train_sizes,

    train_mean,

    'o-',

    label='Train F1'

)

plt.plot(

    train_sizes,

    val_mean,

    's-',

    label='Validation F1'

)

plt.fill_between(

    train_sizes,

    train_mean-train_std,

    train_mean+train_std,

    alpha=0.2

)

plt.fill_between(

    train_sizes,

    val_mean-val_std,

    val_mean+val_std,

    alpha=0.2

)

plt.xlabel("Number of Training Samples")

plt.ylabel("Macro F1")

plt.title("Learning Curve - XGBoost")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig("../results/learning_curve_xgboost.png", dpi=150)

plt.show()

# 13. Save Model
print("="*60)
print("Saving Model...")
print("="*60)

save_dir = '../models'
os.makedirs(save_dir, exist_ok=True)

model_path = os.path.join(save_dir, 'xgboost_model.pkl')
joblib.dump(selected_model, model_path)
print(f"XGBoost model saved to: {model_path}")
encoder_path = os.path.join(save_dir, 'label_encoder_xgboost.pkl')
joblib.dump(label_encoder, encoder_path)
print(f"Label encoder saved to: {encoder_path}")
print("\n🎉 Pipeline completed successfully!")