import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from lightgbm import LGBMClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "Student_Performance_processed.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def main():
    if not os.path.exists(DATA_PATH):
        print(f"Data file not found at: {DATA_PATH}")
        sys.exit(1)

    df_encoded = pd.read_csv(DATA_PATH)

    objective_features = [
        "age",
        "study_hours",
        "attendance_percentage",
        "math_score",
        "science_score",
        "english_score",
        "school_type",
        "internet_access",
        "extra_activities",
        "parent_education",
        "gender_male",
        "gender_female",
        "gender_other",
        "travel_time_<15 min",
        "travel_time_15-30 min",
        "travel_time_30-60 min",
        "travel_time_>60 min",
        "study_method_notes",
        "study_method_textbook",
        "study_method_group study",
        "study_method_coaching",
        "study_method_mixed",
        "study_method_online videos",
    ]

    df_encoded["study_hours_x_attendance"] = (
        df_encoded["study_hours"] * df_encoded["attendance_percentage"]
    )
    df_encoded["study_hours_squared"] = df_encoded["study_hours"] ** 2
    df_encoded["attendance_squared"] = df_encoded["attendance_percentage"] ** 2
    subject_score_features = ["math_score", "science_score", "english_score"]
    df_encoded["subject_score_mean"] = df_encoded[subject_score_features].mean(axis=1)
    df_encoded["subject_score_min"] = df_encoded[subject_score_features].min(axis=1)
    df_encoded["subject_score_max"] = df_encoded[subject_score_features].max(axis=1)
    df_encoded["subject_score_range"] = (
        df_encoded["subject_score_max"] - df_encoded["subject_score_min"]
    )
    df_encoded["subject_score_std"] = df_encoded[subject_score_features].std(axis=1)
    objective_features += [
        "study_hours_x_attendance",
        "study_hours_squared",
        "attendance_squared",
        "subject_score_mean",
        "subject_score_min",
        "subject_score_max",
        "subject_score_range",
        "subject_score_std",
    ]

    X = df_encoded[objective_features]
    y = df_encoded["final_grade"]

    print(f"Loaded processed data: {df_encoded.shape[0]} rows, {df_encoded.shape[1]} columns.")
    print(f"Feature count: {len(objective_features)}")
    print("\nTarget distribution:")
    print(y.value_counts().sort_index())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print(f"\nTrain set: {X_train.shape[0]} rows, Test set: {X_test.shape[0]} rows")

    model = LGBMClassifier(
        objective="multiclass",
        n_estimators=180,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=7,
        min_child_samples=50,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.85,
        class_weight="balanced",
        reg_alpha=0.2,
        reg_lambda=5.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    print("\n" + "=" * 60)
    print("Training LightGBM Classifier...")
    print("=" * 60)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    target_labels = sorted(y.unique())

    print(f"\n{'=' * 60}")
    print(f"[LIGHTGBM] Test Accuracy: {acc * 100:.2f}%")
    print(f"{'=' * 60}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_labels))

    cm = confusion_matrix(y_test, y_pred, labels=target_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=target_labels,
        yticklabels=target_labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - LightGBM Classifier")
    plt.tight_layout()
    # plt.show()

    y_train_pred = model.predict(X_train)
    f1_train = f1_score(y_train, y_train_pred, average="macro")
    f1_test = f1_score(y_test, y_pred, average="macro")
    print("\nOverfitting Check")
    print(f"Macro-F1 Train : {f1_train:.4f}")
    print(f"Macro-F1 Test  : {f1_test:.4f}")
    print(f"Difference     : {f1_train - f1_test:.4f}")

    print("\nCross-Validation on full dataset (3-fold)...")
    cv_f1_scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
    cv_acc_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"CV F1 Scores: {[f'{score:.4f}' for score in cv_f1_scores]}")
    print(f"Mean CV F1 Score: {cv_f1_scores.mean():.4f} (+/- {cv_f1_scores.std():.4f})")
    print(f"CV Accuracy Scores: {[f'{score:.4f}' for score in cv_acc_scores]}")
    print(f"Mean CV Accuracy: {cv_acc_scores.mean():.4f} (+/- {cv_acc_scores.std():.4f})")

    print("\nTop 15 most important features:")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    for i in range(min(15, len(objective_features))):
        print(f"  {i + 1}. {objective_features[indices[i]]} ({importances[indices[i]]:.4f})")

    print("\nGenerating Learning Curve...")
    train_sizes, train_scores, val_scores = learning_curve(
        estimator=model,
        X=X_train,
        y=y_train,
        cv=cv,
        scoring="f1_macro",
        train_sizes=np.linspace(0.2, 1.0, 5),
        n_jobs=1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    plt.figure(figsize=(9, 5))
    plt.plot(train_sizes, train_mean, "o-", label="Train F1")
    plt.plot(train_sizes, val_mean, "s-", label="Validation F1")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)
    plt.xlabel("Number of Training Samples")
    plt.ylabel("Macro F1")
    plt.title("Learning Curve - LightGBM")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    learning_curve_path = os.path.join(RESULTS_DIR, "learning_curve_lightgbm.png")
    plt.savefig(learning_curve_path, dpi=150)
    # plt.show()
    print(f"Learning curve saved to: {learning_curve_path}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "lightgbm_model.pkl")
    joblib.dump(model, model_path)
    print(f"\nLightGBM model saved successfully to: {model_path}!")
    print(f"Model uses {len(objective_features)} features.")


if __name__ == "__main__":
    main()
