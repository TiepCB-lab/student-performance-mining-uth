from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd


GRADE_LABELS = {
    "a": "Xuất sắc",
    "b": "Giỏi",
    "c": "Khá",
    "d": "Trung bình",
    "e": "Yếu",
    "f": "Kém",
}


def load_model_bundle(model_path: Path) -> dict[str, object]:
    return joblib.load(model_path)


def prepare_prediction_input(student: pd.DataFrame, bundle: dict[str, object]) -> pd.DataFrame:
    feature_columns = list(bundle["feature_columns"])
    categorical_columns = list(bundle["categorical_columns"])
    categories = dict(bundle["categories"])

    missing_columns = sorted(set(feature_columns) - set(student.columns))
    if missing_columns:
        raise ValueError(f"Missing required input column(s): {missing_columns}")

    features = student[feature_columns].copy()
    numeric_columns = [col for col in feature_columns if col not in categorical_columns]
    for col in numeric_columns:
        features[col] = pd.to_numeric(features[col], errors="raise")

    for col in categorical_columns:
        dtype = pd.CategoricalDtype(categories=categories[col])
        features[col] = features[col].astype(str).astype(dtype)

    return features


def predict_student(student: pd.DataFrame, bundle: dict[str, object]) -> dict[str, object]:
    model = bundle["model"]
    features = prepare_prediction_input(student, bundle)
    predicted_code = str(model.predict(features)[0])
    predicted_grade = GRADE_LABELS.get(predicted_code, predicted_code)
    probabilities = model.predict_proba(features)[0]
    probability_codes = {
        str(label): float(probability)
        for label, probability in zip(model.classes_, probabilities)
    }
    probability_labels = {
        GRADE_LABELS.get(label, label): probability
        for label, probability in probability_codes.items()
    }

    return {
        "predicted_grade": predicted_grade,
        "predicted_grade_code": predicted_code,
        "predicted_label": predicted_grade,
        "probabilities": probability_labels,
        "probability_codes": probability_codes,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Predict student performance with the trained model."
    )
    parser.add_argument("--model", type=Path, default=Path("models/student_lightgbm_model.joblib"))
    parser.add_argument("--input", type=Path, default=Path("data/raw/sample_student_prediction.csv"))
    args = parser.parse_args()

    bundle = load_model_bundle(args.model)
    students = pd.read_csv(args.input)

    for row_number, (_, student) in enumerate(students.iterrows(), start=1):
        result = predict_student(pd.DataFrame([student]), bundle)
        print(f"Student {row_number}: {result['predicted_grade']} ({result['predicted_grade_code']})")
        print("Probabilities:")
        for label, probability in result["probabilities"].items():
            print(f"  {label}: {probability:.4f}")


if __name__ == "__main__":
    main()
