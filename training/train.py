from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

try:
    import msvc_runtime  # noqa: F401 - helps LightGBM DLL dependencies load on some Windows setups.
except ModuleNotFoundError:
    pass


TARGET_COLUMN = "final_grade"
IGNORED_INPUT_COLUMNS = ["student_id"]
DEFAULT_EXCLUDED_INPUT_COLUMNS = ["overall_score"]
GRADE_ORDER = ["a", "b", "c", "d", "e", "f"]


def build_features(
    df: pd.DataFrame,
    excluded_input_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    missing_columns = {TARGET_COLUMN} - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required column(s): {sorted(missing_columns)}")

    target = df[TARGET_COLUMN].astype(str).str.lower()
    ignored_columns = [
        TARGET_COLUMN,
        *IGNORED_INPUT_COLUMNS,
        *(excluded_input_columns or []),
    ]
    features = df.drop(columns=[col for col in ignored_columns if col in df.columns])
    return features, target


def get_input_schema(
    feature_columns: list[str],
    categorical_columns: list[str],
    categories: dict[str, list[str]],
) -> list[dict[str, object]]:
    categorical_set = set(categorical_columns)
    return [
        {
            "name": column,
            "kind": "category" if column in categorical_set else "number",
            "options": categories.get(column, []),
        }
        for column in feature_columns
    ]


def prepare_categorical_features(
    train_x: pd.DataFrame, test_x: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, list[str]]]:
    categorical_cols = train_x.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    categories: dict[str, list[str]] = {}

    for col in categorical_cols:
        values = sorted(train_x[col].dropna().astype(str).unique().tolist())
        categories[col] = values
        dtype = pd.CategoricalDtype(categories=values)
        train_x[col] = train_x[col].astype(str).astype(dtype)
        test_x[col] = test_x[col].astype(str).astype(dtype)

    return train_x, test_x, categorical_cols, categories


def train_model(
    data_path: Path,
    output_dir: Path,
    metrics_dir: Path | None = None,
    test_size: float = 0.25,
    random_state: int = 42,
    model_filename: str = "student_lightgbm_model.joblib",
    metrics_prefix: str = "",
    excluded_input_columns: list[str] | None = None,
) -> dict[str, object]:
    df = pd.read_csv(data_path)
    excluded_input_columns = excluded_input_columns or DEFAULT_EXCLUDED_INPUT_COLUMNS
    features, target = build_features(df, excluded_input_columns=excluded_input_columns)

    train_x, test_x, train_y, test_y = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    train_x = train_x.copy()
    test_x = test_x.copy()
    train_x, test_x, categorical_cols, categories = prepare_categorical_features(train_x, test_x)

    model = LGBMClassifier(
        objective="multiclass",
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=10,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        verbose=-1,
    )
    model.fit(train_x, train_y, categorical_feature=categorical_cols)

    predictions = model.predict(test_x)
    accuracy = accuracy_score(test_y, predictions)
    report = classification_report(
        test_y,
        predictions,
        labels=GRADE_ORDER,
        zero_division=0,
        output_dict=True,
    )
    report_text = classification_report(
        test_y,
        predictions,
        labels=GRADE_ORDER,
        zero_division=0,
    )
    matrix = confusion_matrix(test_y, predictions, labels=GRADE_ORDER)

    metrics_dir = metrics_dir or output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / model_filename
    metrics_path = metrics_dir / f"{metrics_prefix}metrics.json"
    report_path = metrics_dir / f"{metrics_prefix}classification_report.txt"
    confusion_path = metrics_dir / f"{metrics_prefix}confusion_matrix.csv"
    importance_path = metrics_dir / f"{metrics_prefix}feature_importance.csv"

    joblib.dump(
        {
            "model": model,
            "feature_columns": features.columns.tolist(),
            "categorical_columns": categorical_cols,
            "categories": categories,
            "input_schema": get_input_schema(
                features.columns.tolist(),
                categorical_cols,
                categories,
            ),
            "grade_order": GRADE_ORDER,
            "excluded_input_columns": [*IGNORED_INPUT_COLUMNS, *excluded_input_columns],
        },
        model_path,
    )

    metrics = {
        "dataset_rows": int(len(df)),
        "train_rows": int(len(train_x)),
        "test_rows": int(len(test_x)),
        "accuracy": float(accuracy),
        "grade_distribution": target.value_counts().reindex(GRADE_ORDER, fill_value=0).to_dict(),
        "test_size": test_size,
        "random_state": random_state,
        "excluded_input_columns": [*IGNORED_INPUT_COLUMNS, *excluded_input_columns],
        "target_source": TARGET_COLUMN,
        "feature_columns": features.columns.tolist(),
        "classification_report": report,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(report_text, encoding="utf-8")
    pd.DataFrame(matrix, index=GRADE_ORDER, columns=GRADE_ORDER).to_csv(confusion_path)
    pd.DataFrame(
        {
            "feature": features.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False).to_csv(importance_path, index=False)

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "report_path": str(report_path),
        "confusion_matrix_path": str(confusion_path),
        "feature_importance_path": str(importance_path),
        "accuracy": accuracy,
        "report_text": report_text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train LightGBM to classify student final_grade from the current dataset."
    )
    parser.add_argument("--data", type=Path, default=Path("data/raw/Student_Performance.csv"))
    parser.add_argument("--out", type=Path, default=Path("models"))
    parser.add_argument("--metrics-out", type=Path, default=Path("reports/metrics"))
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--model-filename",
        default="student_lightgbm_model.joblib",
        help="Output model filename inside the output directory.",
    )
    parser.add_argument(
        "--metrics-prefix",
        default="",
        help="Prefix for metrics/report/importance files.",
    )
    parser.add_argument(
        "--include-overall-score",
        action="store_true",
        help="Include overall_score as an input. By default it is excluded to avoid target leakage.",
    )
    args = parser.parse_args()

    excluded_input_columns = [] if args.include_overall_score else DEFAULT_EXCLUDED_INPUT_COLUMNS

    results = train_model(
        args.data,
        args.out,
        metrics_dir=args.metrics_out,
        test_size=args.test_size,
        random_state=args.random_state,
        model_filename=args.model_filename,
        metrics_prefix=args.metrics_prefix,
        excluded_input_columns=excluded_input_columns,
    )
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(results["report_text"])
    print(f"Saved model: {results['model_path']}")
    print(f"Saved metrics: {results['metrics_path']}")
    print(f"Saved confusion matrix: {results['confusion_matrix_path']}")
    print(f"Saved feature importance: {results['feature_importance_path']}")


if __name__ == "__main__":
    main()
