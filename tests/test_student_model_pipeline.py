import pandas as pd
from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app.services.prediction import GRADE_LABELS, prepare_prediction_input, predict_student
from training.train import DEFAULT_EXCLUDED_INPUT_COLUMNS, build_features, get_input_schema


def test_build_features_uses_final_grade_and_excludes_non_input_columns():
    df = pd.DataFrame(
        {
            "student_id": [1, 2],
            "age": [16, 17],
            "gender": ["female", "male"],
            "overall_score": [72.5, 84.1],
            "final_grade": ["c", "b"],
        }
    )

    features, target = build_features(df, excluded_input_columns=[])

    assert list(features.columns) == ["age", "gender", "overall_score"]
    assert target.tolist() == ["c", "b"]


def test_build_features_excludes_overall_score_but_keeps_subject_scores_by_default():
    df = pd.DataFrame(
        {
            "student_id": [1],
            "age": [16],
            "math_score": [72.5],
            "science_score": [80.0],
            "english_score": [75.0],
            "overall_score": [76.0],
            "final_grade": ["c"],
        }
    )

    features, target = build_features(df, excluded_input_columns=DEFAULT_EXCLUDED_INPUT_COLUMNS)

    assert list(features.columns) == ["age", "math_score", "science_score", "english_score"]
    assert target.tolist() == ["c"]


def test_get_input_schema_describes_numeric_and_categorical_fields():
    feature_columns = ["age", "gender", "study_hours"]
    categorical_columns = ["gender"]
    categories = {"gender": ["female", "male"]}

    schema = get_input_schema(feature_columns, categorical_columns, categories)

    assert schema == [
        {"name": "age", "kind": "number", "options": []},
        {"name": "gender", "kind": "category", "options": ["female", "male"]},
        {"name": "study_hours", "kind": "number", "options": []},
    ]


def test_predict_student_returns_final_grade_and_probabilities():
    class FakeModel:
        classes_ = ["a", "b", "c"]

        def predict(self, features):
            assert list(features.columns) == ["age", "gender", "study_hours"]
            return ["b"]

        def predict_proba(self, features):
            return [[0.1, 0.7, 0.2]]

    bundle = {
        "model": FakeModel(),
        "feature_columns": ["age", "gender", "study_hours"],
        "categorical_columns": ["gender"],
        "categories": {"gender": ["female", "male"]},
        "input_schema": [
            {"name": "age", "kind": "number", "options": []},
            {"name": "gender", "kind": "category", "options": ["female", "male"]},
            {"name": "study_hours", "kind": "number", "options": []},
        ],
    }
    student = pd.DataFrame([{"age": 17, "gender": "female", "study_hours": 4.5}])

    result = predict_student(student, bundle)

    assert result["predicted_grade"] == "Giỏi"
    assert result["predicted_grade_code"] == "b"
    assert result["probabilities"] == {
        "Xuất sắc": 0.1,
        "Giỏi": 0.7,
        "Khá": 0.2,
    }
    assert result["probability_codes"] == {"a": 0.1, "b": 0.7, "c": 0.2}


def test_grade_label_mapping_uses_requested_vietnamese_labels():
    assert GRADE_LABELS == {
        "a": "Xuất sắc",
        "b": "Giỏi",
        "c": "Khá",
        "d": "Trung bình",
        "e": "Yếu",
        "f": "Kém",
    }


def test_prepare_prediction_input_rejects_missing_required_fields():
    bundle = {
        "feature_columns": ["age", "gender", "study_hours"],
        "categorical_columns": ["gender"],
        "categories": {"gender": ["female", "male"]},
    }
    student = pd.DataFrame([{"age": 17, "gender": "female"}])

    try:
        prepare_prediction_input(student, bundle)
    except ValueError as exc:
        assert "Missing required input column(s): ['study_hours']" in str(exc)
    else:
        raise AssertionError("prepare_prediction_input should reject missing fields")


def test_backend_predict_endpoint_returns_prediction():
    class FakeModel:
        classes_ = ["a", "b", "c"]

        def predict(self, features):
            return ["c"]

        def predict_proba(self, features):
            return [[0.05, 0.15, 0.8]]

    bundle = {
        "model": FakeModel(),
        "feature_columns": ["age", "gender", "study_hours"],
        "categorical_columns": ["gender"],
        "categories": {"gender": ["female", "male"]},
        "input_schema": [
            {"name": "age", "kind": "number", "options": []},
            {"name": "gender", "kind": "category", "options": ["female", "male"]},
            {"name": "study_hours", "kind": "number", "options": []},
        ],
    }

    client = TestClient(create_app(bundle))
    response = client.post(
        "/predict",
        json={"age": 16, "gender": "male", "study_hours": 2.5},
    )

    assert response.status_code == 200
    assert response.json()["predicted_grade"] == "Khá"
    assert response.json()["predicted_grade_code"] == "c"


def test_backend_allows_frontend_dev_origin():
    bundle = {
        "model": object(),
        "feature_columns": [],
        "categorical_columns": [],
        "categories": {},
        "input_schema": [],
    }
    client = TestClient(create_app(bundle))

    response = client.options(
        "/predict",
        headers={
            "Origin": "http://127.0.0.1:5173",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"
