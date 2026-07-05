from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from ..services.form_renderer import render_form
from ..services.prediction import predict_student


def _coerce_payload(payload: dict[str, Any], bundle: dict[str, object]) -> pd.DataFrame:
    schema = list(bundle["input_schema"])
    row: dict[str, Any] = {}
    for field in schema:
        name = str(field["name"])
        if name not in payload or payload[name] in (None, ""):
            raise ValueError(f"Missing required input field: {name}")
        row[name] = payload[name]
    return pd.DataFrame([row])


def create_predict_router(bundle: dict[str, object]) -> APIRouter:
    router = APIRouter()

    @router.get("/", response_class=HTMLResponse)
    def home() -> str:
        return render_form(bundle)

    @router.get("/schema")
    def schema() -> dict[str, object]:
        return {"fields": bundle["input_schema"]}

    @router.post("/predict")
    async def predict(payload: dict[str, Any]) -> dict[str, object]:
        try:
            student = _coerce_payload(payload, bundle)
            return predict_student(student, bundle)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/predict-form", response_class=HTMLResponse)
    async def predict_form(request: Request) -> str:
        form = await request.form()
        try:
            student = _coerce_payload(dict(form), bundle)
            result = predict_student(student, bundle)
            return render_form(bundle, result)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return router
