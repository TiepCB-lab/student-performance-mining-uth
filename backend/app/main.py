from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import MODEL_PATH
from .routes.predict import create_predict_router
from .services.prediction import load_model_bundle


ALLOWED_ORIGINS = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
]


def create_app(bundle: dict[str, object] | None = None) -> FastAPI:
    model_bundle = bundle if bundle is not None else load_model_bundle(MODEL_PATH)
    app = FastAPI(title="Student Performance Mining UTH API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(create_predict_router(model_bundle))
    return app


app = create_app()
