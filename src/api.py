"""FastAPI app exposing prediction endpoint for student performance."""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from utils import decode_prediction_label, load_model_artifact, prepare_input_dataframe

app = FastAPI(title="Student Performance Prediction API", version="1.0.0")


class StudentFeatures(BaseModel):
    """Input schema for a single student prediction request."""

    school: Optional[str] = Field(default=None, examples=["GP"])
    sex: Optional[str] = Field(default=None, examples=["F"])
    age: Optional[int] = Field(default=None, examples=[17])
    address: Optional[str] = Field(default=None, examples=["U"])
    famsize: Optional[str] = Field(default=None, examples=["GT3"])
    Pstatus: Optional[str] = Field(default=None, examples=["T"])
    Medu: Optional[int] = Field(default=None, examples=[4])
    Fedu: Optional[int] = Field(default=None, examples=[4])
    Mjob: Optional[str] = Field(default=None, examples=["teacher"])
    Fjob: Optional[str] = Field(default=None, examples=["services"])
    reason: Optional[str] = Field(default=None, examples=["course"])
    guardian: Optional[str] = Field(default=None, examples=["mother"])
    traveltime: Optional[int] = Field(default=None, examples=[1])
    studytime: Optional[int] = Field(default=None, examples=[2])
    failures: Optional[int] = Field(default=None, examples=[0])
    schoolsup: Optional[str] = Field(default=None, examples=["no"])
    famsup: Optional[str] = Field(default=None, examples=["yes"])
    paid: Optional[str] = Field(default=None, examples=["no"])
    activities: Optional[str] = Field(default=None, examples=["yes"])
    nursery: Optional[str] = Field(default=None, examples=["yes"])
    higher: Optional[str] = Field(default=None, examples=["yes"])
    internet: Optional[str] = Field(default=None, examples=["yes"])
    romantic: Optional[str] = Field(default=None, examples=["no"])
    famrel: Optional[int] = Field(default=None, examples=[4])
    freetime: Optional[int] = Field(default=None, examples=[3])
    goout: Optional[int] = Field(default=None, examples=[3])
    Dalc: Optional[int] = Field(default=None, examples=[1])
    Walc: Optional[int] = Field(default=None, examples=[1])
    health: Optional[int] = Field(default=None, examples=[5])
    absences: Optional[int] = Field(default=None, examples=[2])
    G1: Optional[int] = Field(default=None, examples=[12])
    G2: Optional[int] = Field(default=None, examples=[13])


@app.on_event("startup")
def startup_event() -> None:
    """Load model artifact once when the API starts."""
    artifact = load_model_artifact()
    app.state.model = artifact["model"]
    app.state.feature_columns = artifact["feature_columns"]


@app.get("/")
def healthcheck() -> dict[str, str]:
    """Simple endpoint to verify service readiness."""
    return {"status": "ok", "message": "Student performance API is running."}


@app.post("/predict")
def predict(features: StudentFeatures) -> dict:
    """Predict student pass/fail result from provided student features."""
    payload = features.model_dump()
    model_input = prepare_input_dataframe(payload, app.state.feature_columns)
    prediction = int(app.state.model.predict(model_input)[0])

    if hasattr(app.state.model, "predict_proba"):
        probability = float(app.state.model.predict_proba(model_input)[0][1])
    else:
        probability = None

    return {
        "prediction": prediction,
        "label": decode_prediction_label(prediction),
        "probability_pass": probability,
    }
