"""Sepsis early-warning API.

Run locally:  uvicorn src.api:app --reload
Docs:         http://127.0.0.1:8000/docs
"""
from typing import Literal, Optional

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from src.features import LAB_FEATS, VITAL_FEATS
from src.serving import SepsisScorer

scorer = SepsisScorer()
app = FastAPI(
    title="Sepsis Early-Warning API",
    version="1.0",
    description="Hourly sepsis risk from ICU vitals and labs. Research prototype; not for clinical use.",
)


class Observation(BaseModel):
    ICULOS: int = Field(ge=1, le=2000, description="Hour of ICU stay (1 = first hour)")
    HR: Optional[float] = Field(None, ge=0, le=300, description="Heart rate (beats/min)")
    O2Sat: Optional[float] = Field(None, ge=0, le=100, description="Pulse oximetry (%)")
    Temp: Optional[float] = Field(None, ge=25, le=45, description="Temperature (°C)")
    SBP: Optional[float] = Field(None, ge=0, le=300, description="Systolic BP (mm Hg)")
    MAP: Optional[float] = Field(None, ge=0, le=300, description="Mean arterial pressure (mm Hg)")
    DBP: Optional[float] = Field(None, ge=0, le=300, description="Diastolic BP (mm Hg)")
    Resp: Optional[float] = Field(None, ge=0, le=100, description="Respiratory rate (breaths/min)")
    labs: dict[str, float] = Field(default_factory=dict,
                                   description="Lab results measured this hour, e.g. {'Lactate': 2.1}")

    @field_validator("labs")
    @classmethod
    def known_labs(cls, v):
        unknown = set(v) - set(LAB_FEATS)
        if unknown:
            raise ValueError(f"Unknown lab names: {sorted(unknown)}. Allowed: {LAB_FEATS}")
        return v


class PredictRequest(BaseModel):
    patient_id: str
    age: float = Field(ge=0, le=120)
    gender: Literal[0, 1] = Field(description="0 = female, 1 = male (dataset coding)")
    unit: Optional[Literal["MICU", "SICU"]] = Field(None, description="Medical or surgical ICU")
    hosp_adm_time: float = Field(0.0, le=0, description="Hours from hospital admission to ICU admission (≤ 0)")
    observations: list[Observation] = Field(min_length=1, description="All hours so far, any order")

    @field_validator("observations")
    @classmethod
    def unique_hours(cls, v):
        hours = [o.ICULOS for o in v]
        if len(hours) != len(set(hours)):
            raise ValueError("Each ICULOS hour may appear only once")
        return v


class Reason(BaseModel):
    feature: str
    value: Optional[float]
    contribution: float


class PredictResponse(BaseModel):
    patient_id: str
    risk: float
    alert: bool
    threshold: float
    hour: int
    top_reasons: list[Reason]
    model_run_id: str


def to_frame(req: PredictRequest) -> pd.DataFrame:
    rows = [{"ICULOS": o.ICULOS, **{v: getattr(o, v) for v in VITAL_FEATS}, **o.labs}
            for o in req.observations]
    df = pd.DataFrame(rows).sort_values("ICULOS")
    df["Age"] = req.age
    df["Gender"] = req.gender
    df["Unit1"] = {"MICU": 1.0, "SICU": 0.0}.get(req.unit)    # empty if unit not given
    df["Unit2"] = {"MICU": 0.0, "SICU": 1.0}.get(req.unit)
    df["HospAdmTime"] = req.hosp_adm_time
    return df


@app.get("/health")
def health():
    return {"status": "ok", "model_run_id": scorer.run_id}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    return {"patient_id": req.patient_id, **scorer.score(to_frame(req))}