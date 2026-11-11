"""Sepsis early-warning API.

Run locally:  uvicorn src.api:app --reload
Docs:         http://127.0.0.1:8000/docs
"""
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import Body, FastAPI, HTTPException

from src.features import VITAL_FEATS
from src.fhir import bundle_to_request
from src.schemas import FhirPredictResponse, PredictRequest, PredictResponse
from src.serving import SepsisScorer

scorer = SepsisScorer()
app = FastAPI(
    title="Sepsis Early-Warning API",
    version="1.0",
    description="Hourly sepsis risk from ICU vitals and labs. Research prototype; not for clinical use.",
)


def to_frame(req: PredictRequest) -> pd.DataFrame:
    rows = [{"ICULOS": o.ICULOS, **{v: getattr(o, v) for v in VITAL_FEATS}, **o.labs}
            for o in req.observations]
    df = pd.DataFrame(rows).sort_values("ICULOS")
    df["Age"] = req.age
    df["Gender"] = req.gender
    df["Unit1"] = {"MICU": 1.0, "SICU": 0.0}.get(req.unit)    # missing if unit not given
    df["Unit2"] = {"MICU": 0.0, "SICU": 1.0}.get(req.unit)
    # Never missing in training, so fill with the Hospital A training median (-2.59 h) if not given
    df["HospAdmTime"] = req.hosp_adm_time if req.hosp_adm_time is not None else -2.59
    return df


@app.get("/health")
def health():
    return {"status": "ok", "model_run_id": scorer.run_id}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    return {"patient_id": req.patient_id, **scorer.score(to_frame(req))}


@app.post("/predict/fhir", response_model=FhirPredictResponse)
def predict_fhir(bundle: dict = Body(..., description="FHIR R4 Bundle with one Patient and Observations"),
                 icu_admission: Optional[datetime] = None):
    try:
        req, ignored = bundle_to_request(bundle, icu_admission)
    except ValueError as e:                     # includes pydantic validation errors
        raise HTTPException(status_code=422, detail=str(e))
    return {"patient_id": req.patient_id, "ignored_observations": ignored,
            **scorer.score(to_frame(req))}