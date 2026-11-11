"""Request and response schemas, shared by the JSON and FHIR endpoints."""
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from src.features import LAB_FEATS


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
    # Changed: now optional. If unknown, it is left missing rather than assumed to be 0.
    hosp_adm_time: Optional[float] = Field(None, le=0,
                                           description="Hours from hospital admission to ICU admission (≤ 0); omit if unknown")
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


# New: the FHIR endpoint also reports how many observations it skipped
class FhirPredictResponse(PredictResponse):
    ignored_observations: int