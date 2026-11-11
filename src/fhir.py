"""Minimal FHIR R4 support: convert a Bundle (one Patient + Observations) into a PredictRequest.

Supports a subset of LOINC codes: vital signs and common sepsis labs. Codes and units should be
checked against an institution's own usage before relying on them.
"""
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from src.schemas import PredictRequest

# LOINC code -> (model variable, accepted UCUM unit codes)
LOINC = {
    "8867-4":  ("HR", {"/min"}),             # heart rate
    "59408-5": ("O2Sat", {"%"}),             # SpO2 by pulse oximetry
    "2708-6":  ("O2Sat", {"%"}),             # oxygen saturation in arterial blood
    "8310-5":  ("Temp", {"Cel", "[degF]"}),  # body temperature
    "8480-6":  ("SBP", {"mm[Hg]"}),          # systolic blood pressure
    "8478-0":  ("MAP", {"mm[Hg]"}),          # mean blood pressure
    "8462-4":  ("DBP", {"mm[Hg]"}),          # diastolic blood pressure
    "9279-1":  ("Resp", {"/min"}),           # respiratory rate
    "2524-7":  ("Lactate", {"mmol/L"}),      # lactate, serum/plasma
    "32693-4": ("Lactate", {"mmol/L"}),      # lactate, blood
    "6690-2":  ("WBC", {"10*3/uL"}),         # white blood cells
    "777-3":   ("Platelets", {"10*3/uL"}),   # platelets
    "2160-0":  ("Creatinine", {"mg/dL"}),    # creatinine
    "3094-0":  ("BUN", {"mg/dL"}),           # urea nitrogen
    "2345-7":  ("Glucose", {"mg/dL"}),       # glucose
}
VITALS = {"HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"}


def _utc(t: datetime) -> datetime:
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def bundle_to_request(bundle: dict, icu_admission: Optional[datetime] = None) -> tuple[PredictRequest, int]:
    """Returns the request and the number of observations that were skipped."""
    if bundle.get("resourceType") != "Bundle":
        raise ValueError("Expected a FHIR Bundle")
    resources = [e.get("resource", {}) for e in bundle.get("entry", [])]
    patients = [r for r in resources if r.get("resourceType") == "Patient"]
    if len(patients) != 1:
        raise ValueError("Bundle must contain exactly one Patient resource")
    patient = patients[0]

    # 1. Read supported Observations
    readings, ignored = [], 0
    for r in resources:
        if r.get("resourceType") != "Observation":
            continue
        codes = [c.get("code") for c in r.get("code", {}).get("coding", [])
                 if c.get("system") == "http://loinc.org"]
        code = next((c for c in codes if c in LOINC), None)
        q = r.get("valueQuantity", {})
        if code is None or "value" not in q or "effectiveDateTime" not in r:
            ignored += 1
            continue
        var, units = LOINC[code]
        unit = q.get("code")
        if unit is not None and unit not in units:
            raise ValueError(f"Observation {code} ({var}): unit '{unit}' not supported; expected {sorted(units)}")
        value = float(q["value"])
        if unit == "[degF]":
            value = (value - 32) * 5 / 9
        readings.append((_utc(datetime.fromisoformat(r["effectiveDateTime"])), var, value))
    if not readings:
        raise ValueError("No supported Observations found")

    # 2. Timestamps -> ICU hours (hour 1 = ICU admission, or the earliest observation)
    start = _utc(icu_admission) if icu_admission else min(t for t, _, _ in readings)
    by_hour = defaultdict(lambda: defaultdict(list))
    for t, var, value in readings:
        hour = int((t - start).total_seconds() // 3600) + 1
        if hour < 1:
            raise ValueError("Observation is timed before ICU admission")
        by_hour[hour][var].append(value)

    # 3. One row per hour; repeated measurements in the same hour are averaged
    observations = []
    for hour in sorted(by_hour):
        means = {var: sum(vals) / len(vals) for var, vals in by_hour[hour].items()}
        observations.append({"ICULOS": hour,
                             **{k: v for k, v in means.items() if k in VITALS},
                             "labs": {k: v for k, v in means.items() if k not in VITALS}})

    # 4. Patient: gender and age at the latest observation
    gender = {"female": 0, "male": 1}.get(patient.get("gender"))
    if gender is None:
        raise ValueError("Patient.gender must be 'female' or 'male' (the coding the model was trained on)")
    if "birthDate" not in patient:
        raise ValueError("Patient.birthDate is required")
    birth = _utc(datetime.fromisoformat(patient["birthDate"]))
    age = (max(t for t, _, _ in readings) - birth).days / 365.25

    request = PredictRequest(patient_id=patient.get("id", "unknown"), age=age,
                             gender=gender, observations=observations)
    return request, ignored