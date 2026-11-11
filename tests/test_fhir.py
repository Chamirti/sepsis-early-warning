"""FHIR endpoint tests, including that FHIR and JSON inputs give identical predictions."""
from datetime import datetime, timezone

import numpy as np
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def obs(code, value, unit, time):
    return {"resource": {
        "resourceType": "Observation", "status": "final",
        "code": {"coding": [{"system": "http://loinc.org", "code": code}]},
        "effectiveDateTime": time,
        "valueQuantity": {"value": value, "code": unit, "system": "http://unitsofmeasure.org"},
    }}


def bundle(*observations):
    return {"resourceType": "Bundle", "type": "collection", "entry": [
        {"resource": {"resourceType": "Patient", "id": "fhir-001", "gender": "male", "birthDate": "1959-03-14"}},
        *observations,
    ]}


BASE = [
    obs("8867-4", 88, "/min", "2026-11-08T08:00:00Z"),
    obs("8478-0", 80, "mm[Hg]", "2026-11-08T08:00:00Z"),
    obs("8867-4", 104, "/min", "2026-11-08T09:10:00Z"),
    obs("8478-0", 66, "mm[Hg]", "2026-11-08T09:10:00Z"),
    obs("8310-5", 38.3, "Cel", "2026-11-08T09:10:00Z"),
    obs("2524-7", 2.6, "mmol/L", "2026-11-08T09:30:00Z"),
]


def test_fhir_matches_json_endpoint():
    """The same data sent as FHIR or as plain JSON must give the same risk."""
    fhir = client.post("/predict/fhir", json=bundle(*BASE))
    assert fhir.status_code == 200, fhir.text

    age = (datetime(2026, 11, 8, 9, 30, tzinfo=timezone.utc)
           - datetime(1959, 3, 14, tzinfo=timezone.utc)).days / 365.25
    plain = client.post("/predict", json={
        "patient_id": "fhir-001", "age": age, "gender": 1,
        "observations": [
            {"ICULOS": 1, "HR": 88, "MAP": 80},
            {"ICULOS": 2, "HR": 104, "MAP": 66, "Temp": 38.3, "labs": {"Lactate": 2.6}},
        ],
    })
    assert plain.status_code == 200, plain.text
    assert np.isclose(fhir.json()["risk"], plain.json()["risk"])


def test_fahrenheit_is_converted():
    in_f = [o if o["resource"]["code"]["coding"][0]["code"] != "8310-5"
            else obs("8310-5", 38.3 * 9 / 5 + 32, "[degF]", "2026-11-08T09:10:00Z") for o in BASE]
    r_c = client.post("/predict/fhir", json=bundle(*BASE)).json()["risk"]
    r_f = client.post("/predict/fhir", json=bundle(*in_f)).json()["risk"]
    assert np.isclose(r_c, r_f)


def test_wrong_unit_rejected():
    r = client.post("/predict/fhir", json=bundle(*BASE, obs("8867-4", 90, "mm[Hg]", "2026-11-08T10:00:00Z")))
    assert r.status_code == 422


def test_unknown_codes_are_skipped_and_counted():
    r = client.post("/predict/fhir", json=bundle(*BASE, obs("99999-9", 1.0, "1", "2026-11-08T09:00:00Z")))
    assert r.status_code == 200
    assert r.json()["ignored_observations"] == 1