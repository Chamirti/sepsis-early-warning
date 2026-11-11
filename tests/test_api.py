"""API tests: health check, valid prediction, input validation, and training/serving consistency."""
from pathlib import Path

import duckdb
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.serving import SepsisScorer

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "sepsis.duckdb"
client = TestClient(app)
EXAMPLE = {
    "patient_id": "test-001", "age": 67, "gender": 1, "unit": "MICU", "hosp_adm_time": -5,
    "observations": [
        {"ICULOS": 1, "HR": 88, "MAP": 80, "Resp": 18, "Temp": 37.1},
        {"ICULOS": 2, "HR": 104, "MAP": 66, "Resp": 24, "Temp": 38.3, "labs": {"Lactate": 2.6}},
    ],
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_returns_valid_risk():
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["risk"] <= 1
    assert body["hour"] == 2
    assert len(body["top_reasons"]) == 5


@pytest.mark.parametrize("bad", [
    {"observations": [{"ICULOS": 1, "HR": 500}]},                    # impossible heart rate
    {"observations": [{"ICULOS": 1, "labs": {"Unicorn": 1.0}}]},     # unknown lab
    {"observations": [{"ICULOS": 1}, {"ICULOS": 1}]},                # duplicate hour
    {"observations": []},                                            # no data
])
def test_invalid_input_rejected(bad):
    r = client.post("/predict", json={**EXAMPLE, **bad})
    assert r.status_code == 422


@pytest.mark.skipif(not DB_PATH.exists(), reason="needs the local database")
def test_serving_matches_training():
    """The API's risk for a real patient must equal the training pipeline's prediction."""
    scorer = SepsisScorer()
    con = duckdb.connect(str(DB_PATH), read_only=True)
    pid = con.execute("SELECT patient_id FROM cohort WHERE split = 'test' AND septic = 1 LIMIT 1").fetchone()[0]
    raw = con.execute(f"SELECT * FROM observations WHERE patient_id = '{pid}' ORDER BY ICULOS").df()
    stored = con.execute(f"SELECT {', '.join(scorer.features)} FROM features "
                         f"WHERE patient_id = '{pid}' ORDER BY ICULOS").df()
    con.close()

    served = scorer.score(raw)["risk"]
    expected = float(scorer.booster.predict(stored.iloc[[-1]].astype("float32"))[0])
    assert np.isclose(served, expected, atol=1e-6)