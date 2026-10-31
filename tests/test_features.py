"""Leakage test: features at hour t must not depend on data after hour t."""
import duckdb
import numpy as np
import pandas as pd

from src.features import feature_sql, VITAL_FEATS, LAB_FEATS, STATIC


def synthetic_patient(n_hours=40, seed=0):
    """A fake patient with missing values, starting at hour 3 like some real records."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"patient_id": "p_test", "ICULOS": np.arange(3, 3 + n_hours)})
    for c in VITAL_FEATS:
        vals = rng.normal(80, 10, n_hours)
        vals[rng.random(n_hours) < 0.2] = np.nan            # vitals ~20% missing
        df[c] = pd.array(vals, dtype="Float64")             # NaN becomes a proper missing value
    for c in LAB_FEATS:
        vals = rng.normal(5, 1, n_hours)
        vals[rng.random(n_hours) < 0.9] = np.nan            # labs ~90% missing
        df[c] = pd.array(vals, dtype="Float64")
    for c in STATIC:
        df[c] = 1.0
    df["hospital"], df["split"], df["SepsisLabel"] = "A", "train", 0
    return df


def compute_features(df):
    con = duckdb.connect()                      # in-memory database
    con.register("src", df)
    out = con.execute(feature_sql("src") + " ORDER BY ICULOS").df()
    con.close()
    return out


def test_one_row_per_input_row():
    df = synthetic_patient()
    assert len(compute_features(df)) == len(df)


def test_features_use_only_past_data():
    df = synthetic_patient()
    full = compute_features(df)
    for cutoff in [5, 12, 25, int(df["ICULOS"].max())]:
        partial = compute_features(df[df["ICULOS"] <= cutoff])
        row_full = full[full["ICULOS"] == cutoff].reset_index(drop=True)
        row_partial = partial[partial["ICULOS"] == cutoff].reset_index(drop=True)
        pd.testing.assert_frame_equal(row_full, row_partial, check_dtype=False)