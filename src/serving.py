"""Scoring logic for the API: raw hourly observations -> features -> risk and top reasons.

Features are computed with the same feature_sql used in training, so training and
serving cannot drift apart.
"""
import json
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features import LAB_FEATS, STATIC, VITAL_FEATS, feature_sql

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
RAW_COLUMNS = VITAL_FEATS + LAB_FEATS + STATIC


class SepsisScorer:
    def __init__(self, model_dir=MODEL_DIR):
        meta = json.loads((model_dir / "metadata.json").read_text())
        self.features = meta["features"]
        self.threshold = meta["threshold"]
        self.run_id = meta["run_id"]
        self.booster = lgb.Booster(model_file=str(model_dir / "model.txt"))

    def features_for(self, hours: pd.DataFrame) -> pd.DataFrame:
        """Features for every hour of one patient, via the training SQL."""
        df = hours.copy()
        df["patient_id"] = "request"
        for c in RAW_COLUMNS:
            if c not in df:
                df[c] = np.nan
        df = df[["patient_id", "ICULOS"] + RAW_COLUMNS]
        df[RAW_COLUMNS] = df[RAW_COLUMNS].astype("Float64")   # NaN -> proper missing value
        con = duckdb.connect()                                  # in-memory database
        con.register("src", df)
        out = con.execute(feature_sql("src", keep=()) + " ORDER BY ICULOS").df()
        con.close()
        return out

    def score(self, hours: pd.DataFrame, top_k: int = 5) -> dict:
        """Risk at the latest hour, with the features that contributed most."""
        feats = self.features_for(hours)
        x = feats[self.features].iloc[[-1]].astype("float32")
        risk = float(self.booster.predict(x)[0])
        # LightGBM's built-in SHAP values (log-odds); the last column is the baseline
        contrib = self.booster.predict(x, pred_contrib=True)[0][:-1]
        top = np.argsort(-np.abs(contrib))[:top_k]
        reasons = [{
            "feature": self.features[i],
            "value": None if pd.isna(x.iloc[0, i]) else round(float(x.iloc[0, i]), 3),
            "contribution": round(float(contrib[i]), 3),
        } for i in top]
        return {
            "risk": risk,
            "alert": risk >= self.threshold,
            "threshold": self.threshold,
            "hour": int(feats["ICULOS"].iloc[-1]),
            "top_reasons": reasons,
            "model_run_id": self.run_id,
        }