"""Export the final model from MLflow into models/ for serving.

Run from the project root:  python -m src.export_model
"""
import json

import mlflow

from src.train import MLFLOW_URI, ROOT

RUN_ID = "8eb22f64550f4aaa813a9a521a0fc3f9"   # final model (validated, tested, externally validated)
THRESHOLD = 0.05                               # chosen on Hospital A validation
OUT = ROOT / "models"


def main():
    mlflow.set_tracking_uri(MLFLOW_URI)
    model = mlflow.lightgbm.load_model(f"runs:/{RUN_ID}/model")
    booster = getattr(model, "booster_", model)          # the underlying LightGBM model
    feats = mlflow.artifacts.load_text(f"runs:/{RUN_ID}/features.txt").split("\n")

    OUT.mkdir(exist_ok=True)
    booster.save_model(str(OUT / "model.txt"))
    (OUT / "metadata.json").write_text(json.dumps(
        {"run_id": RUN_ID, "threshold": THRESHOLD, "features": feats}, indent=2))
    print(f"Exported run {RUN_ID} ({len(feats)} features) to {OUT}")


if __name__ == "__main__":
    main()