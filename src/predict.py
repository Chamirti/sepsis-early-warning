"""Score a split with a trained LightGBM run from MLflow and save the predictions.

  python -m src.predict --run-name lgbm --split test
"""
import argparse

import duckdb
import mlflow

from src.train import DB_PATH, MLFLOW_URI, PRED_DIR, load_split


def latest_run_id(run_name):
    runs = mlflow.search_runs(
        experiment_names=["sepsis-early-warning"],
        filter_string=f"tags.`mlflow.runName` = '{run_name}' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise SystemExit(f"No finished run named {run_name!r}")
    return runs.iloc[0]["run_id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--split", choices=["val", "test", "external"], required=True)
    args = ap.parse_args()

    mlflow.set_tracking_uri(MLFLOW_URI)
    run_id = latest_run_id(args.run_name)
    model = mlflow.lightgbm.load_model(f"runs:/{run_id}/model")
    feats = mlflow.artifacts.load_text(f"runs:/{run_id}/features.txt").split("\n")

    con = duckdb.connect(str(DB_PATH), read_only=True)
    X, y, meta = load_split(con, args.split, feats)
    con.close()

    p = model.predict_proba(X)[:, 1]
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out = PRED_DIR / f"{args.run_name}_{args.split}.parquet"
    meta.assign(y=y, p=p).to_parquet(out, index=False)
    print(f"Saved {len(p):,} predictions from run {run_id} to {out}")


if __name__ == "__main__":
    main()