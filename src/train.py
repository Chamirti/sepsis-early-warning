"""Train a model on Hospital A's train split; evaluate on validation; log to MLflow.

Run from the project root, e.g.:
  python -m src.train --model logreg
  python -m src.train --model lgbm
  python -m src.train --model lgbm --with-iculos         (ablation)
  python -m src.train --model lgbm --without-ordering    (ablation)
  python -m src.train --model lgbm --class-weight        (weighting comparison)
"""
import argparse
from pathlib import Path

import duckdb
import lightgbm as lgb
import mlflow
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import META, ORDERING_FEATURES, TIME_FEATURES

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "sepsis.duckdb"
MLFLOW_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
PRED_DIR = ROOT / "data" / "predictions"


def select_features(all_columns, with_iculos=False, without_ordering=False):
    """Model inputs: every non-meta column, adjusted for the ablation options."""
    feats = [c for c in all_columns if c not in META]          # META includes ICULOS
    if with_iculos:
        feats += TIME_FEATURES
    if without_ordering:
        feats = [c for c in feats if c not in ORDERING_FEATURES]
    return feats


def load_split(con, split, feats):
    cols = list(dict.fromkeys(["patient_id", "ICULOS", "SepsisLabel"] + feats))
    df = con.execute(f"SELECT {', '.join(cols)} FROM features WHERE split = ?", [split]).df()
    X = df[feats].astype("float32")        # float32 halves memory use
    y = df["SepsisLabel"].to_numpy()
    return X, y, df[["patient_id", "ICULOS"]]


def train_logreg(X_tr, y_tr):
    params = {"C": 0.1, "imputer": "median", "max_iter": 2000}
    model = make_pipeline(
        SimpleImputer(strategy="median"),   # logistic regression can't handle missing values
        StandardScaler(),                   
        LogisticRegression(C=params["C"], max_iter=params["max_iter"]),
    )
    model.fit(X_tr, y_tr)
    return model, params


def train_lgbm(X_tr, y_tr, X_va, y_va, scale_pos_weight=1.0):
    params = {
        "n_estimators": 3000, "learning_rate": 0.03, "num_leaves": 63,
        "min_child_samples": 200, "subsample": 0.8, "subsample_freq": 1,
        "colsample_bytree": 0.7, "reg_lambda": 1.0,
        "scale_pos_weight": scale_pos_weight, "metric": "average_precision",
        "random_state": 42, "n_jobs": -1, "verbose": -1,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )
    params["best_iteration"] = model.best_iteration_
    return model, params


def evaluate(y, p):
    return {
        "auroc": roc_auc_score(y, p),
        "auprc": average_precision_score(y, p),
        "positive_rate": float(y.mean()),     # the AUPRC of a random model
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["logreg", "lgbm"], required=True)
    ap.add_argument("--with-iculos", action="store_true")
    ap.add_argument("--without-ordering", action="store_true")
    ap.add_argument("--class-weight", action="store_true")
    args = ap.parse_args()

    con = duckdb.connect(str(DB_PATH), read_only=True)
    all_cols = [r[0] for r in con.execute("DESCRIBE features").fetchall()]
    feats = select_features(all_cols, args.with_iculos, args.without_ordering)
    X_tr, y_tr, _ = load_split(con, "train", feats)
    X_va, y_va, meta_va = load_split(con, "val", feats)
    con.close()

    run_name = (args.model
                + ("+iculos" if args.with_iculos else "")
                + ("-ordering" if args.without_ordering else "")
                + ("+weighted" if args.class_weight else ""))

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("sepsis-early-warning")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model": args.model, "with_iculos": args.with_iculos,
            "without_ordering": args.without_ordering, "class_weight": args.class_weight,
            "n_features": len(feats), "n_train_rows": len(y_tr),
        })

        if args.model == "logreg":
            model, params = train_logreg(X_tr, y_tr)
            mlflow.sklearn.log_model(model, name="model", skops_trusted_types=["numpy.dtype"])
        else:
            spw = (len(y_tr) - y_tr.sum()) / y_tr.sum() if args.class_weight else 1.0
            model, params = train_lgbm(X_tr, y_tr, X_va, y_va, spw)
            mlflow.lightgbm.log_model(model, name="model")

        p_va = model.predict_proba(X_va)[:, 1]
        metrics = evaluate(y_va, p_va)
        mlflow.log_params({f"hp_{k}": v for k, v in params.items()})
        mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
        mlflow.log_text("\n".join(feats), "features.txt")

        # Save validation predictions for Day 3's clinical evaluation
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        meta_va.assign(y=y_va, p=p_va).to_parquet(PRED_DIR / f"{run_name}_val.parquet", index=False)

        print(run_name, {k: round(v, 4) for k, v in metrics.items()})


if __name__ == "__main__":
    main()