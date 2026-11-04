"""Evaluation helpers: bootstrap confidence intervals, simulated alerting, lead time, calibration."""
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

SILENCE_HOURS = 6     # after an alert, no new alert for this patient for 6 h
EARLY_WINDOW = 12     # an alert within 12 h before onset counts as early warning


def add_onset(preds, cohort):
    """Attach septic status and onset hour; flag hours before onset (all hours for non-septic)."""
    df = preds.merge(cohort[["patient_id", "septic", "onset_iculos"]], on="patient_id", how="left")
    df["pre_onset"] = (df["septic"] == 0) | (df["ICULOS"] < df["onset_iculos"])
    return df


def bootstrap_metrics(y, p, patient_ids, n_boot=500, seed=0):
    """AUROC and AUPRC with 95% CIs, resampling patients (not hours).

    Resampling a patient k times is the same as giving their rows weight k,
    which is much faster than copying data.
    """
    y, p = np.asarray(y), np.asarray(p)
    codes, uniq = pd.factorize(np.asarray(patient_ids))
    n_pat = len(uniq)
    rng = np.random.default_rng(seed)
    point = {"auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p)}
    samples = {k: [] for k in point}
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, n_pat, n_pat), minlength=n_pat)[codes]
        m = w > 0
        if y[m].min() == y[m].max():          # need both classes
            continue
        samples["auroc"].append(roc_auc_score(y[m], p[m], sample_weight=w[m]))
        samples["auprc"].append(average_precision_score(y[m], p[m], sample_weight=w[m]))
    return {k: (point[k], np.percentile(v, 2.5), np.percentile(v, 97.5)) for k, v in samples.items()}


def simulate_alerts(df, threshold, silence_hours=SILENCE_HOURS):
    """Fire an alert when risk >= threshold, then stay silent for `silence_hours` for that patient."""
    rows = []
    for pid, g in df.sort_values(["patient_id", "ICULOS"]).groupby("patient_id", sort=False):
        last = -np.inf
        for t, prob in zip(g["ICULOS"].to_numpy(), g["p"].to_numpy()):
            if prob >= threshold and t - last >= silence_hours:
                rows.append((pid, t))
                last = t
    return pd.DataFrame(rows, columns=["patient_id", "alert_iculos"])


def alert_metrics(df, threshold, silence_hours=SILENCE_HOURS, window=EARLY_WINDOW):
    """Patient- and alert-level metrics at one threshold."""
    pts = df.drop_duplicates("patient_id")[["patient_id", "septic", "onset_iculos"]]
    a = simulate_alerts(df, threshold, silence_hours).merge(pts, on="patient_id")

    in_window = (a["septic"] == 1) & (a["alert_iculos"] >= a["onset_iculos"] - window)
    early = a[in_window & (a["alert_iculos"] < a["onset_iculos"])]
    first_early = early.groupby("patient_id")["alert_iculos"].min()
    lead = pts.set_index("patient_id").loc[first_early.index, "onset_iculos"] - first_early

    n_septic = int((pts["septic"] == 1).sum())
    n_nonseptic = int((pts["septic"] == 0).sum())
    patient_days = len(df) / 24
    return {
        "threshold": threshold,
        "early_sensitivity": len(first_early) / n_septic,
        "median_lead_h": float(lead.median()) if len(lead) else np.nan,
        "alerts_per_100_patient_days": 100 * len(a) / patient_days,
        "alert_ppv": float(in_window.mean()) if len(a) else np.nan,
        "nonseptic_patients_alerted": a.loc[a["septic"] == 0, "patient_id"].nunique() / n_nonseptic,
    }


def threshold_table(df, thresholds, **kwargs):
    return pd.DataFrame([alert_metrics(df, t, **kwargs) for t in thresholds])


def calibration(y, p, n_bins=10):
    """Calibration curve (quantile bins) and Brier score."""
    observed, predicted = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted": predicted, "observed_rate": observed}), brier_score_loss(y, p)