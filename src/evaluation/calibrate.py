"""
Probability recalibration (Platt scaling) for all models.

Why: every ML model was trained with positive-class up-weighting
(XGBoost scale_pos_weight = N_neg/N_pos ~ 37.9; RNN BCE pos_weight, same ratio).
This helps ranking but inflates the output scale, so raw scores are not risks
(e.g. Temporal XGBoost's F1-optimal threshold is 0.775 at a 3% event rate).

Method: Platt scaling fitted on the VALIDATION split only, applied unchanged to test:
    p_cal = sigmoid(a + b * logit(p_raw))          (ML models)
    p_cal = sigmoid(a + b * NEWS2_score)           (NEWS2 -> risk mapping)
Two parameters per model, chosen over isotonic regression because the validation
split holds only 14 deteriorating stays (isotonic would overfit). The mapping is
strictly increasing, so AUROC / AUPRC / any threshold-based operating point are
unchanged; only the probability scale (calibration, Brier) changes. The ensemble is
calibrated on its own averaged raw output, so its ranking is also unchanged.

Reported on the test split (primary cohort and sepsis subgroup), raw vs calibrated:
  mean predicted risk vs observed event rate, calibration-in-the-large (intercept with
  logit(p) as offset; ideal 0), calibration slope (ideal 1), expected calibration
  error (10 equal-frequency bins), Brier score with patient-cluster bootstrap 95% CI,
  and Brier skill score vs the constant-prevalence predictor.

Outputs:
  results/calibration_parameters.json
  results/calibration_results.csv
  results/calibrated_test_predictions.parquet
  results/figures/calibration_reliability.png

Usage: python -m src.evaluation.calibrate
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from src.config import FIGURES_DIR, RESULTS_DIR
from src.evaluation.metrics import bootstrap_metric

EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _feature(model: str, score) -> np.ndarray:
    """NEWS2 is an integer score, calibrated on its own scale; ML outputs on the logit scale."""
    s = np.asarray(score, dtype=float)
    return s if model == "NEWS2" else _logit(s)


def _logistic_1d(x, y, offset=None, fit_slope: bool = True, n_iter: int = 100) -> tuple[float, float]:
    """Unpenalised maximum-likelihood logistic regression y ~ a + b*x (+ offset), by Newton-Raphson.
    Implemented directly so results do not depend on scikit-learn's penalty API/version."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    off = np.zeros_like(x) if offset is None else np.asarray(offset, dtype=float)
    X = np.column_stack([np.ones_like(x), x]) if fit_slope else np.ones((len(x), 1))
    beta = np.zeros(X.shape[1])
    if fit_slope:  # sensible start: intercept at the base rate
        beta[0] = np.log(max(y.mean(), EPS) / max(1 - y.mean(), EPS))
    for _ in range(n_iter):
        mu = _sigmoid(X @ beta + off)
        w = mu * (1 - mu)
        step = np.linalg.solve(X.T @ (X * w[:, None]) + 1e-10 * np.eye(X.shape[1]), X.T @ (y - mu))
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return (float(beta[0]), float(beta[1])) if fit_slope else (float(beta[0]), 1.0)


def fit_platt(model: str, score_val, y_val) -> dict:
    a, b = _logistic_1d(_feature(model, score_val), y_val)
    return {"a": a, "b": b}


def apply_platt(model: str, score, params: dict) -> np.ndarray:
    return _sigmoid(params["a"] + params["b"] * _feature(model, score))


def calibration_slope_intercept(y, p) -> tuple[float, float]:
    """Calibration slope (logistic refit on logit p) and calibration-in-the-large (offset model)."""
    lp = _logit(p)
    _, slope = _logistic_1d(lp, y)
    citl, _ = _logistic_1d(np.zeros_like(lp), y, offset=lp, fit_slope=False)
    return slope, citl


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    return float(sum(len(b) / len(p) * abs(p[b].mean() - y[b].mean()) for b in bins if len(b)))


def reliability_points(y, p, n_bins: int = 10):
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    return (np.array([p[b].mean() for b in bins]), np.array([np.asarray(y)[b].mean() for b in bins]))


def calibrate_all(df: pd.DataFrame, models: list[str]):
    """Fit Platt on val rows of ``df``, add ``<model>_cal`` columns for every row, return params."""
    val = df[df["split"] == "val"]
    params = {}
    for m in models:
        params[m] = fit_platt(m, val[m], val["label"])
        df[f"{m}_cal"] = apply_platt(m, df[m], params[m])
    return df, params


def main():
    from src.evaluation.evaluate_all_models import MODEL_ORDER, _load_predictions, _thresholds

    df, _, _ = _load_predictions()
    df, params = calibrate_all(df, MODEL_ORDER)
    thresholds = _thresholds(df)

    # Translate operating thresholds into calibrated-risk terms
    for m in MODEL_ORDER:
        params[m]["f1_opt_threshold_raw"] = thresholds[m]["f1_opt"]
        params[m]["f1_opt_threshold_as_calibrated_risk"] = float(
            apply_platt(m, [thresholds[m]["f1_opt"]], params[m])[0])
    params["XGB_temporal"]["screening_threshold_0.55_as_calibrated_risk"] = float(
        apply_platt("XGB_temporal", [0.55], params["XGB_temporal"])[0])
    params["_method"] = "Platt scaling fitted on validation split; ML models on logit(p), NEWS2 on raw score"
    with open(RESULTS_DIR / "calibration_parameters.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    test = df[df["split"] == "test"].copy()
    cohorts = {"Primary_Cohort": test, "Sepsis_Subgroup": test[test["is_sepsis"] == True]}
    rows = []
    for cname, sub in cohorts.items():
        y = sub["label"].to_numpy(int)
        pids = sub["patientunitstayid"].to_numpy()
        null_brier = float(y.mean() * (1 - y.mean()))
        for m in MODEL_ORDER:
            for version, col in [("raw", m), ("platt", f"{m}_cal")]:
                if m == "NEWS2" and version == "raw":
                    continue  # integer score, not a probability
                p = sub[col].to_numpy(float)
                slope, citl = calibration_slope_intercept(y, p)
                b = bootstrap_metric(y, p, pids, brier_score_loss, n_bootstraps=1000)
                rows.append({
                    "cohort": cname, "model": m, "probabilities": version,
                    "mean_predicted_risk": round(float(p.mean()), 4),
                    "observed_event_rate": round(float(y.mean()), 4),
                    "calibration_in_the_large": round(citl, 3),
                    "calibration_slope": round(slope, 3),
                    "ece_10bin": round(expected_calibration_error(y, p), 4),
                    "brier": round(b["point"], 4),
                    "brier_ci_lower": round(b["ci_lower"], 4),
                    "brier_ci_upper": round(b["ci_upper"], 4),
                    "brier_null_model": round(null_brier, 4),
                    "brier_skill_score": round(1 - b["point"] / null_brier, 3),
                })
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS_DIR / "calibration_results.csv", index=False)

    keep = ["patientunitstayid", "hour", "label", "is_sepsis"] + [f"{m}_cal" for m in MODEL_ORDER]
    test[keep].to_parquet(RESULTS_DIR / "calibrated_test_predictions.parquet", index=False)

    _plot(cohorts["Primary_Cohort"], MODEL_ORDER)
    pd.set_option("display.width", 220)
    print(res.to_string(index=False))
    print(json.dumps({k: v for k, v in params.items() if not k.startswith("_")}, indent=1))
    return res, params


def _plot(sub: pd.DataFrame, models: list[str]):
    from src.evaluation.evaluate_all_models import COLORS

    y = sub["label"].to_numpy(int)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for m in models:
        if m != "NEWS2":
            px, py = reliability_points(y, sub[m].to_numpy(float))
            axes[0].plot(px, py, "o-", color=COLORS[m], ms=4, lw=1.3, label=m)
        cx, cy = reliability_points(y, sub[f"{m}_cal"].to_numpy(float))
        axes[1].plot(cx, cy, "o-", color=COLORS[m], ms=4, lw=1.3, label=m)
    axes[0].plot([0, 1], [0, 1], ":", color="black", lw=0.8)
    axes[0].set(xlim=(0, 1), ylim=(0, 0.3), xlabel="Mean predicted 'probability' (raw, class-weighted)",
                ylabel="Observed 24 h deterioration rate", title="Before: raw model outputs")
    lim = 0.12
    axes[1].plot([0, lim], [0, lim], ":", color="black", lw=0.8, label="Perfect calibration")
    axes[1].set(xlim=(0, lim), ylim=(0, lim), xlabel="Mean predicted risk (Platt-calibrated on validation)",
                ylabel="Observed 24 h deterioration rate", title="After: Platt scaling")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Reliability diagrams, held-out primary test set (10 equal-frequency bins)",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration_reliability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
