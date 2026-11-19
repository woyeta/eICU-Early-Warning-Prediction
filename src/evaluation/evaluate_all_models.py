"""
Unified held-out evaluation of all five models:
NEWS2, Static XGBoost, Temporal XGBoost, LSTM and GRU (+ a pre-specified
Temporal-XGBoost/RNN probability-averaging ensemble).

What this adds over src/baselines/compare.py and src/evaluation/visualize.py:
1. Sequence models on exactly the same test rows, thresholds chosen on validation.
2. Paired patient-level bootstrap CIs and p-values for metric *differences*
   (e.g. AUPRC(Temporal XGB) - AUPRC(Static XGB)), instead of eyeballing overlap
   of two marginal CIs.
3. A matched-sensitivity operating point: every model's threshold is set on the
   validation split so that its hour-level sensitivity equals that of NEWS2 >= 5.
   False alarms and lead times are then compared at equal sensitivity, which is
   the fair way to compare alarm burden (the F1-optimal points of the XGBoost
   models alert on far fewer patients than NEWS2, so their lower alarm burden at
   those points partly reflects lower sensitivity).
4. Brier score before and after Platt recalibration (fitted on validation; see
   src/evaluation/calibrate.py). Recalibration is monotone, so every ranking metric and
   operating point here is unchanged by it.

Outputs:
  results/all_models_test_results.csv
  results/paired_bootstrap_differences.csv
  results/operating_points_comparison.csv
  results/figures/all_models_test_roc_pr_primary.png
  results/figures/all_models_test_roc_pr_sepsis.png
  results/figures/paired_difference_forest.png
  results/alarm_burden_equal_sensitivity.csv, results/figures/alarm_burden_vs_sensitivity.png
  results/all_models_thresholds.json

Usage: python -m src.evaluation.evaluate_all_models
"""

from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.baselines.news2_scorer import apply_news2_to_dataset
from src.config import (
    ALL_FEATURES,
    DATA_PROCESSED,
    FIGURES_DIR,
    MODELS_DIR,
    NEWS2_THRESHOLD,
    RESULTS_DIR,
    SNAPSHOT_FEATURES,
)
from src.evaluation.lead_time import summarize_lead_times
from src.evaluation.metrics import bootstrap_metric, paired_bootstrap_diff
from src.evaluation.visualize import compute_alarm_burden

N_BOOT = 1000
MODEL_ORDER = ["NEWS2", "XGB_static", "XGB_temporal", "LSTM", "GRU", "Ensemble_XGBt+RNN"]
COLORS = {
    "NEWS2": "#7f7f7f",
    "XGB_static": "#2ca02c",
    "XGB_temporal": "#1f4e9c",
    "LSTM": "#9467bd",
    "GRU": "#d9822b",
    "Ensemble_XGBt+RNN": "#c0392b",
}


def _load_predictions() -> tuple[pd.DataFrame, dict, str]:
    v1 = pd.read_parquet(DATA_PROCESSED / "v1.parquet")
    df = v1[v1["split"].isin(["val", "test"])].copy()
    df = apply_news2_to_dataset(df)
    df["NEWS2"] = df["news2_score"].astype(float)
    df["XGB_static"] = joblib.load(MODELS_DIR / "xgb_static.pkl").predict_proba(df[SNAPSHOT_FEATURES])[:, 1]
    df["XGB_temporal"] = joblib.load(MODELS_DIR / "xgb_temporal.pkl").predict_proba(df[ALL_FEATURES])[:, 1]

    seq = pd.read_parquet(RESULTS_DIR / "sequence_predictions.parquet")
    seq_wide = seq.pivot_table(index=["patientunitstayid", "hour", "split"], columns="model",
                               values="pred_prob").reset_index()
    df = df.merge(seq_wide, on=["patientunitstayid", "hour", "split"], how="left", validate="one_to_one")
    assert df[["LSTM", "GRU"]].notna().all().all(), "Sequence predictions missing for some rows"

    with open(RESULTS_DIR / "sequence_params.json", encoding="utf-8") as f:
        seq_params = json.load(f)

    # Pre-specified ensemble: Temporal XGBoost + whichever RNN had higher *validation* AUPRC
    val = df[df["split"] == "val"]
    best_rnn = max(["LSTM", "GRU"], key=lambda m: average_precision_score(val["label"], val[m]))
    df["Ensemble_XGBt+RNN"] = 0.5 * (df["XGB_temporal"] + df[best_rnn])
    return df, seq_params, best_rnn


def _f1_opt_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(prec[:-1] + rec[:-1] > 0, 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1]), 0)
    mask = (thr >= 0.05) & (thr <= 0.95)
    idx = int(np.argmax(np.where(mask, f1, -1)))
    return float(thr[idx])


def _thresholds(df: pd.DataFrame) -> dict:
    """Validation-derived thresholds for the F1-optimal and matched-sensitivity operating points."""
    val = df[df["split"] == "val"]
    y = val["label"].to_numpy(int)
    news2_sens = float(((val["NEWS2"] >= NEWS2_THRESHOLD) & (y == 1)).sum() / max((y == 1).sum(), 1))
    out = {}
    for m in MODEL_ORDER:
        if m == "NEWS2":
            out[m] = {"f1_opt": float(NEWS2_THRESHOLD), "matched_sens": float(NEWS2_THRESHOLD)}
            continue
        p = val[m].to_numpy(float)
        if m == "XGB_static":
            f1t = json.load(open(RESULTS_DIR / "xgb_static_params.json"))["optimal_f1_threshold"]
        elif m == "XGB_temporal":
            f1t = json.load(open(RESULTS_DIR / "xgb_temporal_params.json"))["optimal_f1_threshold"]
        else:
            f1t = _f1_opt_threshold(y, p)
        matched = float(np.quantile(p[y == 1], 1.0 - news2_sens))
        out[m] = {"f1_opt": float(f1t), "matched_sens": matched}
    out["_news2_val_hour_sensitivity"] = news2_sens
    return out


def evaluate(df: pd.DataFrame, thresholds: dict):
    test = df[df["split"] == "test"].copy()
    cohorts = {"Primary_Cohort": test, "Sepsis_Subgroup": test[test["is_sepsis"] == True].copy()}

    metric_rows, diff_rows, op_rows = [], [], []
    for cname, sub in cohorts.items():
        y = sub["label"].to_numpy(int)
        pids = sub["patientunitstayid"].to_numpy()
        for m in MODEL_ORDER:
            p = sub[m].to_numpy(float)
            auroc = bootstrap_metric(y, p, pids, roc_auc_score, n_bootstraps=N_BOOT)
            auprc = bootstrap_metric(y, p, pids, average_precision_score, n_bootstraps=N_BOOT)
            thr = thresholds[m]["f1_opt"]
            metric_rows.append({
                "cohort": cname, "model": m,
                "auroc": round(auroc["point"], 4), "auroc_ci_lower": round(auroc["ci_lower"], 4),
                "auroc_ci_upper": round(auroc["ci_upper"], 4),
                "auprc": round(auprc["point"], 4), "auprc_ci_lower": round(auprc["ci_lower"], 4),
                "auprc_ci_upper": round(auprc["ci_upper"], 4),
                "f1_at_val_f1opt_threshold": round(float(f1_score(y, (p >= thr).astype(int), zero_division=0)), 4),
                "threshold_f1opt": round(thr, 4),
                "brier_raw": round(float(brier_score_loss(y, p)), 4) if m != "NEWS2" else np.nan,
                "brier_platt_calibrated": round(float(brier_score_loss(y, sub[f"{m}_cal"])), 4),
                "prevalence": round(float(y.mean()), 4),
                "brier_null_model": round(float(y.mean() * (1 - y.mean())), 4),
            })

            # operating points (lead time + alarm burden)
            for op in ["f1_opt", "matched_sens"]:
                t = thresholds[m][op]
                ev = sub[["patientunitstayid", "hour", "label", "event_hour"]].copy()
                ev["pred_prob"] = p
                lt = summarize_lead_times(ev, threshold=t, max_lookback_hours=48, inclusive_threshold=True)
                hour_sens = float(((p >= t) & (y == 1)).sum() / max((y == 1).sum(), 1))
                op_rows.append({
                    "cohort": cname, "model": m, "operating_point": op, "threshold": round(t, 4),
                    "hour_level_sensitivity_pct": round(100 * hour_sens, 1),
                    "false_alarms_per_patient_day": round(compute_alarm_burden(ev, t, inclusive_threshold=True), 2),
                    "patients_alerted_48h": f"{lt['n_alerted']}/{lt['n_deteriorating']}",
                    "patient_sensitivity_48h_pct": lt["patient_sensitivity_pct"],
                    "median_lead_time_48h_h": lt["median_lead_time_h"],
                    "iqr_lead_time_48h_h": f"{lt['iqr_25_h']}-{lt['iqr_75_h']}",
                    "pct_alerted_ge_6h": lt["pct_det_alerted_ge_6h"],
                })

        # paired differences
        comparisons = [("XGB_temporal", "XGB_static"), ("XGB_temporal", "NEWS2"), ("XGB_static", "NEWS2"),
                       ("LSTM", "XGB_temporal"), ("GRU", "XGB_temporal"), ("LSTM", "NEWS2"), ("GRU", "NEWS2"),
                       ("Ensemble_XGBt+RNN", "XGB_temporal")]
        for a, b in comparisons:
            for mname, fn in [("AUROC", roc_auc_score), ("AUPRC", average_precision_score)]:
                r = paired_bootstrap_diff(y, sub[a], sub[b], pids, fn, n_bootstraps=N_BOOT)
                diff_rows.append({"cohort": cname, "comparison": f"{a} - {b}", "metric": mname,
                                  "diff": round(r["diff"], 4), "ci_lower": round(r["ci_lower"], 4),
                                  "ci_upper": round(r["ci_upper"], 4), "p_value": round(r["p_value"], 3)})

    return pd.DataFrame(metric_rows), pd.DataFrame(diff_rows), pd.DataFrame(op_rows), cohorts


def plot_roc_pr(cohorts: dict):
    for cname, sub in cohorts.items():
        y = sub["label"].to_numpy(int)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for m in MODEL_ORDER:
            p = sub[m].to_numpy(float)
            fpr, tpr, _ = roc_curve(y, p)
            pr, rc, _ = precision_recall_curve(y, p)
            ls = "--" if m.startswith("Ensemble") else "-"
            axes[0].plot(fpr, tpr, ls, color=COLORS[m], lw=1.6, label=f"{m} (AUROC {roc_auc_score(y, p):.3f})")
            axes[1].plot(rc, pr, ls, color=COLORS[m], lw=1.6,
                         label=f"{m} (AUPRC {average_precision_score(y, p):.3f})")
        axes[0].plot([0, 1], [0, 1], ":", color="black", lw=0.8)
        axes[1].axhline(y.mean(), ls=":", color="black", lw=0.8, label=f"Prevalence {y.mean():.3f}")
        axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
        axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall")
        for ax in axes:
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7.5, loc="lower right" if ax is axes[0] else "upper right")
        title = "Primary Adult ICU Cohort" if cname == "Primary_Cohort" else "Sepsis Subgroup"
        fig.suptitle(f"Held-out test set — all models — {title}", fontweight="bold")
        fig.tight_layout()
        tag = "primary" if cname == "Primary_Cohort" else "sepsis"
        fig.savefig(FIGURES_DIR / f"all_models_test_roc_pr_{tag}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def alarm_curve(cohorts: dict, thresholds: dict) -> pd.DataFrame:
    """
    Descriptive alarm-burden curve on the test split: false alarms per patient-day as a
    function of hour-level sensitivity, for every model. Reading every model off at the
    hour-level sensitivity NEWS2 >= 5 achieves on the same test hours gives an
    equal-sensitivity alarm comparison (thresholds here are test-derived, so this is a
    descriptive comparison of operating characteristics, not a deployable operating point).
    """
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (cname, sub) in zip(axes, cohorts.items()):
        y = sub["label"].to_numpy(int)
        pdays = len(sub) / 24.0
        news2 = sub["NEWS2"].to_numpy(float)
        target = float(((news2 >= NEWS2_THRESHOLD) & (y == 1)).sum() / (y == 1).sum())
        for m in MODEL_ORDER:
            p = sub[m].to_numpy(float)
            fpr, tpr, thr = roc_curve(y, p)
            fa_day = fpr * (y == 0).sum() / pdays
            ls = "--" if m.startswith("Ensemble") else "-"
            ax.plot(100 * tpr, fa_day, ls, color=COLORS[m], lw=1.5, label=m)
            fa_at = float(np.interp(target, tpr, fa_day))
            rows.append({"cohort": cname, "model": m, "target_hour_sensitivity_pct": round(100 * target, 1),
                         "false_alarms_per_patient_day_at_target": round(fa_at, 2)})
        ax.axvline(100 * target, ls=":", color="black", lw=0.8)
        nf = float(((news2 >= NEWS2_THRESHOLD) & (y == 0)).sum() / pdays)
        ax.plot([100 * target], [nf], "s", color="black", label="NEWS2 ≥ 5")
        ax.set(xlabel="Hour-level sensitivity (%)", ylabel="False alarms per patient-day",
               title="Primary cohort" if cname == "Primary_Cohort" else "Sepsis subgroup", ylim=(0, 12))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5)
    fig.suptitle("Alarm burden vs sensitivity on the held-out test set (dotted line = NEWS2 ≥ 5 sensitivity)",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "alarm_burden_vs_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def plot_forest(diff_df: pd.DataFrame):
    sub = diff_df[diff_df["cohort"] == "Primary_Cohort"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, metric in zip(axes, ["AUROC", "AUPRC"]):
        d = sub[sub["metric"] == metric].reset_index(drop=True)
        ypos = np.arange(len(d))[::-1]
        ax.errorbar(d["diff"], ypos, xerr=[d["diff"] - d["ci_lower"], d["ci_upper"] - d["diff"]],
                    fmt="o", color="#1f4e9c", ecolor="#1f4e9c", capsize=3)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(ypos)
        ax.set_yticklabels(d["comparison"], fontsize=8)
        ax.set_xlabel(f"Δ{metric} (paired patient bootstrap, 95% CI)")
        ax.grid(alpha=0.3, axis="x")
        for yp, (_, r) in zip(ypos, d.iterrows()):
            ax.annotate(f"p={r['p_value']:.2f}", (r["ci_upper"], yp), xytext=(4, -3),
                        textcoords="offset points", fontsize=7)
    fig.suptitle("Paired model differences on the held-out primary test set (103 stays, 13 deteriorating)",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "paired_difference_forest.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    from src.evaluation.calibrate import calibrate_all

    df, seq_params, best_rnn = _load_predictions()
    df, platt = calibrate_all(df, MODEL_ORDER)  # Platt scaling fitted on validation (see calibrate.py)
    thresholds = _thresholds(df)
    for m in MODEL_ORDER:
        for op in ["f1_opt", "matched_sens"]:
            thresholds[m][f"{op}_as_calibrated_risk"] = float(
                1 / (1 + np.exp(-(platt[m]["a"] + platt[m]["b"] * (thresholds[m][op] if m == "NEWS2"
                     else np.log(np.clip(thresholds[m][op], 1e-6, 1 - 1e-6) / (1 - np.clip(thresholds[m][op], 1e-6, 1 - 1e-6))))))))
    metrics_df, diff_df, op_df, cohorts = evaluate(df, thresholds)

    metrics_df.to_csv(RESULTS_DIR / "all_models_test_results.csv", index=False)
    diff_df.to_csv(RESULTS_DIR / "paired_bootstrap_differences.csv", index=False)
    op_df.to_csv(RESULTS_DIR / "operating_points_comparison.csv", index=False)
    with open(RESULTS_DIR / "all_models_thresholds.json", "w", encoding="utf-8") as f:
        json.dump({**thresholds, "ensemble_rnn_member": best_rnn,
                   "sequence_windows": {k: v["window_hours"] for k, v in seq_params.items()}}, f, indent=2)
    plot_roc_pr(cohorts)
    plot_forest(diff_df)
    eq_df = alarm_curve(cohorts, thresholds)
    eq_df.to_csv(RESULTS_DIR / "alarm_burden_equal_sensitivity.csv", index=False)
    print(eq_df.to_string(index=False))

    pd.set_option("display.width", 200)
    print(metrics_df.to_string(index=False))
    print(diff_df.to_string(index=False))
    print(op_df.to_string(index=False))


if __name__ == "__main__":
    main()
