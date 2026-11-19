"""
Operational alarm burden and clinical evaluation visualization utilities.

Computes false alarms per patient-day across at-risk hours and plots warning
lead-time distributions (histogram + boxplot) and individual "Hero" patient
deterioration timelines.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_alarm_burden(
    predictions_df: pd.DataFrame,
    threshold: float,
    inclusive_threshold: bool = False,
) -> float:
    """Compute false alarms per patient-day across at-risk hours."""
    if inclusive_threshold:
        triggered = predictions_df["pred_prob"] >= threshold
    else:
        triggered = predictions_df["pred_prob"] > threshold
    false_alarms = int((triggered & (predictions_df["label"] == 0)).sum())
    patient_days = len(predictions_df) / 24.0
    return float(false_alarms / max(patient_days, 1e-6))


def plot_lead_time_distribution(
    lead_times_df: pd.DataFrame,
    cohort_name: str,
    out_path=None,
):
    """Plot combined histogram and boxplot of warning lead times (in hours)."""
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.5, 5.2),
        gridspec_kw={"height_ratios": [3.2, 1.0]},
        sharex=True,
    )
    lts = lead_times_df["lead_time_hours"].dropna().to_numpy(dtype=float)

    if len(lts) > 0:
        med = float(np.median(lts))
        q25 = float(np.percentile(lts, 25))
        q75 = float(np.percentile(lts, 75))
        axes[0].hist(
            lts,
            bins=min(12, max(5, len(lts))),
            color="#2b5c8f",
            edgecolor="white",
            alpha=0.85,
        )
        axes[0].axvline(
            med,
            color="#e41a1c",
            ls="--",
            lw=2.0,
            label=f"Median Lead Time: {med:.1f}h (IQR: {q25:.1f}–{q75:.1f}h)",
        )
        axes[0].axvline(
            6.0,
            color="#2ca25f",
            ls=":",
            lw=1.6,
            label="6h Clinical Action Window",
        )
        axes[0].legend(frameon=True, fontsize=9.5)

        axes[1].boxplot(
            lts,
            vert=False,
            widths=0.55,
            patch_artist=True,
            boxprops=dict(facecolor="#a6bddb", edgecolor="#2b5c8f"),
            medianprops=dict(color="#e41a1c", lw=2.0),
        )
        axes[1].set_yticks([])

    axes[0].set_ylabel("Deteriorating Patients (N)", fontsize=10)
    axes[0].set_title(
        f"Warning Lead-Time Distribution Before Clinical Crash — {cohort_name}",
        fontsize=11.5,
        fontweight="bold",
    )
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].set_xlabel("Warning Lead Time Before True Event (Hours)", fontsize=10)
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_hero_patient_timeline(
    patient_df: pd.DataFrame,
    threshold: float,
    out_path=None,
    title_suffix: str = "",
):
    """
    Plot the 4-panel "Hero Chart" for a deteriorating ICU patient:
    Panel 1: Heart Rate (bpm) + 6h Rolling Mean
    Panel 2: Mean Arterial Pressure / Systolic BP (mmHg)
    Panel 3: Oxygen Saturation SpO2 (%)
    Panel 4: Temporal XGBoost Risk Score P(Deterioration in 24h) vs. Alert Threshold
    """
    sub = patient_df.sort_values("hour").copy()
    pid = int(sub["patientunitstayid"].iloc[0])
    ev_hour = float(sub["event_hour"].dropna().iloc[0])

    # Restrict display to [max(0, ev_hour - 48), ev_hour] for clear clinical narrative
    start_h = max(0.0, ev_hour - 48.0)
    sub = sub[sub["hour"] >= start_h].copy()

    alerts = sub[sub["pred_prob"] > threshold]
    first_alert_h = float(alerts["hour"].min()) if not alerts.empty else None
    lead_h = (ev_hour - first_alert_h) if first_alert_h is not None else None

    fig, axes = plt.subplots(4, 1, figsize=(10.5, 8.2), sharex=True)

    # Panel 1: Heart Rate
    axes[0].plot(sub["hour"], sub["heartrate"], color="#d95f02", marker="o", ms=3.5, lw=1.8, label="Hourly Heart Rate (bpm)")
    if "heartrate_roll_mean_6h" in sub.columns:
        axes[0].plot(sub["hour"], sub["heartrate_roll_mean_6h"], color="#7f2704", ls="--", lw=1.6, label="6h Rolling Mean HR")
    axes[0].set_ylabel("HR (bpm)", fontsize=9.5, fontweight="bold")
    axes[0].legend(loc="upper left", fontsize=8.5, frameon=True)
    axes[0].grid(alpha=0.3)

    # Panel 2: MAP & SBP
    axes[1].plot(sub["hour"], sub["systemicmean"], color="#2b5c8f", marker="s", ms=3.5, lw=1.8, label="Mean Arterial Pressure (MAP, mmHg)")
    axes[1].plot(sub["hour"], sub["systemicsystolic"], color="#6baed6", ls=":", lw=1.5, label="Systolic BP (SBP, mmHg)")
    axes[1].axhline(65, color="#e41a1c", ls=":", lw=1.2, label="MAP = 65 mmHg Hypotension Line")
    axes[1].set_ylabel("BP (mmHg)", fontsize=9.5, fontweight="bold")
    axes[1].legend(loc="upper left", fontsize=8.5, frameon=True)
    axes[1].grid(alpha=0.3)

    # Panel 3: SpO2 & Respiration
    axes[2].plot(sub["hour"], sub["spo2"], color="#2ca25f", marker="^", ms=3.5, lw=1.8, label="SpO₂ (%)")
    axes[2].set_ylabel("SpO₂ (%)", fontsize=9.5, fontweight="bold")
    axes[2].set_ylim(max(80, sub["spo2"].min() - 3), 101)
    axes[2].legend(loc="lower left", fontsize=8.5, frameon=True)
    axes[2].grid(alpha=0.3)

    # Panel 4: Temporal XGBoost Risk Trajectory
    axes[3].plot(
        sub["hour"],
        sub["pred_prob"],
        color="#cb181d",
        marker="o",
        ms=4.0,
        lw=2.3,
        label="Temporal XGBoost Risk Score P(Crash in 24h)",
    )
    axes[3].axhline(
        threshold,
        color="#252525",
        ls="--",
        lw=1.5,
        label=f"Decision Alert Threshold ({threshold:.2f})",
    )
    axes[3].fill_between(
        sub["hour"],
        threshold,
        sub["pred_prob"],
        where=(sub["pred_prob"] > threshold),
        color="#fcbba1",
        alpha=0.55,
        label="Active Early Warning Zone",
    )
    axes[3].set_ylabel("Risk Score", fontsize=9.5, fontweight="bold")
    axes[3].set_xlabel("Hours Since ICU Admission", fontsize=10, fontweight="bold")
    axes[3].set_ylim(0, 1.05)
    axes[3].legend(loc="lower left", fontsize=8.5, frameon=True)
    axes[3].grid(alpha=0.3)

    # Mark First Alert & Actual Event across all panels
    for ax in axes:
        ax.axvline(ev_hour, color="#99000d", lw=2.2, ls="-")
        if first_alert_h is not None:
            ax.axvline(first_alert_h, color="#d95f02", lw=1.8, ls="--")

    if first_alert_h is not None:
        axes[3].annotate(
            f"MODEL ALERT\n({lead_h:.0f}h before crash)",
            xy=(first_alert_h, threshold),
            xytext=(min(ev_hour - 14.0, first_alert_h + 2.0), max(0.18, threshold - 0.32)),
            arrowprops=dict(arrowstyle="->", color="#d95f02", lw=1.6),
            fontsize=8.8,
            fontweight="bold",
            color="#a63603",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff5eb", edgecolor="#d95f02"),
        )
    axes[3].annotate(
        f"ACTUAL EVENT\n(Hour {ev_hour:.0f})",
        xy=(ev_hour, 0.85),
        xytext=(ev_hour - 8.5, 0.85),
        arrowprops=dict(arrowstyle="->", color="#99000d", lw=1.6),
        fontsize=8.8,
        fontweight="bold",
        color="#99000d",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#fee0d2", edgecolor="#99000d"),
    )

    lead_str = f" — Caught {lead_h:.0f} Hours Early" if lead_h is not None else ""
    plt.suptitle(
        f"Hero Patient Trajectory (Stay {pid}){lead_str}{title_suffix}",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def run_lead_time_and_alarm_evaluation():
    """
    Execute the clinical evaluation on the held-out Test set:
    1. Compute warning lead times (48h pre-event window and full stay) across Primary
       and Sepsis cohorts for NEWS2, Static XGBoost, Temporal XGBoost (F1-Optimal 0.7753),
       and Temporal XGBoost (Clinical Screening 0.55).
    2. Compute false alarms per patient-day across all models and cohorts.
    3. Generate lead-time distribution plots and Hero patient timeline figures.
    """
    import json
    import joblib
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
    from src.evaluation.lead_time import compute_lead_times, summarize_lead_times

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    v1_df = pd.read_parquet(DATA_PROCESSED / "v1.parquet")
    test_df = apply_news2_to_dataset(v1_df[v1_df["split"] == "test"].copy())

    xgb_static = joblib.load(MODELS_DIR / "xgb_static.pkl")
    xgb_temporal = joblib.load(MODELS_DIR / "xgb_temporal.pkl")

    with open(RESULTS_DIR / "xgb_static_params.json", "r", encoding="utf-8") as f:
        static_thresh = float(json.load(f)["optimal_f1_threshold"])
    with open(RESULTS_DIR / "xgb_temporal_params.json", "r", encoding="utf-8") as f:
        temp_thresh = float(json.load(f)["optimal_f1_threshold"])

    test_df["pred_news2"] = test_df["news2_score"].astype(float)
    test_df["pred_static"] = xgb_static.predict_proba(test_df[SNAPSHOT_FEATURES])[:, 1]
    test_df["pred_temporal"] = xgb_temporal.predict_proba(test_df[ALL_FEATURES])[:, 1]

    model_specs = [
        ("NEWS2 (Threshold >= 5)", "pred_news2", float(NEWS2_THRESHOLD), True),
        ("XGB_static (F1-Optimal)", "pred_static", static_thresh, True),
        ("XGB_temporal (F1-Optimal)", "pred_temporal", temp_thresh, True),
        ("XGB_temporal (Clinical Screening)", "pred_temporal", 0.55, True),
    ]

    cohorts = [
        ("Primary_Cohort", "Primary Adult ICU Cohort", test_df),
        (
            "Sepsis_Subgroup",
            "Secondary Sepsis Subgroup",
            test_df[test_df["is_sepsis"] == True].copy(),
        ),
    ]

    summary_rows = []
    patient_rows = []
    alarm_rows = []

    for cohort_key, cohort_title, sub_df in cohorts:
        for model_label, col_name, thresh, inc in model_specs:
            df_eval = sub_df.copy()
            df_eval["pred_prob"] = df_eval[col_name]

            for window_name, max_lb in [("48h_pre_event", 48), ("full_stay", None)]:
                stats = summarize_lead_times(
                    df_eval,
                    threshold=thresh,
                    max_lookback_hours=max_lb,
                    inclusive_threshold=inc,
                )
                summary_rows.append(
                    {
                        "cohort": cohort_key,
                        "model": model_label,
                        "window": window_name,
                        "threshold": round(thresh, 4),
                        **stats,
                    }
                )
                lt_df = compute_lead_times(
                    df_eval,
                    threshold=thresh,
                    max_lookback_hours=max_lb,
                    inclusive_threshold=inc,
                )
                for r in lt_df.to_dict(orient="records"):
                    patient_rows.append(
                        {
                            "cohort": cohort_key,
                            "model": model_label,
                            "window": window_name,
                            "threshold": round(thresh, 4),
                            **r,
                        }
                    )

            fa_per_day = compute_alarm_burden(
                df_eval, threshold=thresh, inclusive_threshold=inc
            )
            stats_48 = summarize_lead_times(
                df_eval,
                threshold=thresh,
                max_lookback_hours=48,
                inclusive_threshold=inc,
            )
            stats_full = summarize_lead_times(
                df_eval,
                threshold=thresh,
                max_lookback_hours=None,
                inclusive_threshold=inc,
            )
            total_fa = int(
                ((df_eval["pred_prob"] >= thresh) & (df_eval["label"] == 0)).sum()
            )
            patient_days = round(len(df_eval) / 24.0, 2)
            alarm_rows.append(
                {
                    "cohort": cohort_key,
                    "model": model_label,
                    "threshold": round(thresh, 4),
                    "patient_days": patient_days,
                    "total_false_alarms": total_fa,
                    "false_alarms_per_patient_day": round(fa_per_day, 2),
                    "patient_sensitivity_48h_pct": stats_48["patient_sensitivity_pct"],
                    "median_lead_time_48h_h": stats_48["median_lead_time_h"],
                    "patient_sensitivity_full_pct": stats_full["patient_sensitivity_pct"],
                    "median_lead_time_full_h": stats_full["median_lead_time_h"],
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    patient_lt_df = pd.DataFrame(patient_rows)
    alarm_df = pd.DataFrame(alarm_rows)

    summary_df.to_csv(RESULTS_DIR / "lead_time_summary.csv", index=False)
    patient_lt_df.to_csv(RESULTS_DIR / "lead_time_per_patient.csv", index=False)
    alarm_df.to_csv(RESULTS_DIR / "alarm_burden_results.csv", index=False)

    # Generate Lead-Time Distribution plots on Primary & Sepsis Test cohorts
    df_temp_prim = test_df.copy()
    df_temp_prim["pred_prob"] = df_temp_prim["pred_temporal"]
    lt_prim = compute_lead_times(
        df_temp_prim, threshold=0.55, max_lookback_hours=48, inclusive_threshold=True
    )
    plot_lead_time_distribution(
        lt_prim,
        cohort_name="Primary Adult ICU Cohort (48h Pre-Event Window)",
        out_path=FIGURES_DIR / "lead_time_primary.png",
    )

    df_temp_sep = test_df[test_df["is_sepsis"] == True].copy()
    df_temp_sep["pred_prob"] = df_temp_sep["pred_temporal"]
    lt_sep = compute_lead_times(
        df_temp_sep, threshold=0.55, max_lookback_hours=48, inclusive_threshold=True
    )
    plot_lead_time_distribution(
        lt_sep,
        cohort_name="Secondary Sepsis Subgroup (48h Pre-Event Window)",
        out_path=FIGURES_DIR / "lead_time_sepsis.png",
    )

    # Generate Hero Patient Timeline charts
    hero_primary_df = df_temp_prim[df_temp_prim["patientunitstayid"] == 1795300].copy()
    plot_hero_patient_timeline(
        hero_primary_df,
        threshold=temp_thresh,
        out_path=FIGURES_DIR / "hero_patient_timeline.png",
        title_suffix=" (Sepsis / Vasopressor Crash)",
    )

    hero_second_df = df_temp_prim[df_temp_prim["patientunitstayid"] == 808117].copy()
    plot_hero_patient_timeline(
        hero_second_df,
        threshold=temp_thresh,
        out_path=FIGURES_DIR / "hero_sepsis_patient_timeline.png",
        title_suffix=" (Acute Cardiorespiratory Deterioration)",
    )

    print("=== Lead-Time Summary (results/lead_time_summary.csv) ===")
    print(summary_df.to_string(index=False))
    print("\n=== False Alarm Burden (results/alarm_burden_results.csv) ===")
    print(alarm_df.to_string(index=False))
    return summary_df, alarm_df


if __name__ == "__main__":
    run_lead_time_and_alarm_evaluation()

