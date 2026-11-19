"""
Exploratory Data Analysis (EDA) figures, CONSORT diagram, cohort demographics table,
and end-to-end feature pipeline profiling.
"""

import json
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.config import (
    ALL_VARS,
    DATA_PROCESSED,
    DATA_RAW,
    FIGURES_DIR,
    RESULTS_DIR,
    SNAPSHOT_FEATURES,
)


def build_demographics_and_plots(v1_df: pd.DataFrame, patient_df: pd.DataFrame):
    """
    Generate cohort demographics table (results/cohort_demographics.csv) and
    demographics distribution figure (results/figures/cohort_demographics.png).
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pt_summary = (
        v1_df.groupby("patientunitstayid")
        .agg(
            ever_deteriorated=("label", "max"),
            is_sepsis=("is_sepsis", "first"),
            split=("split", "first"),
            at_risk_hours=("hour", "count"),
        )
        .reset_index()
    )

    p_raw = patient_df.copy()
    p_raw["age_num"] = pd.to_numeric(
        p_raw["age"].astype(str).str.replace("> 89", "89", regex=False),
        errors="coerce",
    )
    p_raw["los_days"] = p_raw["unitdischargeoffset"] / 1440.0
    pt_merged = pd.merge(
        pt_summary,
        p_raw[
            [
                "patientunitstayid",
                "age_num",
                "gender",
                "ethnicity",
                "los_days",
                "unitdischargestatus",
            ]
        ],
        on="patientunitstayid",
        how="left",
    )

    def _summarize(sub, name):
        return {
            "Group": name,
            "Patients (N)": len(sub),
            "Age Median [IQR]": f"{sub['age_num'].median():.0f} [{sub['age_num'].quantile(0.25):.0f}–{sub['age_num'].quantile(0.75):.0f}]",
            "Female (%)": f"{(sub['gender'] == 'Female').mean() * 100:.1f}%",
            "ICU LOS Days Median [IQR]": f"{sub['los_days'].median():.1f} [{sub['los_days'].quantile(0.25):.1f}–{sub['los_days'].quantile(0.75):.1f}]",
            "Sepsis Subgroup (%)": f"{sub['is_sepsis'].mean() * 100:.1f}%",
            "ICU Mortality (%)": f"{(sub['unitdischargestatus'] == 'Expired').mean() * 100:.1f}%",
        }

    demo_rows = [
        _summarize(pt_merged, "All Eligible Adult ICU (Primary)"),
        _summarize(pt_merged[pt_merged["ever_deteriorated"] == 1], "Deteriorating Patients"),
        _summarize(pt_merged[pt_merged["ever_deteriorated"] == 0], "Stable Patients"),
        _summarize(pt_merged[pt_merged["is_sepsis"] == True], "Secondary Sepsis Subgroup"),
    ]
    demo_df = pd.DataFrame(demo_rows)
    demo_df.to_csv(RESULTS_DIR / "cohort_demographics.csv", index=False)

    # Plot demographics distributions (Age, LOS, Gender)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    pt_merged["Outcome"] = np.where(
        pt_merged["ever_deteriorated"] == 1, "Deteriorated (n=90)", "Stable (n=594)"
    )

    sns.histplot(
        data=pt_merged,
        x="age_num",
        hue="Outcome",
        bins=15,
        kde=True,
        palette={"Stable (n=594)": "#2b5c8f", "Deteriorated (n=90)": "#d95f02"},
        ax=axes[0],
    )
    axes[0].set_title("A. Age Distribution by Outcome", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Age (Years)")
    axes[0].set_ylabel("Number of Patients")

    sns.boxplot(
        data=pt_merged[pt_merged["los_days"] <= 20],
        x="Outcome",
        y="los_days",
        hue="Outcome",
        palette={"Stable (n=594)": "#2b5c8f", "Deteriorated (n=90)": "#d95f02"},
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("B. ICU Length of Stay (Capped at 20d)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("ICU Length of Stay (Days)")

    gender_ct = (
        pt_merged.groupby(["Outcome", "gender"])
        .size()
        .unstack(fill_value=0)
    )
    gender_pct = gender_ct.div(gender_ct.sum(axis=1), axis=0) * 100.0
    gender_pct[["Male", "Female"]].plot(
        kind="bar",
        stacked=True,
        color=["#4e79a7", "#f28e2b"],
        ax=axes[2],
        rot=0,
    )
    axes[2].set_title("C. Biological Sex Breakdown", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("Percentage of Patients (%)")
    axes[2].legend(title="Sex", frameon=True)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "cohort_demographics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return demo_df


def plot_consort_diagram(out_path=None):
    """Draw publication-ready CONSORT cohort attrition diagram."""
    if out_path is None:
        out_path = FIGURES_DIR / "consort_diagram.png"

    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    box_style = dict(boxstyle="round,pad=0.45", facecolor="#eef4fa", edgecolor="#2b5c8f", linewidth=1.6)
    excl_style = dict(boxstyle="round,pad=0.35", facecolor="#fff5f0", edgecolor="#d95f02", linewidth=1.3)
    final_style = dict(boxstyle="round,pad=0.45", facecolor="#e6f5ea", edgecolor="#2ca25f", linewidth=1.8)

    steps = [
        (3.2, 9.0, "eICU-CRD Demo Dataset (v2.0.1)\nN = 2,520 ICU stays", box_style),
        (3.2, 7.3, "Adult ICU Stays (Age ≥ 18 years)\nN = 2,512 stays", box_style),
        (3.2, 5.6, "First ICU Admission Only (unitvisitnumber == 1)\nN = 2,111 stays", box_style),
        (3.2, 3.9, "Longitudinal ICU Stays (LOS ≥ 48 hours)\nN = 793 stays (93,999 raw patient-hours)", box_style),
        (
            3.2,
            2.0,
            "Final Frozen At-Risk Primary Cohort (v1.parquet)\n"
            "N = 684 stays (90 deteriorating, 13.2%)\n"
            "64,782 at-risk patient-hours (Train: 478 | Val: 103 | Test: 103)",
            final_style,
        ),
    ]
    exclusions = [
        (7.7, 8.15, "Excluded: Age < 18 or Missing\n(n = 8 stays)", excl_style),
        (7.7, 6.45, "Excluded: Repeat ICU Readmissions\n(n = 401 stays)", excl_style),
        (7.7, 4.75, "Excluded: ICU LOS < 48 hours\n(n = 1,318 stays)", excl_style),
        (
            7.7,
            2.95,
            "Excluded: Baseline / Early Crash ≤ 6h\n(n = 109 stays)\n+ Post-Event Hours Censored (t ≥ t_event)",
            excl_style,
        ),
        (
            7.7,
            0.85,
            "Secondary Sepsis Subgroup Tier\nN = 121 stays (26 deteriorating, 21.5%)\n12,806 at-risk patient-hours",
            box_style,
        ),
    ]

    for x, y, text, st in steps:
        ax.text(x, y, text, ha="center", va="center", fontsize=9.5, fontweight="bold", bbox=st)

    for x, y, text, st in exclusions:
        ax.text(x, y, text, ha="center", va="center", fontsize=8.8, bbox=st)

    # Vertical arrows
    for y_top, y_bot in [(8.5, 7.8), (6.8, 6.1), (5.1, 4.4), (3.4, 2.65)]:
        ax.annotate(
            "",
            xy=(3.2, y_bot),
            xytext=(3.2, y_top),
            arrowprops=dict(arrowstyle="->", color="#2b5c8f", lw=1.8),
        )

    # Horizontal exclusion arrows
    for y_mid in [8.15, 6.45, 4.75, 2.95]:
        ax.annotate(
            "",
            xy=(5.7, y_mid),
            xytext=(3.2, y_mid),
            arrowprops=dict(arrowstyle="->", color="#d95f02", lw=1.4, linestyle="--"),
        )

    # Subgroup arrow
    ax.annotate(
        "",
        xy=(5.6, 0.85),
        xytext=(3.2, 1.35),
        arrowprops=dict(arrowstyle="->", color="#2ca25f", lw=1.6),
    )

    ax.set_title(
        "CONSORT Cohort Attrition & Two-Tier Study Design Flowchart",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_feature_correlation_heatmap(v1_df: pd.DataFrame, out_path=None):
    """Plot correlation heatmap of core snapshot and 6-hour trajectory features."""
    if out_path is None:
        out_path = FIGURES_DIR / "feature_correlation_heatmap.png"

    selected_cols = (
        ALL_VARS
        + ["shock_index"]
        + [f"{v}_roll_mean_6h" for v in ["heartrate", "systemicsystolic", "systemicmean", "spo2", "respiration"]]
        + [f"{v}_slope_6h" for v in ["heartrate", "systemicsystolic", "systemicmean", "spo2", "respiration"]]
    )
    selected_cols = [c for c in selected_cols if c in v1_df.columns]
    corr = v1_df[selected_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 9.5))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        annot=False,
        square=True,
        cbar_kws={"label": "Pearson Correlation Coefficient (r)"},
        ax=ax,
    )
    ax.set_title(
        "Feature Correlation Matrix — Snapshot Vitals/Labs vs. 6-Hour Trajectory Features",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    plt.xticks(rotation=45, ha="right", fontsize=8.5)
    plt.yticks(fontsize=8.5)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sample_patient_vitals(v1_df: pd.DataFrame, out_path=None):
    """Plot longitudinal vital sign trajectories (HR, SBP, SpO2) for 4 sample ICU stays."""
    if out_path is None:
        out_path = FIGURES_DIR / "sample_patient_vitals.png"

    det_pids = (
        v1_df[v1_df["event_hour"].notna()]
        .groupby("patientunitstayid")["hour"]
        .count()
        .loc[lambda s: (s >= 24) & (s <= 72)]
        .index.tolist()
    )
    stb_pids = (
        v1_df[v1_df["event_hour"].isna()]
        .groupby("patientunitstayid")["hour"]
        .count()
        .loc[lambda s: (s >= 36) & (s <= 72)]
        .index.tolist()
    )
    sample_pids = det_pids[:2] + stb_pids[:2]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=False)
    axes = axes.flatten()

    for idx, pid in enumerate(sample_pids):
        sub = v1_df[v1_df["patientunitstayid"] == pid].sort_values("hour")
        ax = axes[idx]
        ev_hr = sub["event_hour"].iloc[0]
        outcome_str = (
            f"Deteriorated at Hour {int(ev_hr)}"
            if pd.notna(ev_hr)
            else "Stable Stay (No Event)"
        )

        ax.plot(sub["hour"], sub["heartrate"], color="#d95f02", lw=2.0, label="Heart Rate (bpm)")
        ax.plot(sub["hour"], sub["systemicsystolic"], color="#2b5c8f", lw=2.0, label="Systolic BP (mmHg)")
        ax.plot(sub["hour"], sub["spo2"], color="#2ca25f", lw=1.8, linestyle="--", label="SpO₂ (%)")
        if pd.notna(ev_hr):
            ax.axvspan(max(0, ev_hr - 24), ev_hr, color="#fee0d2", alpha=0.45, label="24h Pre-Crash Window")

        ax.set_title(f"Patient Stay {pid} — {outcome_str}", fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Hours Since ICU Admission")
        ax.set_ylabel("Physiological Value")
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(loc="best", fontsize=8.5, frameon=True)

    plt.suptitle(
        "Sample Patient Vital Sign Trajectories (1-Hour Medians with 4h Forward-Fill)",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def profile_pipeline_speed():
    """Profile end-to-end cohort extraction, temporal alignment, and feature engineering runtime."""
    from src.data_engineering.extract_cohort import extract_cohort
    from src.data_engineering.temporal_alignment import bin_to_hourly
    from src.data_engineering.feature_engineering import engineer_features

    t_start = time.time()
    t0 = time.time()
    patient_df = pd.read_csv(DATA_RAW / "patient.csv", low_memory=False)
    diagnosis_df = pd.read_csv(DATA_RAW / "diagnosis.csv", low_memory=False)
    cohort_df, _ = extract_cohort(patient_df, diagnosis_df)
    t_cohort = time.time() - t0

    t1 = time.time()
    vitals_df = pd.read_csv(
        DATA_RAW / "vitalPeriodic.csv",
        usecols=[
            "patientunitstayid",
            "observationoffset",
            "temperature",
            "sao2",
            "heartrate",
            "respiration",
            "systemicsystolic",
            "systemicdiastolic",
            "systemicmean",
        ],
        low_memory=False,
    )
    aperiodic_df = pd.read_csv(
        DATA_RAW / "vitalAperiodic.csv",
        usecols=[
            "patientunitstayid",
            "observationoffset",
            "noninvasivesystolic",
            "noninvasivediastolic",
            "noninvasivemean",
        ],
        low_memory=False,
    )
    labs_df = pd.read_csv(
        DATA_RAW / "lab.csv",
        usecols=["patientunitstayid", "labresultoffset", "labname", "labresult"],
        low_memory=False,
    )
    v0_df = bin_to_hourly(
        vitals_df,
        labs_df,
        cohort_df["patientunitstayid"].unique(),
        aperiodic_df=aperiodic_df,
        cohort_df=cohort_df,
    )
    t_align = time.time() - t1

    t2 = time.time()
    v1_feat = engineer_features(v0_df)
    t_feat = time.time() - t2
    t_total = time.time() - t_start

    profile = {
        "extract_cohort_seconds": round(t_cohort, 2),
        "temporal_alignment_seconds": round(t_align, 2),
        "feature_engineering_seconds": round(t_feat, 2),
        "total_pipeline_seconds": round(t_total, 2),
        "under_3_minute_target": bool(t_total < 180.0),
        "output_rows": int(len(v1_feat)),
        "output_columns": int(v1_feat.shape[1]),
    }
    with open(RESULTS_DIR / "pipeline_profile.json", "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
    return profile


def main():
    v1_df = pd.read_parquet(DATA_PROCESSED / "v1.parquet")
    patient_df = pd.read_csv(DATA_RAW / "patient.csv", low_memory=False)

    print("Generating cohort demographics table and plots...")
    demo_df = build_demographics_and_plots(v1_df, patient_df)
    print(demo_df.to_string(index=False))

    print("Generating CONSORT diagram...")
    plot_consort_diagram()

    print("Generating feature correlation heatmap...")
    plot_feature_correlation_heatmap(v1_df)

    print("Generating sample patient vital timelines...")
    plot_sample_patient_vitals(v1_df)

    print("Profiling end-to-end data pipeline runtime...")
    prof = profile_pipeline_speed()
    print(f"Pipeline profile: {prof}")


if __name__ == "__main__":
    main()
