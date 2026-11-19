"""
NEWS2 clinical early warning score baseline implementation and distribution plots.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.config import DATA_PROCESSED, FIGURES_DIR


def _is_missing(val):
    """Return True if val is None or NaN."""
    if val is None:
        return True
    try:
        return bool(np.isnan(val))
    except TypeError:
        return False


def score_news2(hr, rr, spo2, sbp, temp):
    """
    Compute the 5-parameter physiological NEWS2 score from current-hour vital signs.
    Returns integer score in [0, 15]. Missing (NaN/None) parameters contribute 0 points.
    """
    score = 0

    # 1. Respiratory rate (breaths/min)
    if not _is_missing(rr):
        if rr <= 8:
            score += 3
        elif rr <= 11:
            score += 1
        elif rr <= 20:
            score += 0
        elif rr <= 24:
            score += 2
        else:
            score += 3

    # 2. SpO2 (%) — Scale 1
    if not _is_missing(spo2):
        if spo2 <= 91:
            score += 3
        elif spo2 <= 93:
            score += 2
        elif spo2 <= 95:
            score += 1
        else:
            score += 0

    # 3. Systolic BP (mmHg)
    if not _is_missing(sbp):
        if sbp <= 90:
            score += 3
        elif sbp <= 100:
            score += 2
        elif sbp <= 110:
            score += 1
        elif sbp <= 219:
            score += 0
        else:
            score += 3

    # 4. Heart rate (bpm)
    if not _is_missing(hr):
        if hr <= 40:
            score += 3
        elif hr <= 50:
            score += 1
        elif hr <= 90:
            score += 0
        elif hr <= 110:
            score += 1
        elif hr <= 130:
            score += 2
        else:
            score += 3

    # 5. Temperature (Celsius)
    if not _is_missing(temp):
        if temp <= 35.0:
            score += 3
        elif temp <= 36.0:
            score += 1
        elif temp <= 38.0:
            score += 0
        elif temp <= 39.0:
            score += 1
        else:
            score += 2

    return int(score)


def apply_news2_to_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score every row in the dataset using NEWS2 rules.
    Returns a copy of df with an integer 'news2_score' column.
    """
    out = df.copy()
    hr = out["heartrate"].to_numpy(dtype=float) if "heartrate" in out.columns else np.full(len(out), np.nan)
    rr = out["respiration"].to_numpy(dtype=float) if "respiration" in out.columns else np.full(len(out), np.nan)
    spo2 = out["spo2"].to_numpy(dtype=float) if "spo2" in out.columns else np.full(len(out), np.nan)
    sbp = out["systemicsystolic"].to_numpy(dtype=float) if "systemicsystolic" in out.columns else np.full(len(out), np.nan)
    temp = out["temperature"].to_numpy(dtype=float) if "temperature" in out.columns else np.full(len(out), np.nan)

    scores = np.zeros(len(out), dtype=int)

    # RR
    valid_rr = ~np.isnan(rr)
    scores += np.where(valid_rr & (rr <= 8), 3, 0)
    scores += np.where(valid_rr & (rr > 8) & (rr <= 11), 1, 0)
    scores += np.where(valid_rr & (rr > 20) & (rr <= 24), 2, 0)
    scores += np.where(valid_rr & (rr > 24), 3, 0)

    # SpO2
    valid_spo2 = ~np.isnan(spo2)
    scores += np.where(valid_spo2 & (spo2 <= 91), 3, 0)
    scores += np.where(valid_spo2 & (spo2 > 91) & (spo2 <= 93), 2, 0)
    scores += np.where(valid_spo2 & (spo2 > 93) & (spo2 <= 95), 1, 0)

    # SBP
    valid_sbp = ~np.isnan(sbp)
    scores += np.where(valid_sbp & (sbp <= 90), 3, 0)
    scores += np.where(valid_sbp & (sbp > 90) & (sbp <= 100), 2, 0)
    scores += np.where(valid_sbp & (sbp > 100) & (sbp <= 110), 1, 0)
    scores += np.where(valid_sbp & (sbp > 219), 3, 0)

    # HR
    valid_hr = ~np.isnan(hr)
    scores += np.where(valid_hr & (hr <= 40), 3, 0)
    scores += np.where(valid_hr & (hr > 40) & (hr <= 50), 1, 0)
    scores += np.where(valid_hr & (hr > 90) & (hr <= 110), 1, 0)
    scores += np.where(valid_hr & (hr > 110) & (hr <= 130), 2, 0)
    scores += np.where(valid_hr & (hr > 130), 3, 0)

    # Temp
    valid_temp = ~np.isnan(temp)
    scores += np.where(valid_temp & (temp <= 35.0), 3, 0)
    scores += np.where(valid_temp & (temp > 35.0) & (temp <= 36.0), 1, 0)
    scores += np.where(valid_temp & (temp > 38.0) & (temp <= 39.0), 1, 0)
    scores += np.where(valid_temp & (temp > 39.0), 2, 0)

    out["news2_score"] = scores
    return out


def plot_news2_distribution_v0(df: pd.DataFrame, out_path=None):
    """Plot histogram of NEWS2 scores across v0_labeled.parquet."""
    if out_path is None:
        out_path = FIGURES_DIR / "news2_distribution_v0.png"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    scored = apply_news2_to_dataset(df) if "news2_score" not in df.columns else df
    max_s = int(max(scored["news2_score"].max(), 12))
    bins = np.arange(-0.5, max_s + 1.5, 1.0)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(
        scored["news2_score"],
        bins=bins,
        color="#2b5c8f",
        edgecolor="white",
        alpha=0.88,
    )
    ax.axvline(
        4.5,
        color="#d95f02",
        linestyle="--",
        linewidth=1.8,
        label="Clinical Alert Threshold (NEWS2 ≥ 5)",
    )
    pct_low = (scored["news2_score"] <= 4).mean() * 100
    pct_high = (scored["news2_score"] >= 5).mean() * 100
    ax.set_title(
        f"NEWS2 Score Distribution on v0 At-Risk Cohort (N = {len(scored):,} patient-hours)\n"
        f"Low Risk (0–4): {pct_low:.1f}% | Alert Trigger (≥ 5): {pct_high:.1f}%",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Modified 5-Parameter NEWS2 Score", fontsize=10)
    ax.set_ylabel("Number of At-Risk Patient-Hours", fontsize=10)
    ax.set_xticks(range(0, max_s + 1))
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return scored


def plot_news2_by_outcome(df: pd.DataFrame, out_path=None):
    """
    Plot side-by-side normalized NEWS2 score histograms on v1.parquet comparing
    deteriorating patient-hours (or deteriorating patients) vs. stable patients.
    """
    if out_path is None:
        out_path = FIGURES_DIR / "news2_by_outcome.png"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    scored = apply_news2_to_dataset(df) if "news2_score" not in df.columns else df.copy()
    det_pids = set(
        scored.groupby("patientunitstayid")["label"].max().loc[lambda s: s == 1].index
    )
    scored["patient_group"] = np.where(
        scored["patientunitstayid"].isin(det_pids),
        "Deteriorating Patients",
        "Stable Patients",
    )

    max_s = int(max(scored["news2_score"].max(), 12))
    bins = np.arange(-0.5, max_s + 1.5, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

    # Panel A: By Patient Outcome Group
    det_scores = scored[scored["patient_group"] == "Deteriorating Patients"]["news2_score"]
    stb_scores = scored[scored["patient_group"] == "Stable Patients"]["news2_score"]
    axes[0].hist(
        stb_scores,
        bins=bins,
        density=True,
        alpha=0.65,
        color="#2b5c8f",
        edgecolor="white",
        label=f"Never Deteriorated (mean={stb_scores.mean():.2f})",
    )
    axes[0].hist(
        det_scores,
        bins=bins,
        density=True,
        alpha=0.65,
        color="#d95f02",
        edgecolor="white",
        label=f"Eventually Deteriorated (mean={det_scores.mean():.2f})",
    )
    axes[0].axvline(4.5, color="black", linestyle="--", linewidth=1.3, label="Alert (NEWS2 ≥ 5)")
    axes[0].set_title("A. NEWS2 Distribution by Patient Outcome", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Modified NEWS2 Score", fontsize=10)
    axes[0].set_ylabel("Density of Patient-Hours", fontsize=10)
    axes[0].set_xticks(range(0, max_s + 1))
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend(frameon=True, fontsize=9)

    # Panel B: Immediate 24h Pre-Event Window (label == 1) vs Stable Hours (label == 0)
    pos_scores = scored[scored["label"] == 1]["news2_score"]
    neg_scores = scored[scored["label"] == 0]["news2_score"]
    axes[1].hist(
        neg_scores,
        bins=bins,
        density=True,
        alpha=0.65,
        color="#4daf4a",
        edgecolor="white",
        label=f"No Event in Next 24h (label=0, mean={neg_scores.mean():.2f})",
    )
    axes[1].hist(
        pos_scores,
        bins=bins,
        density=True,
        alpha=0.65,
        color="#e41a1c",
        edgecolor="white",
        label=f"Event in Next 24h (label=1, mean={pos_scores.mean():.2f})",
    )
    axes[1].axvline(4.5, color="black", linestyle="--", linewidth=1.3, label="Alert (NEWS2 ≥ 5)")
    axes[1].set_title("B. NEWS2 Distribution by 24h Horizon Label", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Modified NEWS2 Score", fontsize=10)
    axes[1].set_xticks(range(0, max_s + 1))
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend(frameon=True, fontsize=9)

    plt.suptitle("NEWS2 Score Comparison on Frozen v1 Dataset", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return scored


def main():
    v1_path = DATA_PROCESSED / "v1.parquet"
    if v1_path.exists():
        df = pd.read_parquet(v1_path)
        scored = plot_news2_by_outcome(df)
        print("Saved results/figures/news2_by_outcome.png")
        print(
            scored.groupby("label")["news2_score"]
            .agg(["count", "mean", "std", "median"])
            .to_string()
        )
    else:
        v0_path = DATA_PROCESSED / "v0_labeled.parquet"
        df = pd.read_parquet(v0_path)
        plot_news2_distribution_v0(df)


if __name__ == "__main__":
    main()
