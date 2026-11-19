"""
Causal 6-hour trajectory feature engineering and missingness audit.

Computes backward-looking 6-hour rolling statistics (mean, std, min, max),
linear trend slopes, 6-hour deltas, and shock index.
Strictly preserves NaN for unobserved trends/slopes/deltas (never .fillna(0)).
"""

import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.config import (
    ALL_VARS,
    DATA_PROCESSED,
    FIGURES_DIR,
    LAB_VARS,
    VITAL_VARS,
)


def _slope(series: pd.Series) -> float:
    """Reference slope helper: linear trend over valid observations in the last 6 hours."""
    valid = series.dropna()
    if len(valid) < 2:
        return np.nan
    x = np.arange(len(series))[series.notna()]
    return float(np.polyfit(x, valid.values, 1)[0])


def _vectorized_slope_6h(df: pd.DataFrame, group_col: str, val_col: str) -> np.ndarray:
    """
    Exact vectorized equivalent of groupby(group_col)[val_col].rolling(6, min_periods=2).apply(_slope).
    Because OLS slope Cov(x, y) / Var(x) is translation-invariant in x, shifting lags [t-5..t]
    within each patient group computes the exact polyfit(x, y, 1)[0] in O(N) vectorized NumPy.
    """
    grp = df.groupby(group_col, sort=False)[val_col]
    # Build (N, 6) matrix of [t-5, t-4, t-3, t-2, t-1, t]
    lags = [grp.shift(5 - k).to_numpy(dtype=float) for k in range(6)]
    Y = np.column_stack(lags)  # shape (N, 6)
    X = np.arange(6, dtype=float)[None, :]  # shape (1, 6)

    valid_mask = ~np.isnan(Y)
    n_valid = valid_mask.sum(axis=1)

    # Replace NaNs with 0 for masked sums
    Y_zero = np.where(valid_mask, Y, 0.0)
    X_zero = np.where(valid_mask, X, 0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        x_mean = X_zero.sum(axis=1, keepdims=True) / np.maximum(n_valid[:, None], 1)
        y_mean = Y_zero.sum(axis=1, keepdims=True) / np.maximum(n_valid[:, None], 1)

        dx = np.where(valid_mask, X - x_mean, 0.0)
        dy = np.where(valid_mask, Y - y_mean, 0.0)

        num = (dx * dy).sum(axis=1)
        den = (dx * dx).sum(axis=1)
        slopes = np.where((n_valid >= 2) & (den > 1e-12), num / den, np.nan)

    return slopes


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal features to the hourly-binned dataset.
    All features computed CAUSALLY — only data <= current hour.
    IMPORTANT: Leave missing trends/std/deltas as NaN (do NOT .fillna(0)),
    because 0 means "perfectly flat/stable" whereas NaN means "unobserved"
    (which XGBoost handles natively alongside the miss_<var> flags).
    """
    feature_df = df.sort_values(["patientunitstayid", "hour"]).reset_index(drop=True).copy()

    new_cols = {}
    for var in ALL_VARS:
        grp = feature_df.groupby("patientunitstayid", sort=False)[var]

        # Rolling statistics (6-hour window)
        new_cols[f"{var}_roll_mean_6h"] = (
            grp.rolling(6, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        new_cols[f"{var}_roll_std_6h"] = (
            grp.rolling(6, min_periods=2).std().reset_index(level=0, drop=True)
        )
        new_cols[f"{var}_roll_min_6h"] = (
            grp.rolling(6, min_periods=1).min().reset_index(level=0, drop=True)
        )
        new_cols[f"{var}_roll_max_6h"] = (
            grp.rolling(6, min_periods=1).max().reset_index(level=0, drop=True)
        )

        # Slope: linear trend over last 6 hours (NaN if < 2 valid points)
        new_cols[f"{var}_slope_6h"] = _vectorized_slope_6h(
            feature_df, "patientunitstayid", var
        )

        # Delta from 6 hours ago (NaN if current or t-6 value is missing)
        new_cols[f"{var}_delta_6h"] = grp.diff(periods=6)

    feature_df = pd.concat([feature_df, pd.DataFrame(new_cols, index=feature_df.index)], axis=1)

    # Shock index
    feature_df["shock_index"] = (
        feature_df["heartrate"] / feature_df["systemicsystolic"].clip(lower=1.0)
    )

    return feature_df


def generate_missingness_heatmap(df: pd.DataFrame, out_path=None) -> pd.DataFrame:
    """
    Compute % missing per variable across ICU stay hour buckets (after 4h vital
    and 24h lab forward-fill) and save a heatmap to results/figures/missingness_heatmap.png.
    """
    if out_path is None:
        out_path = FIGURES_DIR / "missingness_heatmap.png"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    temp = df.copy()
    # Group hours into 12-hour intervals over the first 7 days (0-168h) for heatmap clarity
    temp_sub = temp[temp["hour"] < 168].copy()
    bins = list(range(0, 169, 12))
    labels = [f"{b}–{b+11}h" for b in bins[:-1]]
    temp_sub["hour_bucket"] = pd.cut(
        temp_sub["hour"], bins=bins, right=False, labels=labels
    )

    miss_matrix = (
        temp_sub.groupby("hour_bucket", observed=False)[ALL_VARS]
        .apply(lambda g: g.isna().mean() * 100.0)
        .T
    )
    overall_miss = df[ALL_VARS].isna().mean() * 100.0
    miss_matrix["Overall (%)"] = overall_miss

    fig, ax = plt.subplots(figsize=(12, 5.8))
    sns.heatmap(
        miss_matrix,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "% Missing After Forward-Fill (4h Vitals / 24h Labs)"},
        ax=ax,
    )
    ax.set_title(
        "Variable Missingness Audit Across ICU Timeline (After 4h Vital & 24h Lab Forward-Fill)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Hours Since ICU Admission", fontsize=10)
    ax.set_ylabel("Clinical Variable (Vitals & Labs)", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return miss_matrix


def main():
    t0 = time.time()
    v0_path = DATA_PROCESSED / "v0.parquet"
    print(f"Loading {v0_path}...")
    v0_df = pd.read_parquet(v0_path)

    print("Computing 6-hour causal trajectory features...")
    v1_features = engineer_features(v0_df)

    out_path = DATA_PROCESSED / "v1_features.parquet"
    v1_features.to_parquet(out_path, index=False)
    elapsed = time.time() - t0
    print(
        f"Saved {out_path} ({v1_features.shape[0]:,} rows, {v1_features.shape[1]} columns) "
        f"in {elapsed:.2f}s."
    )

    print("Running missingness audit after 4h vital / 24h lab forward-fill...")
    miss_summary = generate_missingness_heatmap(v1_features)
    print(miss_summary[["Overall (%)"]].round(2))


if __name__ == "__main__":
    main()
