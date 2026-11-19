"""
Model comparison framework for NEWS2, Static XGBoost, and Temporal XGBoost.

Computes AUROC, AUPRC, and F1 across models and generates side-by-side ROC
and Precision-Recall curves on validation and test splits.
"""

import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
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
    TEMPORAL_THRESHOLD,
)


def compare_models(
    eval_df: pd.DataFrame,
    models_dict: dict,
    thresholds_dict: dict = None,
) -> pd.DataFrame:
    """
    Given a dict of {model_name: predictions_array}, compute AUROC, AUPRC, and F1
    for each model. Returns a comparison DataFrame.
    """
    y_true = eval_df["label"].to_numpy(dtype=int)
    results = []
    for name, preds in models_dict.items():
        preds = np.asarray(preds, dtype=float)
        thresh = (
            thresholds_dict.get(name, 5.0 if name == "NEWS2" else 0.5)
            if thresholds_dict
            else (5.0 if name == "NEWS2" else 0.5)
        )
        y_bin = (preds >= thresh).astype(int)
        results.append(
            {
                "model": name,
                "auroc": round(float(roc_auc_score(y_true, preds)), 4),
                "auprc": round(float(average_precision_score(y_true, preds)), 4),
                "f1": round(float(f1_score(y_true, y_bin, zero_division=0)), 4),
                "threshold": round(float(thresh), 4),
            }
        )
    return pd.DataFrame(results)


def plot_roc_pr_curves(
    eval_df: pd.DataFrame,
    models_dict: dict,
    cohort_label: str,
    out_path,
    split_label: str = "Validation",
):
    """Plot combined ROC and Precision-Recall curves for all 3 models."""
    y_true = eval_df["label"].to_numpy(dtype=int)
    base_rate = float(y_true.mean())

    palette = {
        "NEWS2": ("#636363", "--"),
        "XGB_static": ("#2b5c8f", "-."),
        "XGB_temporal": ("#d95f02", "-"),
    }
    display_names = {
        "NEWS2": "NEWS2 (Clinical Snapshot Rules)",
        "XGB_static": "Static XGBoost (Snapshot ML)",
        "XGB_temporal": "Temporal XGBoost (Snapshot + 6h Trends)",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

    # Panel A: ROC Curve
    for name, preds in models_dict.items():
        fpr, tpr, _ = roc_curve(y_true, preds)
        auc_val = roc_auc_score(y_true, preds)
        color, ls = palette.get(name, ("#333333", "-"))
        label = f"{display_names.get(name, name)} (AUROC = {auc_val:.3f})"
        axes[0].plot(fpr, tpr, color=color, linestyle=ls, linewidth=2.2, label=label)

    axes[0].plot(
        [0, 1],
        [0, 1],
        color="#bbbbbb",
        linestyle=":",
        linewidth=1.2,
        label="Random (0.500)",
    )
    axes[0].set_title(
        f"A. ROC Curves — {cohort_label} ({split_label})",
        fontsize=11,
        fontweight="bold",
    )
    axes[0].set_xlabel("False Positive Rate (1 − Specificity)", fontsize=10)
    axes[0].set_ylabel("True Positive Rate (Sensitivity)", fontsize=10)
    axes[0].legend(loc="lower right", frameon=True, fontsize=9)
    axes[0].grid(alpha=0.3)

    # Panel B: Precision-Recall Curve
    for name, preds in models_dict.items():
        prec, rec, _ = precision_recall_curve(y_true, preds)
        ap_val = average_precision_score(y_true, preds)
        color, ls = palette.get(name, ("#333333", "-"))
        label = f"{display_names.get(name, name)} (AUPRC = {ap_val:.3f})"
        axes[1].plot(rec, prec, color=color, linestyle=ls, linewidth=2.2, label=label)

    axes[1].axhline(
        base_rate,
        color="#bbbbbb",
        linestyle=":",
        linewidth=1.4,
        label=f"Baseline Event Rate ({base_rate:.3f})",
    )
    axes[1].set_title(
        f"B. Precision-Recall Curves — {cohort_label} ({split_label})",
        fontsize=11,
        fontweight="bold",
    )
    axes[1].set_xlabel("Recall (Sensitivity)", fontsize=10)
    axes[1].set_ylabel("Precision (Positive Predictive Value)", fontsize=10)
    axes[1].legend(loc="upper right", frameon=True, fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def run_validation_comparison():
    """
    Evaluate NEWS2, Static XGBoost, and Temporal XGBoost on the Validation split
    for both Primary Cohort and Secondary Sepsis Subgroup. Save full table to
    results/experiment_results.csv and plot validation ROC/PR curves.
    """
    v1_df = pd.read_parquet(DATA_PROCESSED / "v1.parquet")
    val_df = v1_df[v1_df["split"] == "val"].copy()
    val_df = apply_news2_to_dataset(val_df)

    xgb_static = joblib.load(MODELS_DIR / "xgb_static.pkl")
    xgb_temporal = joblib.load(MODELS_DIR / "xgb_temporal.pkl")

    with open(RESULTS_DIR / "xgb_static_params.json", "r", encoding="utf-8") as f:
        static_thresh = float(json.load(f)["optimal_f1_threshold"])
    with open(RESULTS_DIR / "xgb_temporal_params.json", "r", encoding="utf-8") as f:
        temp_thresh = float(json.load(f)["optimal_f1_threshold"])

    thresholds = {
        "NEWS2": float(NEWS2_THRESHOLD),
        "XGB_static": static_thresh,
        "XGB_temporal": temp_thresh,
    }

    val_df["pred_news2"] = val_df["news2_score"].astype(float)
    val_df["pred_static"] = xgb_static.predict_proba(val_df[SNAPSHOT_FEATURES])[:, 1]
    val_df["pred_temporal"] = xgb_temporal.predict_proba(val_df[ALL_FEATURES])[:, 1]

    all_rows = []
    for cohort_key, cohort_title, sub in [
        ("primary", "Primary Adult ICU Cohort", val_df),
        ("sepsis_subgroup", "Secondary Sepsis Subgroup", val_df[val_df["is_sepsis"] == True]),
    ]:
        models_dict = {
            "NEWS2": sub["pred_news2"].to_numpy(),
            "XGB_static": sub["pred_static"].to_numpy(),
            "XGB_temporal": sub["pred_temporal"].to_numpy(),
        }
        comp = compare_models(sub, models_dict, thresholds_dict=thresholds)
        for row in comp.to_dict(orient="records"):
            all_rows.append(
                {
                    "model": row["model"],
                    "cohort": cohort_key,
                    "auroc": row["auroc"],
                    "auprc": row["auprc"],
                    "f1": row["f1"],
                    "threshold": row["threshold"],
                }
            )
        fig_name = (
            "val_roc_pr_primary.png"
            if cohort_key == "primary"
            else "val_roc_pr_sepsis.png"
        )
        plot_roc_pr_curves(
            sub,
            models_dict,
            cohort_label=cohort_title,
            out_path=FIGURES_DIR / fig_name,
            split_label="Validation",
        )

    order = {"NEWS2": 0, "XGB_static": 1, "XGB_temporal": 2}
    cohort_order = {"primary": 0, "sepsis_subgroup": 1}
    res_df = pd.DataFrame(all_rows).sort_values(
        by=["model", "cohort"],
        key=lambda s: s.map(order) if s.name == "model" else s.map(cohort_order),
    )
    res_df.to_csv(RESULTS_DIR / "experiment_results.csv", index=False)
    print("=== Validation Comparison (results/experiment_results.csv) ===")
    print(res_df.to_string(index=False))
    return res_df


def plot_test_ci_bar_chart(test_res_df: pd.DataFrame, out_path=None):
    """Plot AUROC and AUPRC bar charts with 95% patient-bootstrap CI error bars."""
    if out_path is None:
        out_path = FIGURES_DIR / "test_ci_bar_chart.png"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    cohorts = ["Primary_Cohort", "Sepsis_Subgroup"]
    cohort_labels = ["Primary Adult ICU Cohort", "Secondary Sepsis Subgroup"]
    models = ["NEWS2", "XGB_static", "XGB_temporal"]
    model_labels = {
        "NEWS2": "NEWS2 (Snapshot Rules)",
        "XGB_static": "Static XGBoost (Snapshot ML)",
        "XGB_temporal": "Temporal XGBoost (+6h Trends)",
    }
    colors = {"NEWS2": "#7f7f7f", "XGB_static": "#2b5c8f", "XGB_temporal": "#d95f02"}

    x = np.arange(len(cohorts))
    width = 0.25

    for m_idx, (metric_col, low_col, high_col, title_str, ylabel_str) in enumerate(
        [
            ("auroc_point", "auroc_ci_lower", "auroc_ci_upper", "A. Test Set AUROC (with 95% Patient-Bootstrap CI)", "AUROC"),
            ("auprc_point", "auprc_ci_lower", "auprc_ci_upper", "B. Test Set AUPRC (with 95% Patient-Bootstrap CI)", "AUPRC"),
        ]
    ):
        ax = axes[m_idx]
        for i, m in enumerate(models):
            sub = test_res_df[test_res_df["model"] == m].set_index("cohort").reindex(cohorts)
            pts = sub[metric_col].to_numpy(dtype=float)
            lows = sub[low_col].to_numpy(dtype=float)
            highs = sub[high_col].to_numpy(dtype=float)
            yerr = np.vstack([np.maximum(0, pts - lows), np.maximum(0, highs - pts)])

            bars = ax.bar(
                x + (i - 1) * width,
                pts,
                width=width,
                color=colors[m],
                edgecolor="white",
                label=model_labels[m],
                yerr=yerr,
                capsize=4,
                error_kw=dict(lw=1.3, ecolor="#222222"),
            )
            for bar, pt, hi in zip(bars, pts, highs):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    hi + (0.018 if m_idx == 0 else 0.015),
                    f"{pt:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    fontweight="bold",
                )

        ax.set_title(title_str, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(cohort_labels, fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel_str, fontsize=10)
        ax.set_ylim(0, 1.05 if m_idx == 0 else max(0.45, test_res_df["auprc_ci_upper"].max() + 0.08))
        ax.grid(axis="y", alpha=0.3)
        ax.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_test_evaluation():
    """
    Evaluate all 3 locked models (NEWS2, Static XGBoost, Temporal XGBoost) on the
    held-out Test split with 1,000-resample patient-level cluster bootstrap 95% CIs.
    Saves results/final_test_results.csv and final test ROC/PR/CI figures.
    """
    from src.evaluation.metrics import bootstrap_metric

    v1_df = pd.read_parquet(DATA_PROCESSED / "v1.parquet")
    test_df = v1_df[v1_df["split"] == "test"].copy()
    test_df = apply_news2_to_dataset(test_df)

    xgb_static = joblib.load(MODELS_DIR / "xgb_static.pkl")
    xgb_temporal = joblib.load(MODELS_DIR / "xgb_temporal.pkl")

    with open(RESULTS_DIR / "xgb_static_params.json", "r", encoding="utf-8") as f:
        static_thresh = float(json.load(f)["optimal_f1_threshold"])
    with open(RESULTS_DIR / "xgb_temporal_params.json", "r", encoding="utf-8") as f:
        temp_thresh = float(json.load(f)["optimal_f1_threshold"])

    eval_subsets = {
        "Primary_Cohort": ("Primary Adult ICU Cohort", test_df),
        "Sepsis_Subgroup": (
            "Secondary Sepsis Subgroup",
            test_df[test_df["is_sepsis"] == True].copy(),
        ),
    }

    all_results = []
    for cohort_name, (cohort_title, df_sub) in eval_subsets.items():
        y_true = df_sub["label"].to_numpy(dtype=int)
        patient_ids = df_sub["patientunitstayid"].to_numpy()
        news2_preds = df_sub["news2_score"].to_numpy(dtype=float)
        static_preds = xgb_static.predict_proba(df_sub[SNAPSHOT_FEATURES])[:, 1]
        temporal_preds = xgb_temporal.predict_proba(df_sub[ALL_FEATURES])[:, 1]

        models_dict = {
            "NEWS2": news2_preds,
            "XGB_static": static_preds,
            "XGB_temporal": temporal_preds,
        }

        fig_name = (
            "test_roc_pr_primary.png"
            if cohort_name == "Primary_Cohort"
            else "test_roc_pr_sepsis.png"
        )
        plot_roc_pr_curves(
            df_sub,
            models_dict,
            cohort_label=cohort_title,
            out_path=FIGURES_DIR / fig_name,
            split_label="Held-Out Test Set",
        )

        for name, preds in models_dict.items():
            auroc = bootstrap_metric(y_true, preds, patient_ids, roc_auc_score, n_bootstraps=1000)
            auprc = bootstrap_metric(
                y_true, preds, patient_ids, average_precision_score, n_bootstraps=1000
            )
            thresh = (
                float(NEWS2_THRESHOLD)
                if name == "NEWS2"
                else (static_thresh if name == "XGB_static" else temp_thresh)
            )
            y_bin = (preds >= thresh).astype(int)
            f1_val = float(f1_score(y_true, y_bin, zero_division=0))

            all_results.append(
                {
                    "cohort": cohort_name,
                    "model": name,
                    "auroc_point": round(auroc["point"], 4),
                    "auroc_ci_lower": round(auroc["ci_lower"], 4),
                    "auroc_ci_upper": round(auroc["ci_upper"], 4),
                    "auroc_ci": f"({auroc['ci_lower']:.3f}, {auroc['ci_upper']:.3f})",
                    "auprc_point": round(auprc["point"], 4),
                    "auprc_ci_lower": round(auprc["ci_lower"], 4),
                    "auprc_ci_upper": round(auprc["ci_upper"], 4),
                    "auprc_ci": f"({auprc['ci_lower']:.3f}, {auprc['ci_upper']:.3f})",
                    "f1": round(f1_val, 4),
                    "threshold": round(thresh, 4),
                }
            )

    res_df = pd.DataFrame(all_results)
    res_df.to_csv(RESULTS_DIR / "final_test_results.csv", index=False)
    plot_test_ci_bar_chart(res_df)
    print("=== Final Test Set Results (results/final_test_results.csv) ===")
    print(
        res_df[
            ["cohort", "model", "auroc_point", "auroc_ci", "auprc_point", "auprc_ci", "f1"]
        ].to_string(index=False)
    )
    return res_df


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="NEWS2 / Static / Temporal XGBoost comparison")
    ap.add_argument("--stage", choices=["val", "test", "all"], default="test",
                    help="val = validation comparison + ROC/PR; test = held-out test with bootstrap CIs")
    args = ap.parse_args()
    if args.stage in ("val", "all"):
        run_validation_comparison()
    if args.stage in ("test", "all"):
        run_test_evaluation()

