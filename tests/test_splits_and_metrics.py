"""
Unit tests for patient-level stratified splitting and cluster bootstrap CIs.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.data_engineering.build_splits import build_patient_splits
from src.evaluation.metrics import bootstrap_metric


def test_build_patient_splits_zero_leakage():
    rows = []
    for pid in range(1, 161):
        is_sep = (pid % 4) == 0
        ever_det = 1 if (pid % 5) == 0 else 0
        for h in range(5):
            rows.append(
                {
                    "patientunitstayid": pid,
                    "hour": h,
                    "label": ever_det if h == 4 else 0,
                    "is_sepsis": is_sep,
                }
            )
    df = pd.DataFrame(rows)
    split_df = build_patient_splits(df, train_frac=0.70, val_frac=0.15, seed=42)

    train_ids = set(split_df[split_df.split == "train"].patientunitstayid)
    val_ids = set(split_df[split_df.split == "val"].patientunitstayid)
    test_ids = set(split_df[split_df.split == "test"].patientunitstayid)

    assert len(train_ids & val_ids) == 0
    assert len(train_ids & test_ids) == 0
    assert len(val_ids & test_ids) == 0
    assert len(train_ids | val_ids | test_ids) == 160


def test_bootstrap_metric_returns_valid_ci():
    rng = np.random.RandomState(42)
    pids = np.repeat(np.arange(1, 31), 10)
    y_true = (pids <= 6).astype(int)
    y_score = y_true * 0.6 + rng.uniform(0.0, 0.4, size=len(pids))

    res = bootstrap_metric(
        y_true, y_score, pids, roc_auc_score, n_bootstraps=200, seed=42
    )
    assert 0.0 <= res["ci_lower"] <= res["point"] <= res["ci_upper"] <= 1.0
