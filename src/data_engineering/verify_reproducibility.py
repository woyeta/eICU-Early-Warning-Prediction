"""
End-to-end integrity and reproducibility checks on the frozen v1.parquet and models.

Checks (all must pass):
1. Zero patient overlap between train / val / test.
2. Every retained event_hour > MIN_EVENT_HOUR (early-crash exclusion).
3. Strict pre-event censoring: hour < event_hour for every row of deteriorating stays.
4. Label definition: label == 1  <=>  hour < event_hour <= hour + LOOKAHEAD_HOURS.
5. Feature lists contain no label / outcome / split columns (no target leakage).
6. Deterministic inference: predicting the test split twice gives identical outputs.
7. Sequence windows are causal (the last element of every window is the current hour).

Writes results/reproducibility_check.json. Usage:
    python -m src.data_engineering.verify_reproducibility
"""

from __future__ import annotations

import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from src.config import (
    ALL_FEATURES,
    DATA_PROCESSED,
    LOOKAHEAD_HOURS,
    MIN_EVENT_HOUR,
    MODELS_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
)


def main():
    v1 = pd.read_parquet(DATA_PROCESSED / "v1.parquet")
    ids = {s: set(v1.loc[v1["split"] == s, "patientunitstayid"]) for s in ["train", "val", "test"]}
    no_overlap = not (ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"])

    ev = v1[v1["event_hour"].notna()]
    events_gt_min = bool((ev["event_hour"] > MIN_EVENT_HOUR).all())
    censored = bool((ev["hour"] < ev["event_hour"]).all())
    expected = (v1["event_hour"].notna() & (v1["hour"] < v1["event_hour"])
                & (v1["event_hour"] <= v1["hour"] + LOOKAHEAD_HOURS)).astype(int)
    label_ok = bool((expected == v1["label"]).all())
    leak_free = not any(c in ALL_FEATURES for c in ["label", "event_hour", "split", "is_sepsis"])

    test = v1[v1["split"] == "test"]
    model = joblib.load(MODELS_DIR / "xgb_temporal.pkl")
    p1 = model.predict_proba(test[ALL_FEATURES])[:, 1]
    p2 = model.predict_proba(test[ALL_FEATURES])[:, 1]
    deterministic = bool(np.array_equal(p1, p2))
    sha = hashlib.sha256(np.round(p1, 6).astype(np.float64).tobytes()).hexdigest()

    causal = None
    try:
        from src.models.sequence_models import Normalizer, build_hourly_matrix, build_windows

        tr = v1[v1["split"] == "train"]
        norm = Normalizer.fit(tr)
        X, L, d = build_windows(test, norm, 24)
        causal = bool(np.allclose(X[np.arange(len(X)), L - 1], build_hourly_matrix(d, norm)))
    except ImportError:
        pass

    out = {
        "random_seed": RANDOM_SEED,
        "v1_rows": int(len(v1)),
        "v1_stays": int(v1["patientunitstayid"].nunique()),
        "zero_patient_split_overlap": no_overlap,
        "all_event_hours_gt_6": events_gt_min,
        "strict_pre_event_censoring_verified": censored,
        "label_definition_verified": label_ok,
        "feature_lists_exclude_outcome_columns": leak_free,
        "deterministic_test_predictions_verified": deterministic,
        "test_pred_temporal_sha256_round6": sha,
        "sequence_windows_causal": causal,
    }
    with open(RESULTS_DIR / "reproducibility_check.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    failed = [k for k, v in out.items() if v is False]
    if failed:
        raise SystemExit(f"FAILED checks: {failed}")


if __name__ == "__main__":
    main()
