"""
Patient-level cluster bootstrap evaluation metrics.

Resamples unique patient IDs (`patientunitstayid`) with replacement so all
correlated hourly observations from a patient are sampled together as a cluster,
preventing artificially narrow confidence intervals.
"""

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def bootstrap_metric(
    y_true,
    y_score,
    patient_ids,
    metric_fn,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Compute a metric with patient-level cluster bootstrap 95% CI.
    Resamples unique patient IDs (patientunitstayid) with replacement and pulls
    all hourly rows for the sampled patients, avoiding artificial narrowing of CIs
    caused by treating correlated hourly rows from the same patient as independent.
    """
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    patient_ids = np.asarray(patient_ids)

    unique_pids = np.unique(patient_ids)
    pid_to_indices = {pid: np.where(patient_ids == pid)[0] for pid in unique_pids}

    scores = []
    for _ in range(n_bootstraps):
        sampled_pids = rng.choice(unique_pids, size=len(unique_pids), replace=True)
        idx = np.concatenate([pid_to_indices[pid] for pid in sampled_pids])
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(float(metric_fn(y_true[idx], y_score[idx])))

    lower = float(np.percentile(scores, 2.5))
    upper = float(np.percentile(scores, 97.5))
    point = float(metric_fn(y_true, y_score))
    return {"point": point, "ci_lower": lower, "ci_upper": upper}


def paired_bootstrap_diff(
    y_true,
    score_a,
    score_b,
    patient_ids,
    metric_fn,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Paired patient-level cluster bootstrap for the difference metric(A) - metric(B).

    Both models are evaluated on the *same* resampled patients in every replicate,
    so the CI reflects uncertainty in the difference itself (much tighter and more
    honest than comparing two marginal CIs for overlap). The two-sided p-value is
    the bootstrap proportion of replicates on the opposite side of zero, doubled.
    """
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)
    patient_ids = np.asarray(patient_ids)

    unique_pids = np.unique(patient_ids)
    pid_to_indices = {pid: np.where(patient_ids == pid)[0] for pid in unique_pids}

    diffs = []
    for _ in range(n_bootstraps):
        sampled = rng.choice(unique_pids, size=len(unique_pids), replace=True)
        idx = np.concatenate([pid_to_indices[pid] for pid in sampled])
        if len(np.unique(y_true[idx])) < 2:
            continue
        diffs.append(float(metric_fn(y_true[idx], score_a[idx]) - metric_fn(y_true[idx], score_b[idx])))
    diffs = np.asarray(diffs)
    point = float(metric_fn(y_true, score_a) - metric_fn(y_true, score_b))
    p_two_sided = float(min(1.0, 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())))
    return {
        "diff": point,
        "ci_lower": float(np.percentile(diffs, 2.5)),
        "ci_upper": float(np.percentile(diffs, 97.5)),
        "p_value": p_two_sided,
    }
