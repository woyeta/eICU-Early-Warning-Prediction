"""
Unit tests for warning lead-time calculation and false-alarm burden.
"""

import numpy as np
import pandas as pd
from src.evaluation.lead_time import compute_lead_times, summarize_lead_times
from src.evaluation.visualize import compute_alarm_burden


def test_compute_lead_times_uses_true_event_hour():
    # Patient 10 crashes at hour 30 (hours 0..29 in dataset)
    # First alert at hour 18 -> lead_time_hours = 30 - 18 = 12.0h
    df = pd.DataFrame(
        {
            "patientunitstayid": [10, 10, 10, 10, 20, 20],
            "hour": [10, 18, 22, 29, 5, 10],
            "label": [0, 1, 1, 1, 0, 0],
            "event_hour": [30.0, 30.0, 30.0, 30.0, np.nan, np.nan],
            "pred_prob": [0.20, 0.65, 0.75, 0.85, 0.70, 0.10],
        }
    )
    lt = compute_lead_times(df, threshold=0.50, max_lookback_hours=48)
    assert len(lt) == 1
    assert lt.iloc[0]["patientunitstayid"] == 10
    assert lt.iloc[0]["alert_hour"] == 18.0
    assert lt.iloc[0]["event_hour"] == 30.0
    assert lt.iloc[0]["lead_time_hours"] == 12.0

    summary = summarize_lead_times(df, threshold=0.50, max_lookback_hours=48)
    assert summary["n_deteriorating"] == 1
    assert summary["n_alerted"] == 1
    assert summary["median_lead_time_h"] == 12.0


def test_compute_alarm_burden_per_patient_day():
    # 24 at-risk rows (1 patient-day), 3 false alarms (pred_prob > 0.5 and label == 0)
    df = pd.DataFrame(
        {
            "label": [0] * 20 + [1] * 4,
            "pred_prob": [0.8, 0.7, 0.6] + [0.1] * 17 + [0.9] * 4,
        }
    )
    burden = compute_alarm_burden(df, threshold=0.5)
    assert np.isclose(burden, 3.0)
