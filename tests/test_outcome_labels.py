"""
Unit tests for deterioration outcome labeling and pre-event censoring.
"""

import numpy as np
import pandas as pd
from src.evaluation.outcome_labels import build_deterioration_labels


def test_excludes_early_crash_at_or_before_hour_6():
    patient_df = pd.DataFrame(
        [
            {
                "patientunitstayid": 101,
                "unitdischargeoffset": 3600,
                "unitdischargestatus": "Alive",
            }
        ]
    )
    # Vasopressor started at minute 240 (hour 4 <= 6) -> must be excluded
    infusion_df = pd.DataFrame(
        [
            {
                "patientunitstayid": 101,
                "infusionoffset": 240,
                "drugname": "Norepinephrine (mcg/min)",
                "drugrate": "5.0",
            }
        ]
    )
    resp_df = pd.DataFrame(
        columns=["patientunitstayid", "ventstartoffset", "respcarestatusoffset", "airwaytype"]
    )

    out = build_deterioration_labels(
        [101], patient_df, infusion_df, resp_df, lookahead_hours=24, min_event_hour=6
    )
    assert out.empty


def test_censors_post_event_hours_and_labels_24h_window():
    patient_df = pd.DataFrame(
        [
            {
                "patientunitstayid": 202,
                "unitdischargeoffset": 3000,  # 50 hours
                "unitdischargestatus": "Alive",
            }
        ]
    )
    # Vasopressor started at minute 1800 (hour 30 > 6)
    infusion_df = pd.DataFrame(
        [
            {
                "patientunitstayid": 202,
                "infusionoffset": 1800,
                "drugname": "Norepinephrine",
                "drugrate": "8.0",
            }
        ]
    )
    resp_df = pd.DataFrame(
        columns=["patientunitstayid", "ventstartoffset", "respcarestatusoffset", "airwaytype"]
    )

    out = build_deterioration_labels(
        [202], patient_df, infusion_df, resp_df, lookahead_hours=24, min_event_hour=6
    )
    # Hours 0..29 should be present; hours >= 30 censored
    assert out["hour"].max() == 29
    assert len(out) == 30
    assert (out["event_hour"] == 30.0).all()
    # Hours 0..5 (>24h before hour 30) should have label 0
    assert (out[out["hour"] <= 5]["label"] == 0).all()
    # Hours 6..29 (within 24h before hour 30) should have label 1
    assert (out[out["hour"] >= 6]["label"] == 1).all()


def test_stable_patient_all_zero_labels():
    patient_df = pd.DataFrame(
        [
            {
                "patientunitstayid": 303,
                "unitdischargeoffset": 1200,  # 20 hours
                "unitdischargestatus": "Alive",
            }
        ]
    )
    infusion_df = pd.DataFrame(
        columns=["patientunitstayid", "infusionoffset", "drugname", "drugrate"]
    )
    resp_df = pd.DataFrame(
        columns=["patientunitstayid", "ventstartoffset", "respcarestatusoffset", "airwaytype"]
    )

    out = build_deterioration_labels(
        [303], patient_df, infusion_df, resp_df, lookahead_hours=24, min_event_hour=6
    )
    assert len(out) == 21
    assert (out["label"] == 0).all()
    assert out["event_hour"].isna().all()
