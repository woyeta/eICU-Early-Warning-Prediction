"""
Unit tests for NEWS2 baseline scorer including missing values and boundary thresholds.
"""

import numpy as np
import pandas as pd
from src.baselines.news2_scorer import apply_news2_to_dataset, score_news2


def test_normal_vitals_score_zero():
    assert score_news2(hr=75, rr=16, spo2=98, sbp=120, temp=37.0) == 0


def test_extreme_tachycardia_scores_3():
    assert score_news2(hr=135, rr=16, spo2=98, sbp=120, temp=37.0) == 3


def test_severe_hypotension_and_hypoxia():
    assert score_news2(hr=80, rr=16, spo2=89, sbp=85, temp=37.0) == 6


def test_moderate_tachypnea_and_fever():
    assert score_news2(hr=80, rr=22, spo2=97, sbp=125, temp=38.5) == 3


def test_critical_multi_organ_derangement_max_score():
    assert score_news2(hr=140, rr=28, spo2=88, sbp=80, temp=34.5) == 15


def test_missing_values_handled_gracefully():
    assert score_news2(hr=np.nan, rr=np.nan, spo2=np.nan, sbp=np.nan, temp=np.nan) == 0
    assert score_news2(hr=None, rr=16, spo2=np.nan, sbp=85, temp=None) == 3


def test_exact_boundary_thresholds():
    # RR boundaries: 8->3, 9->1, 11->1, 12->0, 20->0, 21->2, 24->2, 25->3
    assert score_news2(75, 8, 98, 120, 37.0) == 3
    assert score_news2(75, 9, 98, 120, 37.0) == 1
    assert score_news2(75, 11, 98, 120, 37.0) == 1
    assert score_news2(75, 12, 98, 120, 37.0) == 0
    assert score_news2(75, 20, 98, 120, 37.0) == 0
    assert score_news2(75, 21, 98, 120, 37.0) == 2
    assert score_news2(75, 24, 98, 120, 37.0) == 2
    assert score_news2(75, 25, 98, 120, 37.0) == 3

    # SBP boundaries: 90->3, 91->2, 100->2, 101->1, 110->1, 111->0, 219->0, 220->3
    assert score_news2(75, 16, 98, 90, 37.0) == 3
    assert score_news2(75, 16, 98, 91, 37.0) == 2
    assert score_news2(75, 16, 98, 100, 37.0) == 2
    assert score_news2(75, 16, 98, 101, 37.0) == 1
    assert score_news2(75, 16, 98, 110, 37.0) == 1
    assert score_news2(75, 16, 98, 111, 37.0) == 0
    assert score_news2(75, 16, 98, 219, 37.0) == 0
    assert score_news2(75, 16, 98, 220, 37.0) == 3


def test_apply_news2_to_dataset_matches_rowwise():
    df = pd.DataFrame(
        [
            {"heartrate": 75, "respiration": 16, "spo2": 98, "systemicsystolic": 120, "temperature": 37.0},
            {"heartrate": 135, "respiration": np.nan, "spo2": 90, "systemicsystolic": 88, "temperature": 39.2},
        ]
    )
    scored = apply_news2_to_dataset(df)
    assert scored["news2_score"].tolist() == [0, 3 + 0 + 3 + 3 + 2]
