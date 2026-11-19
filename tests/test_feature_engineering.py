"""
Unit tests for causal 6-hour trajectory feature engineering.
"""

import numpy as np
import pandas as pd
from src.config import ALL_VARS
from src.data_engineering.feature_engineering import _slope, engineer_features


def test_slope_vectorized_matches_reference_with_nans():
    df = pd.DataFrame(
        {
            "patientunitstayid": [1] * 8 + [2] * 6,
            "hour": list(range(8)) + list(range(6)),
        }
    )
    for var in ALL_VARS:
        df[var] = np.nan
    df["heartrate"] = [
        70.0,
        np.nan,
        74.0,
        76.0,
        np.nan,
        80.0,
        82.0,
        84.0,
        100.0,
        95.0,
        90.0,
        np.nan,
        np.nan,
        75.0,
    ]
    df["systemicsystolic"] = 120.0

    feat = engineer_features(df)
    ref_slopes = (
        df.groupby("patientunitstayid")["heartrate"]
        .transform(lambda x: x.rolling(6, min_periods=2).apply(_slope, raw=False))
        .to_numpy()
    )
    vec_slopes = feat["heartrate_slope_6h"].to_numpy()

    assert np.allclose(ref_slopes, vec_slopes, equal_nan=True)


def test_unobserved_trends_remain_nan_never_zero():
    df = pd.DataFrame(
        {
            "patientunitstayid": [10, 10, 10],
            "hour": [0, 1, 2],
        }
    )
    for var in ALL_VARS:
        df[var] = np.nan
    df.loc[0, "heartrate"] = 80.0  # Only 1 observation -> std, slope, delta must be NaN

    feat = engineer_features(df)
    assert feat["heartrate_roll_std_6h"].isna().all()
    assert feat["heartrate_slope_6h"].isna().all()
    assert feat["heartrate_delta_6h"].isna().all()
