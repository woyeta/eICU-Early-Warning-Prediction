"""
Unit tests for the LSTM/GRU sequence-model extension and paired bootstrap.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.config import ALL_VARS, MISSINGNESS_FLAGS
from src.evaluation.metrics import paired_bootstrap_diff
from src.models.sequence_models import (
    SEQ_CHANNELS,
    Normalizer,
    RecurrentClassifier,
    build_hourly_matrix,
    build_windows,
)


def _toy_df(n_stays=3, hours=(5, 2, 8)):
    rows = []
    for pid, h in zip(range(1, n_stays + 1), hours):
        for t in range(h):
            r = {"patientunitstayid": pid, "hour": t, "label": 0, "shock_index": 0.7 + 0.01 * t}
            for v in ALL_VARS:
                r[v] = float(pid * 100 + t)  # value encodes (stay, hour)
            for m in MISSINGNESS_FLAGS:
                r[m] = 0
            rows.append(r)
    return pd.DataFrame(rows)


def test_windows_are_causal_and_within_stay():
    df = _toy_df()
    norm = Normalizer.fit(df)
    X, L, d = build_windows(df, norm, window=4)
    feats = build_hourly_matrix(d, norm)
    # last real slot of every window is the current hour
    assert np.allclose(X[np.arange(len(X)), L - 1], feats)
    # lengths grow from 1 to the window cap within each stay
    assert list(L[:5]) == [1, 2, 3, 4, 4]
    assert list(L[5:7]) == [1, 2]
    # padding beyond the real length is zero and no window mixes stays:
    # stay-2 hour-0 window must contain only its own row
    assert np.allclose(X[5, 1:], 0.0)
    hr_idx = SEQ_CHANNELS.index("heartrate")
    raw_hr = X[:, :, hr_idx] * norm.std["heartrate"] + norm.mean["heartrate"]
    for i in range(len(d)):
        vals = raw_hr[i, : L[i]]
        pid, t = d.loc[i, "patientunitstayid"], d.loc[i, "hour"]
        expected = [pid * 100 + h for h in range(t - L[i] + 1, t + 1)]
        assert np.allclose(vals, expected, atol=1e-3)


def test_missing_values_map_to_zero_after_scaling():
    df = _toy_df()
    df.loc[0, "lactate"] = np.nan
    norm = Normalizer.fit(df)
    feats = build_hourly_matrix(df, norm)
    assert feats[0, SEQ_CHANNELS.index("lactate")] == 0.0
    assert np.isfinite(feats).all()


def test_recurrent_classifier_forward_shapes():
    for cell in ["lstm", "gru"]:
        model = RecurrentClassifier(n_features=len(SEQ_CHANNELS), cell=cell, hidden=8)
        x = torch.randn(5, 6, len(SEQ_CHANNELS))
        lengths = torch.tensor([6, 1, 3, 6, 2])
        out = model(x, lengths)
        assert out.shape == (5,)


def test_padding_does_not_change_prediction():
    torch.manual_seed(0)
    model = RecurrentClassifier(n_features=len(SEQ_CHANNELS), cell="gru", hidden=8).eval()
    x = torch.randn(1, 3, len(SEQ_CHANNELS))
    padded = torch.cat([x, torch.randn(1, 5, len(SEQ_CHANNELS))], dim=1)  # garbage after length
    with torch.no_grad():
        a = model(x, torch.tensor([3]))
        b = model(padded, torch.tensor([3]))
    assert torch.allclose(a, b, atol=1e-6)


def test_paired_bootstrap_identical_models_zero_diff():
    rng = np.random.RandomState(0)
    pids = np.repeat(np.arange(40), 5)
    y = (rng.rand(200) < 0.2).astype(int)
    s = rng.rand(200)
    r = paired_bootstrap_diff(y, s, s, pids, roc_auc_score, n_bootstraps=100)
    assert r["diff"] == 0.0 and r["ci_lower"] == 0.0 and r["ci_upper"] == 0.0


def test_platt_scaling_recovers_parameters_and_preserves_ranking():
    from src.evaluation.calibrate import apply_platt, calibration_slope_intercept, fit_platt

    rng = np.random.RandomState(1)
    z = rng.normal(size=20000)
    true_p = 1 / (1 + np.exp(-(-3.0 + 0.8 * z)))
    y = (rng.rand(len(z)) < true_p).astype(int)
    raw = 1 / (1 + np.exp(-z))  # mis-scaled "model output" whose logit is z
    params = fit_platt("XGB_temporal", raw, y)
    assert abs(params["a"] + 3.0) < 0.15 and abs(params["b"] - 0.8) < 0.1
    cal = apply_platt("XGB_temporal", raw, params)
    assert roc_auc_score(y, cal) == roc_auc_score(y, raw)  # monotone: ranking unchanged
    slope, citl = calibration_slope_intercept(y, cal)
    assert abs(slope - 1.0) < 0.1 and abs(citl) < 0.1
