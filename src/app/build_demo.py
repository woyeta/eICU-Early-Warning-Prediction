"""
Build the demo interface: demo/index.html, one self-contained page that opens
offline in any browser and also works on GitHub Pages.

Packs the held-out test patients (hourly vitals, NEWS2 score and the calibrated
risk of every model) plus the summary results into one JSON blob and injects it
into demo/template.html. The output opens in any browser with no server.

Alert thresholds are the equal-sensitivity operating points (each model flags
the same share of pre-crash hours as NEWS2 >= 5 on validation), expressed as
calibrated risk, so the models are compared on a fair footing.

Run from the project root:  python -m src.app.build_demo
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines.news2_scorer import apply_news2_to_dataset

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
APP = ROOT / "demo"

MODELS = [
    ("GRU", "GRU_cal", "GRU", "Recurrent network, last 6 h"),
    ("LSTM", "LSTM_cal", "LSTM", "Recurrent network, last 48 h"),
    ("XGB_temporal", "XGB_temporal_cal", "Temporal XGBoost", "Trees on 6 h trend features"),
    ("XGB_static", "XGB_static_cal", "Static XGBoost", "Trees on the current hour only"),
    ("NEWS2", "NEWS2_cal", "NEWS2", "Bedside score, alert at 5+"),
]
VITALS = [
    ("heartrate", "hr"), ("systemicmean", "map"), ("spo2", "spo2"),
    ("respiration", "rr"), ("systemicsystolic", "sbp"), ("temperature", "temp"),
]
LOOKBACK = 48


def r(x, nd):
    return None if pd.isna(x) else round(float(x), nd)


def detection(preds, events, col, thr):
    """First alert within the 48 h before the event, per deteriorating patient."""
    leads = []
    for pid, ev in events.items():
        sub = preds[(preds.patientunitstayid == pid) & (preds.hour >= ev - LOOKBACK) & (preds.hour < ev)]
        hit = sub[sub[col] >= thr]
        if len(hit):
            leads.append(ev - int(hit.hour.min()))
    return len(leads), (float(np.median(leads)) if leads else None)


def main():
    preds = pd.read_parquet(RES / "calibrated_test_predictions.parquet")
    v0 = pd.read_parquet(ROOT / "data/processed/v0_labeled.parquet")
    cohort = pd.read_parquet(ROOT / "data/interim/cohort.parquet")
    thr = json.loads((RES / "all_models_thresholds.json").read_text())
    metrics = pd.read_csv(RES / "all_models_test_results.csv")
    alarms = pd.read_csv(RES / "alarm_burden_equal_sensitivity.csv")
    shap = pd.read_csv(RES / "shap_top_features.csv")

    test_ids = preds.patientunitstayid.unique()
    v = v0[v0.patientunitstayid.isin(test_ids)]
    v = apply_news2_to_dataset(v)
    df = preds.merge(
        v[["patientunitstayid", "hour", "event_hour", "news2_score"] + [c for c, _ in VITALS]],
        on=["patientunitstayid", "hour"], how="left",
    ).sort_values(["patientunitstayid", "hour"])

    thresholds = {k: thr[k]["matched_sens_as_calibrated_risk"] for k, *_ in MODELS}
    events = df.groupby("patientunitstayid").event_hour.first().dropna().astype(int).to_dict()

    cinfo = cohort.set_index("patientunitstayid")
    patients = []
    for pid, g in df.groupby("patientunitstayid"):
        c = cinfo.loc[pid]
        p = {
            "id": int(pid),
            "age": str(c["age"]),
            "sex": str(c["gender"])[:1] if pd.notna(c["gender"]) else "?",
            "dx": str(c["apacheadmissiondx"]) if pd.notna(c["apacheadmissiondx"]) else "Not recorded",
            "unit": str(c["unittype"]),
            "hospital": int(c["hospitalid"]),
            "sepsis": bool(g.is_sepsis.iloc[0]),
            "event": events.get(pid),
            "h0": int(g.hour.min()),
            "news2": [None if pd.isna(x) else int(x) for x in g.news2_score],
            "risk": {k: [r(x, 4) for x in g[col]] for k, col, *_ in MODELS},
            "vit": {short: [r(x, 1) for x in g[col]] for col, short in VITALS},
        }
        assert (g.hour.diff().dropna() == 1).all(), f"gap in hours for {pid}"
        patients.append(p)

    prim = metrics[metrics.cohort == "Primary_Cohort"].set_index("model")
    al = alarms[alarms.cohort == "Primary_Cohort"].set_index("model")
    models = []
    for k, col, name, desc in MODELS:
        n_det, med = detection(df, events, col, thresholds[k])
        m = prim.loc[k]
        models.append({
            "key": k, "name": name, "desc": desc, "thr": round(thresholds[k], 4),
            "auroc": [m.auroc, m.auroc_ci_lower, m.auroc_ci_upper],
            "auprc": [m.auprc, m.auprc_ci_lower, m.auprc_ci_upper],
            "fa_day": float(al.loc[k, "false_alarms_per_patient_day_at_target"]),
            "warned": n_det, "lead_med": med,
        })

    data = {
        "patients": patients,
        "models": models,
        "n_events": len(events),
        "shap": [
            {"f": row.feature, "v": round(float(row.mean_abs_shap_primary), 4), "trend": bool(row.is_temporal_6h_feature)}
            for row in shap.sort_values("rank_primary").head(8).itertuples()
        ],
        "target_sens": float(al["target_hour_sensitivity_pct"].iloc[0]),
    }

    blob = json.dumps(data, separators=(",", ":"), allow_nan=False)
    html = (APP / "template.html").read_text(encoding="utf-8")
    assert "/*__DATA__*/null" in html
    out = APP / "index.html"
    out.write_text(html.replace("/*__DATA__*/null", blob), encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size / 1024:.0f} KB, {len(patients)} patients)")

    for m in models:
        print(f"  {m['name']:<17} thr={m['thr']:.4f} warned={m['warned']}/{len(events)} median lead={m['lead_med']}")


if __name__ == "__main__":
    main()
