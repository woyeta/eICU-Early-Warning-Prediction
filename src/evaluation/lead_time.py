"""
Clinical warning lead-time analysis for deteriorating ICU patients.

Measures exact hours between the model's first pre-event alert (`pred_prob > threshold`)
and the true clinical deterioration event (`event_hour`), evaluated both within the
48-hour pre-event window (`max_lookback_hours=48`) and across the overall stay.
"""

import numpy as np
import pandas as pd


def compute_lead_times(
    predictions_df: pd.DataFrame,
    threshold: float,
    max_lookback_hours=48,
    inclusive_threshold: bool = False,
) -> pd.DataFrame:
    """
    For each patient who actually deteriorated (event_hour is not NaN):
    1. Read the true deterioration hour `true_event_hour = group['event_hour'].iloc[0]`
       (Note: do NOT use `group[group.label == 1].hour.min()`, because label == 1
       starts 24h before the crash and the crash hour itself is censored out!)
    2. Look at pre-event hours (`hour < true_event_hour`). If `max_lookback_hours`
       is set (e.g. 48h), restrict to `[true_event_hour - max_lookback_hours, true_event_hour)`
       so random false alarms many days prior are not miscounted as warning lead time.
    3. Find the earliest hour where `pred_prob > threshold` (or `>= threshold` for integer scores).
    4. Compute `lead_time_hours = true_event_hour - first_alert_hour`.

    Returns DataFrame with [patientunitstayid, alert_hour, event_hour, lead_time_hours]
    """
    results = []
    for pid, group in predictions_df.groupby("patientunitstayid"):
        if group["event_hour"].isna().all():
            continue  # Patient never deteriorated — skip

        true_event_hour = float(group["event_hour"].dropna().iloc[0])

        # Only look at predictions strictly BEFORE the true crash hour
        pre_event = group[group["hour"] < true_event_hour]
        if max_lookback_hours is not None:
            pre_event = pre_event[
                pre_event["hour"] >= (true_event_hour - max_lookback_hours)
            ]

        if inclusive_threshold:
            alerts = pre_event[pre_event["pred_prob"] >= threshold]
        else:
            alerts = pre_event[pre_event["pred_prob"] > threshold]

        if alerts.empty:
            continue  # Model didn't alert in the pre-event window — missed

        first_alert_hour = float(alerts["hour"].min())
        lead_time = true_event_hour - first_alert_hour

        results.append(
            {
                "patientunitstayid": int(pid),
                "alert_hour": first_alert_hour,
                "event_hour": true_event_hour,
                "lead_time_hours": float(lead_time),
            }
        )

    return pd.DataFrame(
        results,
        columns=["patientunitstayid", "alert_hour", "event_hour", "lead_time_hours"],
    )


def summarize_lead_times(
    predictions_df: pd.DataFrame,
    threshold: float,
    max_lookback_hours=48,
    inclusive_threshold: bool = False,
) -> dict:
    """
    Summarize warning lead-time distribution across all deteriorating patients in predictions_df.
    """
    det_pids = (
        predictions_df[predictions_df["event_hour"].notna()]["patientunitstayid"]
        .unique()
    )
    n_det = int(len(det_pids))
    lt_df = compute_lead_times(
        predictions_df,
        threshold=threshold,
        max_lookback_hours=max_lookback_hours,
        inclusive_threshold=inclusive_threshold,
    )
    n_caught = int(len(lt_df))
    if n_caught == 0 or n_det == 0:
        return {
            "n_deteriorating": n_det,
            "n_alerted": 0,
            "patient_sensitivity_pct": 0.0,
            "median_lead_time_h": 0.0,
            "iqr_25_h": 0.0,
            "iqr_75_h": 0.0,
            "pct_det_alerted_ge_6h": 0.0,
            "pct_det_alerted_ge_12h": 0.0,
        }

    lts = lt_df["lead_time_hours"].to_numpy(dtype=float)
    return {
        "n_deteriorating": n_det,
        "n_alerted": n_caught,
        "patient_sensitivity_pct": round(100.0 * n_caught / n_det, 1),
        "median_lead_time_h": round(float(np.median(lts)), 2),
        "iqr_25_h": round(float(np.percentile(lts, 25)), 2),
        "iqr_75_h": round(float(np.percentile(lts, 75)), 2),
        "pct_det_alerted_ge_6h": round(100.0 * float((lts >= 6.0).sum()) / n_det, 1),
        "pct_det_alerted_ge_12h": round(100.0 * float((lts >= 12.0).sum()) / n_det, 1),
    }
