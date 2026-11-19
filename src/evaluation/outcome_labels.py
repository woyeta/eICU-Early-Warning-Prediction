"""
Outcome labeling and strict pre-event censoring for the ICU Deterioration Early Warning System.

Identifies the earliest deterioration event hour per patient across:
1. New vasopressor initiation (infusionDrug: norepinephrine, epinephrine, dopamine,
   vasopressin, phenylephrine with drugrate > 0)
2. New mechanical ventilation (respiratoryCare: positive ventstartoffset or
   invasive airway Oral ETT / Tracheostomy)
3. ICU mortality (patient: unitdischargestatus == 'Expired')

Enforces the strict At-Risk & Early-Crash Exclusion Rule:
- Excludes patients with earliest_event <= min_event_hour (6h).
- Censors all hours h >= earliest_event.
- Labels pre-event hours h < earliest_event as 1 if h < earliest_event <= h + lookahead_hours else 0.
- Retains `event_hour` as a metadata helper column for lead-time calculation.
"""

import numpy as np
import pandas as pd
from src.config import (
    DATA_PROCESSED,
    DATA_RAW,
    LOOKAHEAD_HOURS,
    MIN_EVENT_HOUR,
)

VASOPRESSOR_PATTERN = "norepinephrine|epinephrine|dopamine|vasopressin|phenylephrine"


def _find_new_vasopressor_hours(pid: int, infusion_df: pd.DataFrame) -> list[int]:
    """Return sorted list of hourly offsets where vasopressor infusion (drugrate > 0) occurred."""
    sub = infusion_df[infusion_df["patientunitstayid"] == pid]
    if sub.empty:
        return []
    drug_match = sub["drugname"].str.contains(
        VASOPRESSOR_PATTERN, case=False, na=False
    )
    rates = pd.to_numeric(sub["drugrate"], errors="coerce")
    valid = sub[drug_match & (rates > 0)]
    if valid.empty:
        return []
    hours = np.floor(valid["infusionoffset"] / 60.0).astype(int).tolist()
    return sorted(set(hours))


def _find_new_ventilation_hours(pid: int, resp_df: pd.DataFrame) -> list[int]:
    """Return sorted list of hourly offsets where invasive mechanical ventilation started."""
    sub = resp_df[resp_df["patientunitstayid"] == pid]
    if sub.empty:
        return []
    vent_mask = (sub["ventstartoffset"] > 0) | (
        sub["airwaytype"].isin(["Oral ETT", "Nasal ETT", "Tracheostomy"])
    )
    valid = sub[vent_mask]
    if valid.empty:
        return []
    eff_offset = np.where(
        valid["ventstartoffset"] != 0,
        valid["ventstartoffset"],
        valid["respcarestatusoffset"],
    )
    hours = np.floor(eff_offset / 60.0).astype(int).tolist()
    return sorted(set(hours))


def _find_death_hour(pid: int, patient_df: pd.DataFrame):
    """Return integer hour of ICU death if unitdischargestatus == 'Expired', else None."""
    sub = patient_df[patient_df["patientunitstayid"] == pid]
    if sub.empty:
        return None
    row = sub.iloc[0]
    if str(row.get("unitdischargestatus", "")).strip().lower() == "expired":
        return int(np.floor(float(row["unitdischargeoffset"]) / 60.0))
    return None


def build_deterioration_labels(
    cohort_ids,
    patient_df: pd.DataFrame,
    infusion_df: pd.DataFrame,
    resp_df: pd.DataFrame,
    lookahead_hours: int = LOOKAHEAD_HOURS,
    min_event_hour: int = MIN_EVENT_HOUR,
) -> pd.DataFrame:
    """
    For each patient stay in cohort_ids:
    1. Identify earliest event hour: min(vasopressor_start, intubation, death).
    2. Early-crash / baseline exclusion: if earliest_event <= min_event_hour (6h),
       exclude the patient stay completely (already on life support or crashed
       before a 6-hour trend window could form).
    3. Strict at-risk censoring: hours >= earliest_event are CENSORED
       (dropped from at-risk prediction points).
    4. For pre-event hours (h < earliest_event):
       label = 1 if earliest_event <= h + lookahead_hours else 0.
    5. For stays with no event: all hours are at-risk, label = 0, event_hour = np.nan.
    6. Retain `event_hour` as a helper column for lead-time calculation.

    Returns DataFrame: [patientunitstayid, hour, label, event_hour] (only at-risk hours).
    """
    cohort_id_set = set(cohort_ids)
    p_sub = patient_df[patient_df["patientunitstayid"].isin(cohort_id_set)]
    inf_sub = infusion_df[infusion_df["patientunitstayid"].isin(cohort_id_set)]
    resp_sub = resp_df[resp_df["patientunitstayid"].isin(cohort_id_set)]

    max_hour_map = {
        int(row.patientunitstayid): max(0, int(np.floor(row.unitdischargeoffset / 60.0)))
        for row in p_sub.itertuples()
    }

    labels = []
    for pid in sorted(cohort_id_set):
        max_hour = max_hour_map.get(int(pid), 0)

        vaso_hours = _find_new_vasopressor_hours(pid, inf_sub)
        vent_hours = _find_new_ventilation_hours(pid, resp_sub)
        death_hour = _find_death_hour(pid, p_sub)
        all_events = sorted(
            set(
                vaso_hours
                + vent_hours
                + ([death_hour] if death_hour is not None else [])
            )
        )

        earliest_event = all_events[0] if len(all_events) > 0 else None

        # Exclude patients who crashed at or before Hour 6 (or prior to admission)
        if earliest_event is not None and earliest_event <= min_event_hour:
            continue

        for h in range(0, max_hour + 1):
            if earliest_event is not None and h >= earliest_event:
                # Patient already deteriorated — censor post-event hours
                continue
            has_event_ahead = earliest_event is not None and (
                h < earliest_event <= h + lookahead_hours
            )
            labels.append(
                {
                    "patientunitstayid": int(pid),
                    "hour": int(h),
                    "label": int(has_event_ahead),
                    "event_hour": float(earliest_event)
                    if earliest_event is not None
                    else np.nan,
                }
            )

    return pd.DataFrame(labels)


def main():
    print("Loading v0.parquet and raw event tables...")
    v0_df = pd.read_parquet(DATA_PROCESSED / "v0.parquet")
    patient_df = pd.read_csv(DATA_RAW / "patient.csv", low_memory=False)
    infusion_df = pd.read_csv(DATA_RAW / "infusionDrug.csv", low_memory=False)
    resp_df = pd.read_csv(DATA_RAW / "respiratoryCare.csv", low_memory=False)

    cohort_ids = v0_df["patientunitstayid"].unique()
    labels_df = build_deterioration_labels(
        cohort_ids=cohort_ids,
        patient_df=patient_df,
        infusion_df=infusion_df,
        resp_df=resp_df,
        lookahead_hours=LOOKAHEAD_HOURS,
        min_event_hour=MIN_EVENT_HOUR,
    )

    v0_labeled = pd.merge(
        v0_df,
        labels_df,
        on=["patientunitstayid", "hour"],
        how="inner",
    )
    # If a stay's observed vitals/labs ended >24h prior to earliest_event (no label==1 rows),
    # clear event_hour so event_hour.notna() matches ever_deteriorated (max(label) == 1)
    pos_pids = set(
        v0_labeled.groupby("patientunitstayid")["label"]
        .max()
        .loc[lambda s: s == 1]
        .index
    )
    v0_labeled.loc[~v0_labeled["patientunitstayid"].isin(pos_pids), "event_hour"] = np.nan
    v0_labeled = v0_labeled.sort_values(["patientunitstayid", "hour"]).reset_index(
        drop=True
    )

    out_path = DATA_PROCESSED / "v0_labeled.parquet"
    v0_labeled.to_parquet(out_path, index=False)

    print(f"Saved {out_path}")
    print("=== Primary Cohort (All Eligible Adult ICU) ===")
    print(f"At-risk rows: {v0_labeled.shape[0]:,}")
    print(f"Total patients: {v0_labeled.patientunitstayid.nunique()}")
    print(
        f"Deteriorating patients: "
        f"{v0_labeled.groupby('patientunitstayid')['label'].max().sum()}"
    )
    print(f"At-risk label balance: {v0_labeled.label.mean():.2%} positive")
    print("=== Secondary Sepsis Subgroup ===")
    sepsis = v0_labeled[v0_labeled.is_sepsis == True]
    print(f"Sepsis patients: {sepsis.patientunitstayid.nunique()}")
    print(
        f"Deteriorating sepsis patients: "
        f"{sepsis.groupby('patientunitstayid')['label'].max().sum()}"
    )
    print(f"Sepsis label balance: {sepsis.label.mean():.2%} positive")


if __name__ == "__main__":
    main()
