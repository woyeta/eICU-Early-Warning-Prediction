"""
Temporal alignment, physiological outlier clipping, 1-hour median binning,
and differential forward-fill (4h vitals, 24h labs) for eICU-CRD Demo data.
"""

import numpy as np
import pandas as pd
from src.config import (
    ALL_VARS,
    BIN_SIZE_HOURS,
    DATA_PROCESSED,
    DATA_RAW,
    LAB_FORWARD_FILL_HOURS,
    LAB_VARS,
    VITAL_FORWARD_FILL_HOURS,
    VITAL_VARS,
)
from src.data_engineering.extract_cohort import extract_cohort


def _clean_and_clip_vitals(vitals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize vital sign columns and cap physiologically impossible values before binning:
    - HR: clip to [20, 300]
    - Respiration: clip to [4, 80]
    - SpO2 (from sao2/spo2): clip to [0, 100]
    - Temperature: convert Fahrenheit [77, 113] to Celsius, drop disconnected probe (<25),
      and clip to [25, 45]
    - SBP: clip to [30, 300]
    - DBP: clip to [15, 200]
    - MAP: clip to [20, 200]
    """
    df = vitals_df.copy()
    if "sao2" in df.columns and "spo2" not in df.columns:
        df = df.rename(columns={"sao2": "spo2"})

    for col in VITAL_VARS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "heartrate" in df.columns:
        df.loc[df["heartrate"] <= 0, "heartrate"] = np.nan
        df["heartrate"] = df["heartrate"].clip(20.0, 300.0)

    if "respiration" in df.columns:
        df.loc[df["respiration"] <= 0, "respiration"] = np.nan
        df["respiration"] = df["respiration"].clip(4.0, 80.0)

    if "spo2" in df.columns:
        df.loc[df["spo2"] <= 0, "spo2"] = np.nan
        df["spo2"] = df["spo2"].clip(0.0, 100.0)

    if "temperature" in df.columns:
        # Convert Fahrenheit readings [77, 113] to Celsius
        f_mask = (df["temperature"] >= 77.0) & (df["temperature"] <= 113.0)
        df.loc[f_mask, "temperature"] = (df.loc[f_mask, "temperature"] - 32.0) * (5.0 / 9.0)
        # Disconnected probe artifacts (< 20 C) set to NaN before clipping to [25, 45]
        df.loc[df["temperature"] < 20.0, "temperature"] = np.nan
        df["temperature"] = df["temperature"].clip(25.0, 45.0)

    if "systemicsystolic" in df.columns:
        df.loc[df["systemicsystolic"] <= 0, "systemicsystolic"] = np.nan
        df["systemicsystolic"] = df["systemicsystolic"].clip(30.0, 300.0)

    if "systemicdiastolic" in df.columns:
        df.loc[df["systemicdiastolic"] <= 0, "systemicdiastolic"] = np.nan
        df["systemicdiastolic"] = df["systemicdiastolic"].clip(15.0, 200.0)

    if "systemicmean" in df.columns:
        df.loc[df["systemicmean"] <= 0, "systemicmean"] = np.nan
        df["systemicmean"] = df["systemicmean"].clip(20.0, 200.0)

    return df


def bin_to_hourly(
    vitals_df: pd.DataFrame,
    labs_df: pd.DataFrame,
    cohort_ids,
    bin_size_hours: int = BIN_SIZE_HOURS,
    vital_ffill_hours: int = VITAL_FORWARD_FILL_HOURS,
    lab_ffill_hours: int = LAB_FORWARD_FILL_HOURS,
    aperiodic_df: pd.DataFrame = None,
    cohort_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    For each patient stay in cohort_ids:
    1. Filter vitals/labs to only those stays
    2. Convert observationoffset/labresultoffset from minutes to hours
    3. Assign each observation to an hourly bin (floor)
    4. Aggregate: median within each bin (robust to outliers), using non-invasive BP
       from vitalAperiodic as fallback when invasive systemic BP is missing
    5. Forward-fill gaps up to `vital_ffill_hours` (4h) for VITAL_VARS
       and up to `lab_ffill_hours` (24h) for LAB_VARS (matching clinical lab draw frequency)
    6. Add missingness flag (miss_<var> = 1 if still NaN after forward-fill, else 0)
    Returns: DataFrame with columns [patientunitstayid, hour, var1, var2, ..., miss_var1, ...]
    """
    cohort_id_set = set(cohort_ids)
    bin_minutes = 60.0 * bin_size_hours

    # 1. Process periodic vitals (restrict to >= 0 offset)
    vp = vitals_df[vitals_df["patientunitstayid"].isin(cohort_id_set)].copy()
    vp = vp[vp["observationoffset"] >= 0].copy()
    vp = _clean_and_clip_vitals(vp)
    vp["hour"] = np.floor(vp["observationoffset"] / bin_minutes).astype(int)

    vital_cols_present = [c for c in VITAL_VARS if c in vp.columns]
    vp_hourly = (
        vp.groupby(["patientunitstayid", "hour"], as_index=False)[vital_cols_present]
        .median()
    )

    # 2. Fallback to non-invasive blood pressure from vitalAperiodic when invasive BP is missing
    if aperiodic_df is not None and not aperiodic_df.empty:
        va = aperiodic_df[aperiodic_df["patientunitstayid"].isin(cohort_id_set)].copy()
        va = va[va["observationoffset"] >= 0].copy()
        for col, low, high in [
            ("noninvasivesystolic", 30.0, 300.0),
            ("noninvasivediastolic", 15.0, 200.0),
            ("noninvasivemean", 20.0, 200.0),
        ]:
            if col in va.columns:
                va[col] = pd.to_numeric(va[col], errors="coerce")
                va.loc[va[col] <= 0, col] = np.nan
                va[col] = va[col].clip(low, high)
        va["hour"] = np.floor(va["observationoffset"] / bin_minutes).astype(int)
        nibp_cols = [
            c
            for c in ["noninvasivesystolic", "noninvasivediastolic", "noninvasivemean"]
            if c in va.columns
        ]
        va_hourly = (
            va.groupby(["patientunitstayid", "hour"], as_index=False)[nibp_cols]
            .median()
        )
        vp_hourly = pd.merge(
            vp_hourly, va_hourly, on=["patientunitstayid", "hour"], how="outer"
        )
        for inv_col, nibp_col in [
            ("systemicsystolic", "noninvasivesystolic"),
            ("systemicdiastolic", "noninvasivediastolic"),
            ("systemicmean", "noninvasivemean"),
        ]:
            if inv_col in vp_hourly.columns and nibp_col in vp_hourly.columns:
                vp_hourly[inv_col] = vp_hourly[inv_col].fillna(vp_hourly[nibp_col])
                vp_hourly = vp_hourly.drop(columns=[nibp_col])

    # 3. Process laboratory results (allow pre-admission baseline labs within -6h mapped to hour 0)
    lb = labs_df[
        labs_df["patientunitstayid"].isin(cohort_id_set)
        & labs_df["labname"].isin(LAB_VARS)
    ].copy()
    lb["labresult"] = pd.to_numeric(lb["labresult"], errors="coerce")
    lb = lb[lb["labresult"].notna() & (lb["labresult"] > 0)].copy()
    lb = lb[lb["labresultoffset"] >= -360].copy()
    lb["hour"] = np.maximum(0, np.floor(lb["labresultoffset"] / bin_minutes).astype(int))

    # Clip extreme lab artifacts to physiologically plausible bounds
    lab_bounds = {
        "lactate": (0.1, 30.0),
        "creatinine": (0.1, 25.0),
        "WBC x 1000": (0.1, 200.0),
        "platelets x 1000": (1.0, 1500.0),
        "total bilirubin": (0.1, 60.0),
    }
    for lname, (lmin, lmax) in lab_bounds.items():
        mask = lb["labname"] == lname
        lb.loc[mask, "labresult"] = lb.loc[mask, "labresult"].clip(lmin, lmax)

    lb_hourly = (
        lb.groupby(["patientunitstayid", "hour", "labname"])["labresult"]
        .median()
        .unstack("labname")
        .reset_index()
    )
    for lcol in LAB_VARS:
        if lcol not in lb_hourly.columns:
            lb_hourly[lcol] = np.nan

    # 4. Build complete regular hourly grid per patient stay
    max_hours = {}
    if cohort_df is not None and "unitdischargeoffset" in cohort_df.columns:
        for row in cohort_df.itertuples():
            if row.patientunitstayid in cohort_id_set:
                max_hours[row.patientunitstayid] = max(
                    0, int(np.floor(row.unitdischargeoffset / bin_minutes))
                )
    obs_max_vp = vp_hourly.groupby("patientunitstayid")["hour"].max().to_dict()
    obs_max_lb = lb_hourly.groupby("patientunitstayid")["hour"].max().to_dict()

    grid_frames = []
    for pid in sorted(cohort_id_set):
        m_hr = max_hours.get(pid, max(obs_max_vp.get(pid, 0), obs_max_lb.get(pid, 0)))
        # Cap grid at discharge hour (or max observed vital/lab hour if shorter)
        obs_m = max(obs_max_vp.get(pid, 0), obs_max_lb.get(pid, 0))
        if obs_m > 0:
            m_hr = min(m_hr, obs_m) if m_hr > 0 else obs_m
        if m_hr <= 0:
            continue
        grid_frames.append(
            pd.DataFrame(
                {
                    "patientunitstayid": pid,
                    "hour": np.arange(0, m_hr + 1, dtype=int),
                }
            )
        )

    grid = pd.concat(grid_frames, ignore_index=True)
    merged = pd.merge(grid, vp_hourly, on=["patientunitstayid", "hour"], how="left")
    merged = pd.merge(
        merged,
        lb_hourly[["patientunitstayid", "hour"] + LAB_VARS],
        on=["patientunitstayid", "hour"],
        how="left",
    )
    merged = merged.sort_values(["patientunitstayid", "hour"]).reset_index(drop=True)

    # 5. Forward-fill vitals up to 4 hours and labs up to 24 hours per patient
    grp = merged.groupby("patientunitstayid", sort=False)
    for vcol in VITAL_VARS:
        if vcol not in merged.columns:
            merged[vcol] = np.nan
        merged[vcol] = grp[vcol].ffill(limit=vital_ffill_hours)

    for lcol in LAB_VARS:
        if lcol not in merged.columns:
            merged[lcol] = np.nan
        merged[lcol] = grp[lcol].ffill(limit=lab_ffill_hours)

    # 6. Add missingness indicator flags (miss_<var> = 1 if still NaN after forward-fill, else 0)
    for var in ALL_VARS:
        merged[f"miss_{var}"] = merged[var].isna().astype(int)

    # Attach is_sepsis if cohort_df provided
    if cohort_df is not None and "is_sepsis" in cohort_df.columns:
        sepsis_map = cohort_df.set_index("patientunitstayid")["is_sepsis"].to_dict()
        merged["is_sepsis"] = merged["patientunitstayid"].map(sepsis_map).fillna(False)

    return merged


def main():
    print("Loading raw eICU-CRD Demo tables...")
    patient_df = pd.read_csv(DATA_RAW / "patient.csv", low_memory=False)
    diagnosis_df = pd.read_csv(DATA_RAW / "diagnosis.csv", low_memory=False)
    cohort_df, attrition = extract_cohort(patient_df, diagnosis_df)

    vitals_df = pd.read_csv(
        DATA_RAW / "vitalPeriodic.csv",
        usecols=[
            "patientunitstayid",
            "observationoffset",
            "temperature",
            "sao2",
            "heartrate",
            "respiration",
            "systemicsystolic",
            "systemicdiastolic",
            "systemicmean",
        ],
        low_memory=False,
    )
    aperiodic_df = pd.read_csv(
        DATA_RAW / "vitalAperiodic.csv",
        usecols=[
            "patientunitstayid",
            "observationoffset",
            "noninvasivesystolic",
            "noninvasivediastolic",
            "noninvasivemean",
        ],
        low_memory=False,
    )
    labs_df = pd.read_csv(
        DATA_RAW / "lab.csv",
        usecols=["patientunitstayid", "labresultoffset", "labname", "labresult"],
        low_memory=False,
    )

    print("Aligning vitals and labs into 1-hour bins with differential forward-fill...")
    v0_df = bin_to_hourly(
        vitals_df=vitals_df,
        labs_df=labs_df,
        cohort_ids=cohort_df["patientunitstayid"].unique(),
        aperiodic_df=aperiodic_df,
        cohort_df=cohort_df,
    )

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED / "v0.parquet"
    v0_df.to_parquet(out_path, index=False)
    print(
        f"Saved {out_path}: {len(v0_df):,} hourly rows across "
        f"{v0_df['patientunitstayid'].nunique():,} stays "
        f"({v0_df[v0_df['is_sepsis']]['patientunitstayid'].nunique():,} sepsis stays)."
    )


if __name__ == "__main__":
    main()
