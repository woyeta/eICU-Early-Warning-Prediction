"""
Patient-level stratified train/val/test splitting and dataset freezing (v1.parquet).

Splits strictly by `patientunitstayid` (never by hourly row), stratified jointly by
`ever_deteriorated` and `is_sepsis`. Merges outcome labels into `v1_features.parquet`
and freezes `data/processed/v1.parquet`.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import (
    DATA_PROCESSED,
    DATA_RAW,
    LOOKAHEAD_HOURS,
    MIN_EVENT_HOUR,
    RANDOM_SEED,
    TRAIN_FRAC,
    VAL_FRAC,
)
from src.evaluation.outcome_labels import build_deterioration_labels


def build_patient_splits(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Split by PATIENT, not by row. All hours for a patient go into the same split.
    Stratify jointly by whether the patient ever had a deterioration event
    AND whether they are in the sepsis subgroup.
    """
    df = df.copy()
    patients = (
        df.groupby("patientunitstayid")
        .agg({"label": "max", "is_sepsis": "first"})
        .reset_index()
    )
    patients.columns = ["patientunitstayid", "ever_deteriorated", "is_sepsis"]

    # Joint stratification key ensures balanced representation of crashes and sepsis cases
    patients["strat_key"] = (
        patients["ever_deteriorated"].astype(str)
        + "_"
        + patients["is_sepsis"].astype(str)
    )

    train_pts, temp_pts = train_test_split(
        patients,
        test_size=1.0 - train_frac,
        stratify=patients["strat_key"],
        random_state=seed,
    )
    val_pts, test_pts = train_test_split(
        temp_pts,
        test_size=val_frac / (val_frac + (1.0 - train_frac - val_frac)),
        stratify=temp_pts["strat_key"],
        random_state=seed,
    )

    df["split"] = "train"
    df.loc[df.patientunitstayid.isin(val_pts.patientunitstayid), "split"] = "val"
    df.loc[df.patientunitstayid.isin(test_pts.patientunitstayid), "split"] = "test"

    # Programmatic leak-free verification
    train_ids = set(df[df.split == "train"].patientunitstayid)
    val_ids = set(df[df.split == "val"].patientunitstayid)
    test_ids = set(df[df.split == "test"].patientunitstayid)
    assert len(train_ids & val_ids) == 0, "Patient leak between train and val!"
    assert len(train_ids & test_ids) == 0, "Patient leak between train and test!"
    assert len(val_ids & test_ids) == 0, "Patient leak between val and test!"

    return df


def main():
    print("Loading v1_features.parquet and raw outcome tables...")
    v1_feat = pd.read_parquet(DATA_PROCESSED / "v1_features.parquet")
    patient_df = pd.read_csv(DATA_RAW / "patient.csv", low_memory=False)
    infusion_df = pd.read_csv(DATA_RAW / "infusionDrug.csv", low_memory=False)
    resp_df = pd.read_csv(DATA_RAW / "respiratoryCare.csv", low_memory=False)

    cohort_ids = v1_feat["patientunitstayid"].unique()
    labels_df = build_deterioration_labels(
        cohort_ids=cohort_ids,
        patient_df=patient_df,
        infusion_df=infusion_df,
        resp_df=resp_df,
        lookahead_hours=LOOKAHEAD_HOURS,
        min_event_hour=MIN_EVENT_HOUR,
    )

    v1 = pd.merge(
        v1_feat,
        labels_df,
        on=["patientunitstayid", "hour"],
        how="inner",
    )
    pos_pids = set(
        v1.groupby("patientunitstayid")["label"]
        .max()
        .loc[lambda s: s == 1]
        .index
    )
    v1.loc[~v1["patientunitstayid"].isin(pos_pids), "event_hour"] = np.nan
    v1 = v1.sort_values(["patientunitstayid", "hour"]).reset_index(drop=True)

    print("Applying joint patient-stratified train/val/test splits (70/15/15)...")
    v1 = build_patient_splits(v1, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, seed=RANDOM_SEED)

    # Verification assertions
    assert v1.label.notna().all(), "Missing labels found"
    assert (v1.event_hour.dropna() > 6).all(), "Found patient crashing at or before hour 6"
    assert (
        v1[v1.event_hour.notna()]["hour"] < v1[v1.event_hour.notna()]["event_hour"]
    ).all(), "Post-event uncensored rows found"

    out_path = DATA_PROCESSED / "v1.parquet"
    v1.to_parquet(out_path, index=False)
    print(f"FROZEN dataset saved to {out_path}")
    print(f"Total at-risk rows: {v1.shape[0]:,}")
    print(f"Total patients: {v1.patientunitstayid.nunique()}")
    print(f"Deteriorating patients: {v1.groupby('patientunitstayid')['label'].max().sum()}")
    print(f"Sepsis subgroup patients: {v1[v1.is_sepsis == True].patientunitstayid.nunique()}")
    print("Split breakdown:")
    for s in ["train", "val", "test"]:
        sub = v1[v1.split == s]
        n_pts = sub.patientunitstayid.nunique()
        n_det = sub.groupby("patientunitstayid")["label"].max().sum()
        n_sep = sub[sub.is_sepsis == True].patientunitstayid.nunique()
        print(
            f"  {s.upper()}: {n_pts} patients ({n_det} deteriorating, {n_sep} sepsis), "
            f"{len(sub):,} at-risk rows"
        )


if __name__ == "__main__":
    main()
