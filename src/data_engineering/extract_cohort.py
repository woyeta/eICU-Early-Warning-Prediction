"""
Cohort extraction for the ICU Deterioration Early Warning System.

Extracts the Primary Cohort (all eligible adult ICU stays: age >= 18, first stay,
LOS >= 48h) and flags the Secondary Sepsis Subgroup ('is_sepsis').
"""

import argparse
import pandas as pd
from src.config import (
    DATA_INTERIM,
    DATA_RAW,
    FIRST_STAY_ONLY,
    MIN_AGE,
    MIN_LOS_MINUTES,
)


def extract_cohort(patient_df: pd.DataFrame, diagnosis_df: pd.DataFrame):
    """
    Extracts Primary Cohort (all eligible adult ICU stays) and flags the Sepsis Subgroup.
    Logs attrition at each filtering step.
    Returns: filtered patient DataFrame (with 'is_sepsis' column) + attrition dict.
    """
    attrition = {}
    attrition["total_demo_stays"] = int(patient_df.patientunitstayid.nunique())

    # Filter 1: Age >= 18 (safely handle '> 89' and NaN)
    patient_df = patient_df.copy()
    patient_df["age_num"] = pd.to_numeric(
        patient_df.age.astype(str).str.replace("> 89", "89", regex=False),
        errors="coerce",
    )
    cohort = patient_df[patient_df.age_num >= MIN_AGE].copy()
    attrition["adult_stays"] = int(cohort.patientunitstayid.nunique())

    # Filter 2: First ICU stay only
    if FIRST_STAY_ONLY:
        cohort = cohort[cohort.unitvisitnumber == 1].copy()
    attrition["first_stay_adults"] = int(cohort.patientunitstayid.nunique())

    # Filter 3: LOS >= 48h (unitdischargeoffset is in minutes)
    cohort = cohort[cohort.unitdischargeoffset >= MIN_LOS_MINUTES].copy()
    attrition["los_48h_primary"] = int(cohort.patientunitstayid.nunique())

    # Flag secondary sepsis subgroup
    sepsis_ids = diagnosis_df[
        diagnosis_df.diagnosisstring.str.contains(
            "sepsis|septic shock", case=False, na=False
        )
    ].patientunitstayid.unique()
    cohort["is_sepsis"] = cohort.patientunitstayid.isin(sepsis_ids)
    attrition["sepsis_subgroup"] = int(cohort["is_sepsis"].sum())

    return cohort, attrition


def main():
    parser = argparse.ArgumentParser(description="Extract eICU-CRD Demo cohort.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run extraction and print attrition counts without writing output files.",
    )
    args = parser.parse_args()

    patient_df = pd.read_csv(DATA_RAW / "patient.csv", low_memory=False)
    diagnosis_df = pd.read_csv(DATA_RAW / "diagnosis.csv", low_memory=False)

    cohort, attrition = extract_cohort(patient_df, diagnosis_df)

    print("=== Cohort Attrition Summary ===")
    for step, count in attrition.items():
        print(f"  {step:22s}: {count:,}")

    if not args.dry_run:
        DATA_INTERIM.mkdir(parents=True, exist_ok=True)
        out_path = DATA_INTERIM / "cohort.parquet"
        cohort.to_parquet(out_path, index=False)
        print(f"Saved cohort to {out_path}")


if __name__ == "__main__":
    main()
