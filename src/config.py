"""All shared constants, imported by every module, so no values are hardcoded."""

import pathlib

# Paths
PROJECT_ROOT = pathlib.Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Cohort definitions
PRIMARY_COHORT = "all_eligible_adult_icu"
SUBGROUP_SEPSIS = "sepsis"

# Cohort inclusion filters
MIN_AGE = 18
MIN_LOS_MINUTES = 2880          # 48 hours
FIRST_STAY_ONLY = True          # unitvisitnumber == 1
MIN_EVENT_HOUR = 6              # Exclude stays crashing at or before Hour 6 (need 6h trend window)

# Temporal alignment & missingness
BIN_SIZE_HOURS = 1              # 1-hour bins
VITAL_FORWARD_FILL_HOURS = 4    # Vitals carried forward up to 4 hours
LAB_FORWARD_FILL_HOURS = 24     # Labs carried forward up to 24 hours (daily lab cycle)

# Feature engineering
ROLLING_WINDOWS = [6]           # hours — rolling mean/std/min/max
SLOPE_WINDOW = 6                # hours — linear slope lookback

# Modeling & Evaluation
LOOKAHEAD_HOURS = 24            # predict deterioration within this forward window
CENSOR_POST_EVENT = True        # Censor patient hours after deterioration event onset
RANDOM_SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# Variables
VITAL_VARS = [
    "heartrate",
    "respiration",
    "spo2",
    "systemicsystolic",
    "systemicdiastolic",
    "systemicmean",
    "temperature",
]
LAB_VARS = [
    "lactate",
    "creatinine",
    "WBC x 1000",
    "platelets x 1000",
    "total bilirubin",
]
ALL_VARS = VITAL_VARS + LAB_VARS

# Missingness indicator flags (1 if missing after forward-fill, else 0)
MISSINGNESS_FLAGS = [f"miss_{var}" for var in ALL_VARS]

# Snapshot features (current values only — also used by the static baseline)
SNAPSHOT_FEATURES = VITAL_VARS + LAB_VARS + ["shock_index", "hour"]

# Temporal features (rolling stats, slopes, deltas)
TEMPORAL_FEATURES = []
for _var in ALL_VARS:
    TEMPORAL_FEATURES += [
        f"{_var}_roll_mean_6h",
        f"{_var}_roll_std_6h",
        f"{_var}_roll_min_6h",
        f"{_var}_roll_max_6h",
        f"{_var}_slope_6h",
        f"{_var}_delta_6h",
    ]

# All features for the main model (Note: never include 'label', 'event_hour', or 'split'!)
ALL_FEATURES = SNAPSHOT_FEATURES + TEMPORAL_FEATURES + MISSINGNESS_FLAGS

# Optimal decision thresholds selected on the validation set
NEWS2_THRESHOLD = 5
TEMPORAL_THRESHOLD = 0.7753



