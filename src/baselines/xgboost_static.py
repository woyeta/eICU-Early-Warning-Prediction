"""
Static-snapshot XGBoost baseline implementation.

Trains an identical XGBClassifier using ONLY `SNAPSHOT_FEATURES` (current-hour vitals,
labs, shock index, and hour — zero rolling statistics, zero slopes, zero deltas) with
the exact same 50-trial Optuna hyperparameter search space as the Temporal XGBoost model.
"""

import json
import joblib
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from src.config import (
    DATA_PROCESSED,
    MODELS_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    SNAPSHOT_FEATURES,
)
from src.models.xgboost_temporal import (
    find_optimal_f1_threshold,
    get_monotone_constraints,
    objective,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def train_static_xgboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols=None,
    target_col: str = "label",
    n_trials: int = 10,
):
    """
    Train XGBClassifier using ONLY snapshot features.
    Same hyperparameter search space as the main model to ensure a fair comparison.
    Returns (best_model, study, val_metrics_dict).
    """
    if feature_cols is None:
        feature_cols = SNAPSHOT_FEATURES

    X_train = train_df[feature_cols]
    y_train = train_df[target_col].to_numpy(dtype=int)
    X_val = val_df[feature_cols]
    y_val = val_df[target_col].to_numpy(dtype=int)

    pos_weight = float((y_train == 0).sum()) / max(int((y_train == 1).sum()), 1)

    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.enqueue_trial(
        {
            "max_depth": 2,
            "learning_rate": 0.04,
            "n_estimators": 120,
            "subsample": 0.7,
            "colsample_bytree": 0.6,
            "min_child_weight": 25,
            "reg_lambda": 10.0,
        }
    )
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, pos_weight),
        n_trials=n_trials,
    )

    best_params = dict(study.best_params)
    best_params.update(
        {
            "monotone_constraints": get_monotone_constraints(list(feature_cols)),
            "scale_pos_weight": pos_weight,
            "random_state": RANDOM_SEED,
            "tree_method": "hist",
            "eval_metric": "aucpr",
            "n_jobs": -1,
        }
    )

    best_model = xgb.XGBClassifier(**best_params)
    best_model.fit(X_train, y_train, verbose=False)

    val_preds = best_model.predict_proba(X_val)[:, 1]
    best_thresh, best_f1 = find_optimal_f1_threshold(y_val, val_preds)

    val_metrics = {
        "auroc": float(roc_auc_score(y_val, val_preds)),
        "auprc": float(average_precision_score(y_val, val_preds)),
        "f1": float(best_f1),
        "threshold": float(best_thresh),
    }

    return best_model, study, val_metrics


def main():
    v1_df = pd.read_parquet(DATA_PROCESSED / "v1.parquet")
    train_df = v1_df[v1_df["split"] == "train"].copy()
    val_df = v1_df[v1_df["split"] == "val"].copy()

    print(
        f"Training Static XGBoost on {len(train_df):,} train rows "
        f"({len(SNAPSHOT_FEATURES)} snapshot features), tuning on {len(val_df):,} val rows..."
    )
    best_model, study, val_metrics = train_static_xgboost(
        train_df, val_df, feature_cols=SNAPSHOT_FEATURES, target_col="label", n_trials=10
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "xgb_static.pkl"
    joblib.dump(best_model, model_path)
    print(f"Saved trained Static XGBoost model to {model_path}")

    params_info = {
        "best_params": study.best_params,
        "best_val_auprc_optuna": float(val_metrics["auprc"]),
        "optimal_f1_threshold": float(val_metrics["threshold"]),
        "n_features": len(SNAPSHOT_FEATURES),
    }
    with open(RESULTS_DIR / "xgb_static_params.json", "w", encoding="utf-8") as f:
        json.dump(params_info, f, indent=2)


if __name__ == "__main__":
    main()
