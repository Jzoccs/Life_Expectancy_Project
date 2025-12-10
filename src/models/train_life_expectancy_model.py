"""
Train regression models to predict life expectancy using:
- A simple Linear Regression baseline
- An AutoML model from FLAML (tree-based regressors)

This script expects the cleaned dataset:
    data/processed/cleaned_life_expectancy_gdp_co2.csv

Steps:
1. Load data
2. Basic checks
3. Prepare features & target
4. Train/test split
5. Train baseline Linear Regression
6. Run FLAML AutoML for regression
7. Evaluate and compare models
8. Save test predictions to CSV

To run:
    python src/models/train_life_expectancy_automl.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from flaml import AutoML
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------- Data loading & preparation ----------


def load_data() -> pd.DataFrame:
    """
    Load the cleaned dataset from data/processed.

    Returns
    -------
    df : pd.DataFrame
        The cleaned modeling dataset.
    """
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "processed" / "cleaned_life_expectancy_gdp_co2.csv"

    df = pd.read_csv(data_path)
    print(f"Loaded data from {data_path}")
    print("Shape:", df.shape)
    return df


def basic_checks(df: pd.DataFrame) -> None:
    """
    Print some basic info and sanity checks.
    """
    print("\n=== BASIC INFO ===")
    print(df.info())

    print("\n=== SUMMARY STATS (numeric) ===")
    print(df.describe())

    print("\nMissing values per column:")
    print(df.isna().sum())


def prepare_features_and_target(df: pd.DataFrame):
    """
    Select feature columns (X) and target (y) from the DataFrame.

    For now, we use:
        - gdp_per_capita
        - co2_per_capita
        - year

    and predict:
        - life_expectancy
    """
    target_variable = "life_expectancy"
    feature_variables = ["gdp_per_capita", "co2_per_capita", "year"]

    # Keep only rows without missing values in X or y
    df_model = df.dropna(subset=feature_variables + [target_variable]).copy()

    X = df_model[feature_variables]
    y = df_model[target_variable]

    print("\n=== MODELING DATA ===")
    print("Features:", feature_variables)
    print("Target:", target_variable)
    print("Shape X:", X.shape, "Shape y:", y.shape)

    return X, y, df_model


def train_test_split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into train and test sets.

    Parameters
    ----------
    test_size : float
        Fraction of data to use for testing.
    random_state : int
        Seed for reproducibility.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    print("\n=== TRAIN / TEST SPLIT ===")
    print("X_train:", X_train.shape, "X_test:", X_test.shape)
    return X_train, X_test, y_train, y_test


# ---------- Evaluation helper ----------


def evaluate_model(name, y_test, y_pred):
    """
    Print evaluation metrics for a regression model.
    """
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n=== {name} RESULTS ===")
    print(f"RMSE: {rmse:0.3f}")
    print(f"MAE : {mae:0.3f}")
    print(f"R^2 : {r2:0.3f}")

    return {"rmse": rmse, "mae": mae, "r2": r2}



# ---------- AutoML with FLAML ----------


def train_automl_model(X_train, X_test, y_train, y_test, time_budget_seconds=60):
    """
    Use FLAML AutoML to search for a good regression model.

    Parameters
    ----------
    time_budget_seconds : int
        Total time budget for AutoML search in seconds.
    """
    automl = AutoML()

    automl_settings = {
        "time_budget": time_budget_seconds,
        "metric": "r2",          # optimize for R^2
        "task": "regression",
        "log_file_name": "automl_life_expectancy.log",
        # You can adjust this list; keeping it small for speed/clarity
        "estimator_list": ["lgbm", "rf", "xgboost"],
        "seed": 42,
    }

    print("\n=== RUNNING FLAML AUTOML ===")
    automl.fit(
        X_train=X_train,
        y_train=y_train,
        **automl_settings,
    )

    print("\nBest estimator type:", automl.best_estimator)
    print("Best config:", automl.best_config)
    print("Best validation loss (1 - R^2):", automl.best_loss)
    approx_val_r2 = 1 - automl.best_loss
    print(f"Approx validation R^2: {approx_val_r2:0.3f}")

    # Evaluate on test set
    y_pred_automl = automl.predict(X_test)
    automl_results = evaluate_model("FLAML AutoML", y_test, y_pred_automl)

    return automl, y_pred_automl, automl_results


# ---------- Save predictions ----------


def save_test_predictions(
    df_model,
    X_test,
    y_test,
    y_pred_automl,
):
    """
    Save a CSV containing the test rows with predictions from both models.

    This is useful for your report and to let your instructor inspect predictions.
    """
    project_root = Path(__file__).resolve().parents[2]
    output_path = project_root / "data" / "processed" / "test_predictions_life_expectancy_automl.csv"

    preds_df = df_model.loc[X_test.index].copy()
    preds_df["y_actual"] = y_test
    preds_df["y_pred_automl"] = y_pred_automl

    preds_df.to_csv(output_path, index=False)
    print(f"\nSaved test predictions to {output_path}")


# ---------- Main script ----------


def main():
    # 1. Load data
    df = load_data()

    # 2. Basic checks (good for assignment write-up)
    basic_checks(df)

    # 3. Prepare features and target
    X, y, df_model = prepare_features_and_target(df)

    # 4. Train / test split
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # 5. AutoML with FLAML
    automl, y_pred_automl, automl_results = train_automl_model(
        X_train, X_test, y_train, y_test, time_budget_seconds=60
    )

    # 6. Save test predictions
    save_test_predictions(
        df_model,
        X_test,
        y_test,
        y_pred_automl,
    )

    print("\n=== SUMMARY ===")
    print("FLAML AutoML:", automl_results)


def run_automl_experiment(time_budget_seconds=60, estimators=None):
    """
    Run a full AutoML experiment with flexible time budget and estimators. 
    This allows users on a website to pick and test their favorite regression model

    Parameters
    ----------
    time_budget_seconds : int
        Total time FLAML can spend searching.
    estimators : list[str] or None
        List of estimator names (e.g. ["lgbm", "rf", "xgboost"]).
        If None, uses a default list.
    """
    from flaml import AutoML

    # 1. Load and prepare data
    df = load_data()
    X, y, df_model = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # 2. Prepare AutoML settings
    if estimators is None or len(estimators) == 0:
        estimators = ["lgbm", "rf", "xgboost"]

    automl = AutoML()
    automl_settings = {
        "time_budget": time_budget_seconds,
        "metric": "r2",
        "task": "regression",
        "log_file_name": "automl_life_expectancy.log",
        "estimator_list": estimators,
        "seed": 42,
    }

    automl.fit(X_train=X_train, y_train=y_train, **automl_settings)

    # 3. Evaluate on test set
    y_pred_automl = automl.predict(X_test)
    metrics = evaluate_model("FLAML AutoML", y_test, y_pred_automl)

    return {
        "automl": automl,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred_automl,
        "metrics": metrics,
        "df_model": df_model,
    }

if __name__ == "__main__":
    main()
