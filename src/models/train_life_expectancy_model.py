"""
Train and compare multiple regression models for life expectancy.

This script:
- Loads the cleaned life expectancy + GDP + CO2 dataset
- Performs basic sanity checks and prepares features/target
- Splits the data into train and test sets
- Trains:
    * A Linear Regression baseline (with standardization)
    * Three FLAML AutoML models, each restricted to a single estimator
      - lgbm
      - rf (Random Forest)
      - xgboost
- Collects RMSE, MAE, and R² for each model
- Saves a comparison table to:
    data/processed/model_comparison_metrics.csv

You can then visualize these metrics in your Quarto website
(e.g., bar charts comparing RMSE / MAE / R² by model).

To run:
    python -m src.models.train_life_expectancy_models
or:
    python src/models/train_life_expectancy_models.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from flaml import AutoML
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------
# Data loading & preparation helpers
# --------------------------------------------------------------------


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


def train_test_split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Split the data into train and test sets.

    Parameters
    ----------
    X, y :
        Features and target.
    test_size : float, default=0.2
        Fraction of data to use for testing.
    random_state : int, default=42
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


def evaluate_model(name: str, y_test, y_pred) -> dict:
    """
    Compute and print standard regression metrics.

    Parameters
    ----------
    name : str
        Name of the model (for printing).
    y_test : array-like
        True target values.
    y_pred : array-like
        Predicted target values.

    Returns
    -------
    metrics : dict
        Dictionary with keys: rmse, mae, r2.
    """
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n=== {name} RESULTS ===")
    print(f"RMSE: {rmse:0.3f}")
    print(f"MAE : {mae:0.3f}")
    print(f"R^2 : {r2:0.3f}")

    return {"rmse": rmse, "mae": mae, "r2": r2}


# --------------------------------------------------------------------
# Model training helpers
# --------------------------------------------------------------------


def train_linear_baseline(X_train, X_test, y_train, y_test):
    """
    Train a simple Linear Regression model with standardization.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame or np.ndarray
        Train and test feature matrices.
    y_train, y_test : pd.Series or np.ndarray
        Train and test targets.

    Returns
    -------
    model : sklearn.Pipeline
        Fitted pipeline (StandardScaler + LinearRegression).
    y_pred : np.ndarray
        Predictions on the test set.
    metrics : dict
        Dictionary with RMSE, MAE, and R² on the test set.
    """
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    metrics = evaluate_model("Linear Regression", y_test, y_pred)

    return pipe, y_pred, metrics


def train_automl_model(
    X_train,
    X_test,
    y_train,
    y_test,
    time_budget_seconds: int = 60,
    estimator_list=None,
):
    """
    Use FLAML AutoML to search for a good regression model.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame or np.ndarray
        Train and test feature matrices.
    y_train, y_test : pd.Series or np.ndarray
        Train and test targets.
    time_budget_seconds : int, default=60
        Total time budget for the AutoML search in seconds.
    estimator_list : list[str] or None, default=None
        List of estimator names (e.g. ["lgbm", "rf", "xgboost"]).
        If None or empty, uses a default list of tree-based models.

    Returns
    -------
    automl : flaml.automl.AutoML
        Fitted FLAML AutoML object.
    y_pred_automl : np.ndarray
        Predictions on the test set.
    automl_results : dict
        Metrics (RMSE, MAE, R²) on the test set.
    """
    automl = AutoML()

    if estimator_list is None or len(estimator_list) == 0:
        estimator_list = ["lgbm", "rf", "xgboost"]

    automl_settings = {
        "time_budget": time_budget_seconds,
        "metric": "r2",  # optimize for R²
        "task": "regression",
        "log_file_name": "automl_life_expectancy.log",
        "estimator_list": estimator_list,
        "seed": 42,
    }

    print("\n=== RUNNING FLAML AUTOML ===")
    print("Estimators:", estimator_list)

    automl.fit(
        X_train=X_train,
        y_train=y_train,
        **automl_settings,
    )

    print("\nBest estimator type:", automl.best_estimator)
    print("Best config:", automl.best_config)
    print("Best validation loss (1 - R²):", automl.best_loss)
    approx_val_r2 = 1 - automl.best_loss
    print(f"Approx validation R²: {approx_val_r2:0.3f}")

    # Evaluate on test set
    y_pred_automl = automl.predict(X_test)
    automl_results = evaluate_model(
        f"FLAML ({'+'.join(estimator_list)})", y_test, y_pred_automl
    )

    return automl, y_pred_automl, automl_results


def train_all_models(X_train, X_test, y_train, y_test, total_time_budget: int = 240):
    """
    Train multiple models and collect metrics.

    Trains:
      - Linear Regression baseline
      - FLAML AutoML constrained to each of: ["lgbm", "rf", "xgboost"]

    The total_time_budget is split across the three AutoML runs.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame or np.ndarray
    y_train, y_test : pd.Series or np.ndarray
    total_time_budget : int, default=240
        Total time budget in seconds to split across the FLAML runs.

    Returns
    -------
    results : dict
        Nested dictionary of the form:
        {
            "linear_regression": {
                "model": fitted_model,
                "y_pred": np.ndarray,
                "metrics": {"rmse": ..., "mae": ..., "r2": ...},
            },
            "lgbm": {...},
            "rf": {...},
            "xgboost": {...},
        }
    """
    results = {}

    # 1. Linear Regression baseline
    lin_model, lin_pred, lin_metrics = train_linear_baseline(
        X_train, X_test, y_train, y_test
    )
    results["linear_regression"] = {
        "model": lin_model,
        "y_pred": lin_pred,
        "metrics": lin_metrics,
    }

    # 2. AutoML for each tree-based estimator separately
    estimators = ["lgbm", "rf", "xgboost"]
    per_model_budget = max(total_time_budget // len(estimators), 30)

    for est in estimators:
        print(f"\n=== AutoML for {est} only ===")
        automl_model, y_pred, metrics = train_automl_model(
            X_train,
            X_test,
            y_train,
            y_test,
            time_budget_seconds=per_model_budget,
            estimator_list=[est],  # key: restrict AutoML to a single estimator
        )
        results[est] = {
            "model": automl_model,
            "y_pred": y_pred,
            "metrics": metrics,
        }

    return results

def build_and_save_test_predictions(
    df_model: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    all_results: dict,
) -> pd.DataFrame:
    """
    Build a test-set predictions table for all models and save it as a CSV.

    The output includes:
      - ID / feature columns from the original data (for the test rows)
      - the actual life_expectancy (y_actual)
      - one prediction column per model, e.g. y_pred_linear_regression, y_pred_lgbm, ...

    Returns
    -------
    test_df : pd.DataFrame
        DataFrame with actuals and predictions for each model on the test set.
    """
    # Columns from the original cleaned dataset that you want to keep.
    # This is flexible: only keep the ones that actually exist in df_model.
    candidate_cols = [
        "iso3",
        "country_name",
        "year",
        "gdp_per_capita",
        "co2_per_capita",
        "life_expectancy",
    ]
    cols_to_keep = [c for c in candidate_cols if c in df_model.columns]

    # Use X_test.index to grab the corresponding original rows from df_model
    test_df = df_model.loc[X_test.index, cols_to_keep].copy()

    # Rename life_expectancy -> y_actual (if present)
    if "life_expectancy" in test_df.columns:
        test_df = test_df.rename(columns={"life_expectancy": "y_actual"})
    else:
        # If it's missing for some reason, fall back to y_test
        test_df["y_actual"] = y_test.values

    # Add one prediction column per model
    for model_name, res in all_results.items():
        test_df[f"y_pred_{model_name}"] = res["y_pred"]

    # Save to CSV
    project_root = Path(__file__).resolve().parents[2]
    preds_path = (
        project_root
        / "data"
        / "processed"
        / "test_predictions_life_expectancy_models.csv"
    )
    test_df.to_csv(preds_path, index=False)
    print(f"\nSaved test predictions to {preds_path}")

    return test_df

# --------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------

def main():
    """
    End-to-end entry point to train all models and save metrics for the website.
    """
    # 1. Load data
    df = load_data()

    # 2. Basic checks
    basic_checks(df)

    # 3. Prepare features and target
    X, y, df_model = prepare_features_and_target(df)

    # 4. Train / test split
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # 5. Train all models (baseline + 3 AutoML variants)
    all_results = train_all_models(
        X_train, X_test, y_train, y_test, total_time_budget=240
    )

    # 5b. Build + save test predictions CSV (NEW)
    test_predictions_df = build_and_save_test_predictions(
        df_model=df_model,
        X_test=X_test,
        y_test=y_test,
        all_results=all_results,
    )

    # 6. Build metrics DataFrame for plotting later (existing)
    rows = []
    for name, res in all_results.items():
        m = res["metrics"]
        rows.append(
            {
                "model": name,
                "rmse": m["rmse"],
                "mae": m["mae"],
                "r2": m["r2"],
            }
        )
    metrics_df = pd.DataFrame(rows)
    print("\n=== COMPARISON TABLE ===")
    print(metrics_df)

    # 7. Save metrics for the website (existing)
    project_root = Path(__file__).resolve().parents[2]
    metrics_path = project_root / "data" / "processed" / "model_comparison_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved model comparison metrics to {metrics_path}")

if __name__ == "__main__":
    main()
