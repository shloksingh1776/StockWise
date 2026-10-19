"""Demand Forecasting Engine for StockWise (Milestone 2).

Provides reusable feature engineering, chronological data partitioning,
scikit-learn pipeline construction, model evaluation, and reporting utilities
for multi-store daily sales forecasting.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

# Feature configuration constants
LAG_DAYS: List[int] = [1, 7, 14, 28]
ROLLING_WINDOWS: List[int] = [7, 28]
CATEGORICAL_FEATURES: List[str] = ["store", "item", "dayofweek", "month"]
NUMERICAL_FEATURES: List[str] = [
    "year",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
]

# Chronological split boundary defaults
DEFAULT_TRAIN_END: str = "2017-07-01"
DEFAULT_VAL_START: str = "2017-07-01"
DEFAULT_VAL_END: str = "2017-09-30"
DEFAULT_TEST_START: str = "2017-10-01"
DEFAULT_TEST_END: str = "2017-12-31"


class ForecastingError(ValueError):
    """Exception raised when feature engineering or splitting validation fails."""

    pass


class SeasonalNaiveForecaster(BaseEstimator, RegressorMixin):
    """Weekly Seasonal-Naive forecasting baseline.

    Predicts sales using the same store-product combination's observed sales
    from exactly 7 days prior (lag 7).
    """

    def __init__(self, lag_feature_name: str = "lag_7") -> None:
        self.lag_feature_name = lag_feature_name
        self.lag_feature_idx_: Optional[int] = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Any] = None) -> SeasonalNaiveForecaster:
        """Fit baseline forecaster.

        Identifies feature index if input is DataFrame or records input expectations.
        """
        if isinstance(X, pd.DataFrame):
            if self.lag_feature_name not in X.columns:
                raise ForecastingError(
                    f"Required lag feature '{self.lag_feature_name}' missing from input columns: {list(X.columns)}"
                )
            self.lag_feature_idx_ = X.columns.get_loc(self.lag_feature_name)
        else:
            # If array, lag_feature_name should be passed or assumed first column if 1D
            self.lag_feature_idx_ = None
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict target using 7-day lagged sales."""
        if isinstance(X, pd.DataFrame):
            if self.lag_feature_name not in X.columns:
                raise ForecastingError(
                    f"Required lag feature '{self.lag_feature_name}' missing from input columns: {list(X.columns)}"
                )
            return X[self.lag_feature_name].to_numpy(dtype=float)
        elif isinstance(X, np.ndarray):
            if self.lag_feature_idx_ is not None:
                return X[:, self.lag_feature_idx_].astype(float)
            elif X.ndim == 1:
                return X.astype(float)
            else:
                # If 2D array and index not set, default to lag_7 position if standard feature list is used
                raise ForecastingError("Cannot determine lag_7 column index from raw NumPy array without column metadata.")
        else:
            raise TypeError(f"Unsupported input type for SeasonalNaiveForecaster: {type(X)}")


def create_forecasting_features(
    df: pd.DataFrame,
    date_column: str = "date",
    store_column: str = "store",
    item_column: str = "item",
    sales_column: str = "sales",
    lag_days: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None,
    drop_na: bool = True,
) -> pd.DataFrame:
    """Engineer chronological features for daily demand forecasting.

    Builds calendar features, entity indicators, historical sales lags, and shifted
    past rolling averages grouped by store and item. Strictly guards against
    target data leakage by computing rolling windows with a shift of 1 day.

    Parameters
    ----------
    df : pd.DataFrame
        Input sales DataFrame.
    date_column : str, default 'date'
        Date column name.
    store_column : str, default 'store'
        Store identifier column name.
    item_column : str, default 'item'
        Item identifier column name.
    sales_column : str, default 'sales'
        Sales target column name.
    lag_days : list of int, optional
        Lags in days to compute (defaults to [1, 7, 14, 28]).
    rolling_windows : list of int, optional
        Window lengths in days for shifted rolling averages (defaults to [7, 28]).
    drop_na : bool, default True
        Whether to drop initial rows lacking sufficient history for the longest lag/rolling window.

    Returns
    -------
    pd.DataFrame
        DataFrame with newly generated features sorted chronologically.

    Raises
    ------
    ForecastingError
        If required columns are missing from the input DataFrame.
    """
    required = [date_column, store_column, item_column, sales_column]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ForecastingError(f"Missing required columns for feature engineering: {missing}")

    if lag_days is None:
        lag_days = LAG_DAYS
    if rolling_windows is None:
        rolling_windows = ROLLING_WINDOWS

    # Copy to avoid mutating user data
    data = df.copy()
    data[date_column] = pd.to_datetime(data[date_column])

    # Sort strictly by store, item, and date to ensure correct time-series alignment
    data = data.sort_values(by=[store_column, item_column, date_column]).reset_index(drop=True)

    # 1. Calendar Features
    data["dayofweek"] = data[date_column].dt.dayofweek
    data["month"] = data[date_column].dt.month
    data["year"] = data[date_column].dt.year

    # 2. Historical Lags grouped by (store, item)
    grouped_sales = data.groupby([store_column, item_column])[sales_column]
    for lag in lag_days:
        data[f"lag_{lag}"] = grouped_sales.shift(lag)

    # 3. Shifted Past Rolling Averages
    # Shifting by 1 guarantees that day t sales NEVER enters the feature calculation for day t
    for window in rolling_windows:
        data[f"rolling_mean_{window}"] = grouped_sales.transform(
            lambda s: s.shift(1).rolling(window=window).mean()
        )

    # 4. Handle initial rows without sufficient history
    feature_cols = [f"lag_{l}" for l in lag_days] + [f"rolling_mean_{w}" for w in rolling_windows]
    if drop_na:
        # Explicitly drop rows where any engineered lag or rolling feature is NaN
        data = data.dropna(subset=feature_cols).reset_index(drop=True)

    return data


def split_chronological_data(
    df: pd.DataFrame,
    date_column: str = "date",
    train_end: str = DEFAULT_TRAIN_END,
    val_start: str = DEFAULT_VAL_START,
    val_end: str = DEFAULT_VAL_END,
    test_start: str = DEFAULT_TEST_START,
    test_end: str = DEFAULT_TEST_END,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition dataset into training, validation, and test sets chronologically.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame.
    date_column : str, default 'date'
        Date column name.
    train_end : str, default '2017-07-01'
        Exclusive cutoff date for training targets (dates < train_end).
    val_start : str, default '2017-07-01'
        Inclusive start date for validation targets.
    val_end : str, default '2017-09-30'
        Inclusive end date for validation targets.
    test_start : str, default '2017-10-01'
        Inclusive start date for final test targets.
    test_end : str, default '2017-12-31'
        Inclusive end date for final test targets.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame, pd.DataFrame)
        (train_df, val_df, test_df)

    Raises
    ------
    ForecastingError
        If split bounds are invalid or any split contains zero observations.
    """
    if date_column not in df.columns:
        raise ForecastingError(f"Date column '{date_column}' not found in DataFrame.")

    dates = pd.to_datetime(df[date_column])
    t_end = pd.to_datetime(train_end)
    v_start = pd.to_datetime(val_start)
    v_end = pd.to_datetime(val_end)
    te_start = pd.to_datetime(test_start)
    te_end = pd.to_datetime(test_end)

    if not (t_end <= v_start <= v_end < te_start <= te_end):
        raise ForecastingError(
            f"Invalid chronological split order: train_end={train_end}, "
            f"val=[{val_start}, {val_end}], test=[{test_start}, {test_end}]"
        )

    train_mask = dates < t_end
    val_mask = (dates >= v_start) & (dates <= v_end)
    test_mask = (dates >= te_start) & (dates <= te_end)

    train_df = df.loc[train_mask].copy().reset_index(drop=True)
    val_df = df.loc[val_mask].copy().reset_index(drop=True)
    test_df = df.loc[test_mask].copy().reset_index(drop=True)

    if train_df.empty:
        raise ForecastingError(f"Training split is empty for date < {train_end}.")
    if val_df.empty:
        raise ForecastingError(f"Validation split is empty for date range [{val_start}, {val_end}].")
    if test_df.empty:
        raise ForecastingError(f"Test split is empty for date range [{test_start}, {test_end}].")

    return train_df, val_df, test_df


def build_forecasting_pipeline(
    regressor: Any,
    cat_cols: Optional[List[str]] = None,
    num_cols: Optional[List[str]] = None,
) -> Pipeline:
    """Build a scikit-learn Pipeline with ColumnTransformer for regression.

    Categorical features (store, item, dayofweek, month) are one-hot encoded
    with drop='first' to avoid multicollinearity. Numerical features
    (year, lags, rolling averages) are scaled using StandardScaler.

    Parameters
    ----------
    regressor : scikit-learn compatible estimator
        The regression model to train.
    cat_cols : list of str, optional
        Categorical feature names (defaults to CATEGORICAL_FEATURES).
    num_cols : list of str, optional
        Numerical feature names (defaults to NUMERICAL_FEATURES).

    Returns
    -------
    Pipeline
        Configured scikit-learn Pipeline.
    """
    if cat_cols is None:
        cat_cols = CATEGORICAL_FEATURES
    if num_cols is None:
        num_cols = NUMERICAL_FEATURES

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=True, handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )
    return pipeline


def calculate_metrics(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
    """Calculate evaluation metrics: MAE, RMSE, R-squared, and negative predictions.

    Parameters
    ----------
    y_true : array-like
        Ground truth actual sales values.
    y_pred : array-like
        Predicted sales values.

    Returns
    -------
    dict
        Dictionary containing MAE, RMSE, R2, negative_count, min_pred, max_pred.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    rmse = float(root_mean_squared_error(y_true_arr, y_pred_arr))
    r2 = float(r2_score(y_true_arr, y_pred_arr))
    neg_count = int(np.sum(y_pred_arr < 0))
    min_pred = float(np.min(y_pred_arr))
    max_pred = float(np.max(y_pred_arr))

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "negative_count": neg_count,
        "min_pred": round(min_pred, 4),
        "max_pred": round(max_pred, 4),
    }


def get_model_specifications() -> Dict[str, Dict[str, Any]]:
    """Return dictionary of models and hyperparameters evaluated in Milestone 2."""
    return {
        "Weekly Seasonal Naive": {
            "type": "baseline",
            "model": SeasonalNaiveForecaster(lag_feature_name="lag_7"),
            "description": "Predicts target sales using identical store-product sales from 7 days earlier.",
            "hyperparameters": {"lag": 7},
        },
        "Linear Regression": {
            "type": "ml",
            "model": LinearRegression(),
            "description": "Standard Ordinary Least Squares linear regression with OneHot and StandardScaled features.",
            "hyperparameters": {"fit_intercept": True},
        },
        "Ridge Regression": {
            "type": "ml",
            "model": Ridge(alpha=1.0, random_state=42),
            "description": "L2 regularized linear regression (alpha=1.0) preventing feature weight inflation.",
            "hyperparameters": {"alpha": 1.0, "solver": "auto", "random_state": 42},
        },
        "Lasso Regression": {
            "type": "ml",
            "model": Lasso(alpha=0.1, max_iter=1000, random_state=42),
            "description": "L1 regularized sparse linear regression (alpha=0.1) enforcing feature sparsity.",
            "hyperparameters": {"alpha": 0.1, "max_iter": 1000, "random_state": 42},
        },
        "Elastic Net Regression": {
            "type": "ml",
            "model": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000, random_state=42),
            "description": "Elastic Net balancing L1 and L2 penalties (alpha=0.1, l1_ratio=0.5).",
            "hyperparameters": {"alpha": 0.1, "l1_ratio": 0.5, "max_iter": 1000, "random_state": 42},
        },
    }


def evaluate_models_on_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    cat_cols: Optional[List[str]] = None,
    num_cols: Optional[List[str]] = None,
    sales_column: str = "sales",
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, np.ndarray]]:
    """Train all candidate models on train_df and evaluate on val_df.

    Parameters
    ----------
    train_df : pd.DataFrame
        Feature-engineered training split.
    val_df : pd.DataFrame
        Feature-engineered validation split.
    test_df : pd.DataFrame, optional
        Feature-engineered test split (evaluated on best model).
    cat_cols : list of str, optional
        Categorical feature names.
    num_cols : list of str, optional
        Numerical feature names.
    sales_column : str, default 'sales'
        Sales target column.

    Returns
    -------
    tuple of (pd.DataFrame, dict, dict)
        (comparison_df, fitted_models, val_predictions)
    """
    if cat_cols is None:
        cat_cols = CATEGORICAL_FEATURES
    if num_cols is None:
        num_cols = NUMERICAL_FEATURES

    feature_cols = cat_cols + num_cols
    X_train = train_df[feature_cols]
    y_train = train_df[sales_column]
    X_val = val_df[feature_cols]
    y_val = val_df[sales_column]

    model_specs = get_model_specifications()
    results: List[Dict[str, Any]] = []
    fitted_models: Dict[str, Any] = {}
    val_predictions: Dict[str, np.ndarray] = {}

    for name, spec in model_specs.items():
        t0 = time.time()
        if spec["type"] == "baseline":
            model = spec["model"]
            model.fit(X_train, y_train)
            fit_time = time.time() - t0
            t_pred = time.time()
            preds = model.predict(X_val)
            pred_time = time.time() - t_pred
            fitted_models[name] = model
        else:
            pipeline = build_forecasting_pipeline(spec["model"], cat_cols=cat_cols, num_cols=num_cols)
            pipeline.fit(X_train, y_train)
            fit_time = time.time() - t0
            t_pred = time.time()
            preds = pipeline.predict(X_val)
            pred_time = time.time() - t_pred
            fitted_models[name] = pipeline

        val_predictions[name] = preds
        metrics = calculate_metrics(y_val, preds)

        results.append({
            "Model": name,
            "Type": spec["type"],
            "Validation MAE": metrics["MAE"],
            "Validation RMSE": metrics["RMSE"],
            "Validation R2": metrics["R2"],
            "Negative Preds": metrics["negative_count"],
            "Min Pred": metrics["min_pred"],
            "Max Pred": metrics["max_pred"],
            "Fit Time (s)": round(fit_time, 2),
            "Inference Time (s)": round(pred_time, 2),
        })

    comparison_df = pd.DataFrame(results).sort_values("Validation MAE").reset_index(drop=True)
    return comparison_df, fitted_models, val_predictions


def plot_model_comparison(comparison_df: pd.DataFrame, output_path: Union[str, Path]) -> None:
    """Generate bar chart comparing validation MAE and RMSE across candidate models.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Model comparison summary table.
    output_path : str or Path
        Destination image path.
    """
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # MAE Plot
    sns.barplot(
        data=comparison_df,
        x="Validation MAE",
        y="Model",
        hue="Model",
        palette="viridis",
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Validation MAE (Lower is Better)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Mean Absolute Error (Units)")
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.2f", padding=3, fontsize=10)

    # RMSE Plot
    sns.barplot(
        data=comparison_df,
        x="Validation RMSE",
        y="Model",
        hue="Model",
        palette="magma",
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Validation RMSE (Lower is Better)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Root Mean Squared Error (Units)")
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.2f", padding=3, fontsize=10)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_actual_vs_predicted(
    dates: pd.Series,
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    output_path: Union[str, Path],
    num_days: int = 45,
    random_state: int = 42,
) -> None:
    """Generate actual vs predicted daily aggregate time series and scatter plots.

    Parameters
    ----------
    dates : pd.Series
        Dates for the validation/test period.
    y_true : pd.Series
        Ground truth actual sales.
    y_pred : np.ndarray
        Model predicted sales.
    model_name : str
        Name of the evaluated model.
    output_path : str or Path
        Destination image path.
    num_days : int, default 45
        Number of initial days in the split to plot for daily aggregate clarity.
    random_state : int, default 42
        Random seed for reproducible scatter subsample generation.
    """
    eval_df = pd.DataFrame({"date": dates, "actual": y_true, "predicted": y_pred})
    daily_agg = eval_df.groupby("date").agg({"actual": "mean", "predicted": "mean"}).reset_index()

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # 1. Time series trend (mean daily sales per store-item pair)
    plot_slice = daily_agg.head(num_days)
    axes[0].plot(plot_slice["date"], plot_slice["actual"], label="Actual Sales", color="#1f77b4", lw=2, marker="o")
    axes[0].plot(
        plot_slice["date"],
        plot_slice["predicted"],
        label=f"{model_name} Forecast",
        color="#ff7f0e",
        lw=2,
        linestyle="--",
        marker="s",
    )
    axes[0].set_title(f"Average Daily Sales: Actual vs. {model_name} (First {num_days} Days)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Average Units Sold")
    axes[0].legend(frameon=True)
    axes[0].tick_params(axis="x", rotation=45)

    # 2. Scatter plot actual vs predicted on sample
    rng = np.random.default_rng(random_state)
    sample_indices = rng.choice(len(eval_df), size=min(5000, len(eval_df)), replace=False)
    sample_eval = eval_df.iloc[sample_indices]

    axes[1].scatter(sample_eval["actual"], sample_eval["predicted"], alpha=0.25, color="#2ca02c", s=15)
    max_val = max(sample_eval["actual"].max(), sample_eval["predicted"].max())
    axes[1].plot([0, max_val], [0, max_val], color="red", linestyle="--", lw=1.5, label="Perfect Forecast (y = x)")
    axes[1].set_title(f"Actual vs. Predicted Scatter Sample ({model_name})", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Actual Sales Units")
    axes[1].set_ylabel("Predicted Sales Units")
    axes[1].legend(frameon=True)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_residuals_analysis(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    output_path: Union[str, Path],
    random_state: int = 42,
) -> None:
    """Generate residual distribution and residuals vs fitted plots.

    Parameters
    ----------
    y_true : pd.Series
        Ground truth actual sales.
    y_pred : np.ndarray
        Model predicted sales.
    model_name : str
        Name of the evaluated model.
    output_path : str or Path
        Destination image path.
    random_state : int, default 42
        Random seed for reproducible scatter subsample generation.
    """
    residuals = y_true.to_numpy(dtype=float) - y_pred

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 1. Residual distribution
    sns.histplot(residuals, bins=50, kde=True, ax=axes[0], color="#4C72B0")
    axes[0].axvline(0, color="red", linestyle="--", lw=1.5)
    axes[0].set_title(f"Residual Distribution ({model_name})", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Residual (Actual - Predicted)")
    axes[0].set_ylabel("Frequency")

    # 2. Residuals vs Predicted
    rng = np.random.default_rng(random_state)
    sample_indices = rng.choice(len(residuals), size=min(5000, len(residuals)), replace=False)
    axes[1].scatter(y_pred[sample_indices], residuals[sample_indices], alpha=0.25, color="#C44E52", s=15)
    axes[1].axhline(0, color="black", linestyle="--", lw=1.5)
    axes[1].set_title(f"Residuals vs. Fitted Values ({model_name})", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Fitted (Predicted) Values")
    axes[1].set_ylabel("Residual (Actual - Predicted)")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
