"""Dashboard utilities and helper service layer for StockWise Milestone 4.

Provides headless, unit-testable functions for data loading, report ingestion,
time-series filtering, KPI summaries, and scenario planning adapters that
interface cleanly with `stockwise.data_loader`, `stockwise.forecasting`,
and `stockwise.inventory`.

All business logic and inventory calculations remain strictly in their core
modules; this service layer handles data shaping, validation, and presentation
preparation without altering underlying analytical formulas.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import Ridge

from stockwise.data_loader import load_raw_data
from stockwise.forecasting import (
    CATEGORICAL_FEATURES,
    DEFAULT_TEST_END,
    DEFAULT_TEST_START,
    DEFAULT_TRAIN_END,
    DEFAULT_VAL_END,
    DEFAULT_VAL_START,
    NUMERICAL_FEATURES,
    build_forecasting_pipeline,
    create_forecasting_features,
    split_chronological_data,
)
from stockwise.inventory import (
    InventoryPolicyError,
    InventoryPosition,
    ReplenishmentPolicy,
    ReplenishmentRecommendation,
    calculate_sku_residual_variability,
    evaluate_replenishment_scenario,
    summarize_demand_and_risk,
)

# Benchmark Holdout Test Metrics (Untouched holdout test split: 2017-10-01 to 2017-12-31)
# Preserved strictly as a fixed benchmark; never used for tuning, selection, or replenishment policies.
HOLDOUT_TEST_BENCHMARK: Dict[str, Any] = {
    "test_period": "2017-10-01 to 2017-12-31 (Q4 2017)",
    "observations": 46000,
    "ridge_regression": {
        "model_name": "Ridge Regression (Selected)",
        "mae": 6.7529,
        "rmse": 8.7568,
        "r2": 0.9055,
        "negative_predictions": 35,
        "negative_pct": 0.076,
        "min_prediction": -4.30,
    },
    "seasonal_naive_baseline": {
        "model_name": "Weekly Seasonal Naive",
        "mae": 9.0763,
        "rmse": 12.0507,
        "r2": 0.8210,
        "negative_predictions": 0,
        "negative_pct": 0.0,
        "min_prediction": 1.0,
    },
    "mae_reduction_pct": 25.6,
    "rmse_reduction_pct": 27.3,
    "methodology_note": (
        "This evaluation reflects fixed-model holdout protocol. The model trained through "
        "2017-06-30 was evaluated on the final test split without retraining or tuning."
    ),
}

# Canonical Risk Quadrant Definitions and Label Mappings
CANONICAL_RISK_QUADRANTS: Dict[str, str] = {
    "High Velocity - Volatile": "High Velocity - Volatile (Requires Inventory Review)",
    "High Velocity - Stable": "High Velocity - Stable (Steady Fast-Mover)",
    "Low Velocity - Stable": "Low Velocity - Stable (Low Maintenance)",
    "Lumpy / Erratic Demand": "Lumpy / Erratic Demand (Intermittent)",
}

CANONICAL_RISK_DISPLAY_LABELS: Dict[str, str] = {v: k for k, v in CANONICAL_RISK_QUADRANTS.items()}

# Module-level cache containers for reusable data and model artifacts
_DATA_CACHE: Dict[str, Any] = {}


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parents[2]


def get_default_data_path() -> Path:
    """Return default path to train.csv."""
    return get_project_root() / "data" / "raw" / "train.csv"


def get_default_reports_dir() -> Path:
    """Return default path to reports directory."""
    return get_project_root() / "reports"


def load_dashboard_sales_data(
    file_path: Optional[Union[str, Path]] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load and validate raw sales dataset using the established data loader.

    Parameters
    ----------
    file_path : str or Path, optional
        Path to sales CSV. Defaults to data/raw/train.csv relative to project root.
    use_cache : bool, default True
        Whether to cache the loaded DataFrame in memory to avoid redundant I/O.

    Returns
    -------
    pd.DataFrame
        Validated sales DataFrame with datetime parsed 'date' column.
    """
    target_path = Path(file_path) if file_path is not None else get_default_data_path()
    cache_key = f"sales_data_{target_path.resolve()}"

    if use_cache and cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key].copy()

    df = load_raw_data(target_path)
    if use_cache:
        _DATA_CACHE[cache_key] = df
    return df.copy()


def load_model_comparison(
    file_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load pre-computed validation model comparison table from reports.

    Parameters
    ----------
    file_path : str or Path, optional
        Path to model_comparison.csv. Defaults to reports/model_comparison.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with model comparison metrics sorted by Validation MAE.

    Raises
    ------
    FileNotFoundError
        If model_comparison.csv does not exist.
    """
    target_path = (
        Path(file_path)
        if file_path is not None
        else get_default_reports_dir() / "model_comparison.csv"
    )
    if not target_path.is_file():
        raise FileNotFoundError(f"Model comparison report not found at: {target_path.resolve()}")

    df = pd.read_csv(target_path)
    required = ["Model", "Type", "Validation MAE", "Validation RMSE", "Validation R2"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"model_comparison.csv is missing required columns: {missing}")

    return df.sort_values("Validation MAE").reset_index(drop=True)


def load_saved_replenishment_scenarios(
    file_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load pre-evaluated representative replenishment scenarios from reports.

    Parameters
    ----------
    file_path : str or Path, optional
        Path to inventory_replenishment_scenarios.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame containing 72 scenario evaluations across representative SKUs.
    """
    target_path = (
        Path(file_path)
        if file_path is not None
        else get_default_reports_dir() / "inventory_replenishment_scenarios.csv"
    )
    if not target_path.is_file():
        raise FileNotFoundError(
            f"Inventory scenarios report not found at: {target_path.resolve()}"
        )

    return pd.read_csv(target_path)


def get_holdout_test_metrics() -> Dict[str, Any]:
    """Return untouched Q4 2017 holdout test benchmark metrics."""
    return HOLDOUT_TEST_BENCHMARK.copy()


def get_store_and_item_options(df: pd.DataFrame) -> Tuple[List[int], List[int]]:
    """Return sorted lists of unique stores and items present in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Sales dataset containing 'store' and 'item' columns.

    Returns
    -------
    tuple of (list of int, list of int)
        Sorted store IDs and item IDs.
    """
    if "store" not in df.columns or "item" not in df.columns:
        raise ValueError("DataFrame must contain 'store' and 'item' columns.")

    stores = sorted(int(s) for s in df["store"].dropna().unique())
    items = sorted(int(i) for i in df["item"].dropna().unique())
    return stores, items


def get_date_range(df: pd.DataFrame, date_column: str = "date") -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Return min and max timestamps present in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with date column.
    date_column : str, default 'date'
        Name of date column.

    Returns
    -------
    tuple of (pd.Timestamp, pd.Timestamp)
        Earliest and latest date in the dataset.
    """
    if df.empty:
        raise ValueError("Cannot extract date range from an empty DataFrame.")
    if date_column not in df.columns:
        raise ValueError(f"Column '{date_column}' not found in DataFrame.")

    dates = pd.to_datetime(df[date_column])
    return dates.min(), dates.max()


def filter_sales_data(
    df: pd.DataFrame,
    store: Optional[int] = None,
    item: Optional[int] = None,
    start_date: Optional[Union[str, pd.Timestamp, datetime.date]] = None,
    end_date: Optional[Union[str, pd.Timestamp, datetime.date]] = None,
    date_column: str = "date",
    store_column: str = "store",
    item_column: str = "item",
) -> pd.DataFrame:
    """Filter sales dataset by store, item, and date range.

    Parameters
    ----------
    df : pd.DataFrame
        Sales DataFrame.
    store : int, optional
        Target store ID to filter by.
    item : int, optional
        Target item ID to filter by.
    start_date : str, pd.Timestamp, or datetime.date, optional
        Inclusive start date.
    end_date : str, pd.Timestamp, or datetime.date, optional
        Inclusive end date.
    date_column : str, default 'date'
        Date column name.
    store_column : str, default 'store'
        Store column name.
    item_column : str, default 'item'
        Item column name.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame sorted chronologically by date.
    """
    if df.empty:
        return df.copy()

    filtered = df.copy()

    if store is not None:
        filtered = filtered[filtered[store_column] == store]

    if item is not None:
        filtered = filtered[filtered[item_column] == item]

    if start_date is not None:
        start_ts = pd.to_datetime(start_date)
        filtered = filtered[pd.to_datetime(filtered[date_column]) >= start_ts]

    if end_date is not None:
        end_ts = pd.to_datetime(end_date)
        filtered = filtered[pd.to_datetime(filtered[date_column]) <= end_ts]

    if not filtered.empty and date_column in filtered.columns:
        filtered = filtered.sort_values(by=date_column).reset_index(drop=True)

    return filtered


def compute_overview_kpis(
    df: pd.DataFrame,
    summary_df: Optional[pd.DataFrame] = None,
    sales_column: str = "sales",
    store_column: str = "store",
    item_column: str = "item",
    date_column: str = "date",
) -> Dict[str, Any]:
    """Compute executive overview KPIs and dataset boundary metadata.

    Parameters
    ----------
    df : pd.DataFrame
        Full or filtered sales DataFrame.
    summary_df : pd.DataFrame, optional
        Demand risk summary table from summarize_demand_and_risk.
    sales_column : str, default 'sales'
        Sales quantity column.
    store_column : str, default 'store'
        Store ID column.
    item_column : str, default 'item'
        Item ID column.
    date_column : str, default 'date'
        Date column.

    Returns
    -------
    dict
        Structured KPI metrics and explicit dataset boundary notes.
    """
    if df.empty:
        return {
            "total_sales": 0,
            "avg_daily_sales": 0.0,
            "total_stores": 0,
            "total_items": 0,
            "total_series": 0,
            "total_days": 0,
            "start_date": None,
            "end_date": None,
            "critical_risk_count": 0,
            "elevated_risk_count": 0,
            "standard_risk_count": 0,
            "low_risk_count": 0,
            "limitations": get_dataset_disclaimers(),
        }

    dates = pd.to_datetime(df[date_column])
    start_dt = dates.min()
    end_dt = dates.max()
    unique_dates_count = int(dates.nunique())

    total_sales = int(df[sales_column].sum())
    avg_daily_sales = float(round(df[sales_column].mean(), 2))
    stores_count = int(df[store_column].nunique())
    items_count = int(df[item_column].nunique())
    total_series = int(df[[store_column, item_column]].drop_duplicates().shape[0])

    kpis: Dict[str, Any] = {
        "total_sales": total_sales,
        "avg_daily_sales": avg_daily_sales,
        "total_stores": stores_count,
        "total_items": items_count,
        "total_series": total_series,
        "total_days": unique_dates_count,
        "start_date": start_dt.strftime("%Y-%m-%d"),
        "end_date": end_dt.strftime("%Y-%m-%d"),
        "limitations": get_dataset_disclaimers(),
    }

    if summary_df is not None and not summary_df.empty and "review_priority" in summary_df.columns:
        counts = summary_df["review_priority"].value_counts().to_dict()
        kpis["critical_risk_count"] = int(counts.get("CRITICAL", 0))
        kpis["elevated_risk_count"] = int(counts.get("ELEVATED", 0))
        kpis["standard_risk_count"] = int(counts.get("STANDARD", 0))
        kpis["low_risk_count"] = int(counts.get("LOW", 0))
    else:
        kpis["critical_risk_count"] = 0
        kpis["elevated_risk_count"] = 0
        kpis["standard_risk_count"] = 0
        kpis["low_risk_count"] = 0

    return kpis


def get_dataset_disclaimers() -> Dict[str, List[str]]:
    """Return explicit statements of what the benchmark dataset establishes vs cannot establish."""
    return {
        "what_dataset_establishes": [
            "Historical daily point-of-sale transactions across 10 stores and 50 items (500 series).",
            "5-year longitudinal sales observations from 2013-01-01 to 2017-12-31 (913,000 daily rows).",
            "Empirical multi-scale seasonality: weekly day-of-week patterns and annual seasonal cycles.",
            "Benchmarked out-of-sample forecasting performance using strict chronological splits.",
        ],
        "what_dataset_cannot_establish": [
            "Actual on-hand or physical inventory balances in stores or distribution centers.",
            "Historical supplier fulfillment records, delivery delay distributions, or lead times.",
            "Purchase order logs, min/max batch sizes, pack sizes, or pallet constraints.",
            "Inventory carrying/holding costs, shortage penalty costs, or freight purchase rates.",
            "Historical stockout events, inventory shortages, or customer lost sales.",
        ],
    }


def prepare_demand_risk_data(
    sales_df: pd.DataFrame,
    recent_history_days: int = 90,
    forecast_df: Optional[pd.DataFrame] = None,
    cutoff_date: Optional[Union[str, pd.Timestamp]] = "2017-07-01",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Generate or retrieve cached demand-risk quadrant segmentation across all series.

    Calls `stockwise.inventory.summarize_demand_and_risk` without altering formulas.
    If cutoff_date is specified and the dataset contains observations on or past that date,
    data is filtered to `date < cutoff_date` to preserve validation and test holdout
    protection (matching the authoritative Milestone 3 baseline).

    Parameters
    ----------
    sales_df : pd.DataFrame
        Sales DataFrame.
    recent_history_days : int, default 90
        Recent historical window length in days used to evaluate empirical velocity/volatility.
    forecast_df : pd.DataFrame, optional
        Optional predicted sales DataFrame for attached forecasts.
    cutoff_date : str or pd.Timestamp, optional, default '2017-07-01'
        Cutoff boundary date to prevent evaluation on validation/test splits.
    use_cache : bool, default True
        Whether to cache the resulting summary DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with demand velocity, volatility, percentiles, risk score, and quadrants.
    """
    effective_df = sales_df
    if cutoff_date is not None and not sales_df.empty and "date" in sales_df.columns:
        ts_cutoff = pd.to_datetime(cutoff_date)
        max_dt = pd.to_datetime(sales_df["date"]).max()
        if max_dt >= ts_cutoff:
            effective_df = sales_df[pd.to_datetime(sales_df["date"]) < ts_cutoff]

    cache_key = f"demand_risk_summary_{len(effective_df)}_{recent_history_days}"
    if use_cache and cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key].copy()

    summary = summarize_demand_and_risk(
        sales_df=effective_df,
        forecast_df=forecast_df,
        recent_history_days=recent_history_days,
    )

    if use_cache:
        _DATA_CACHE[cache_key] = summary

    return summary.copy()


def filter_demand_risk_data(
    summary_df: pd.DataFrame,
    store: Optional[int] = None,
    item: Optional[int] = None,
    risk_category: Optional[str] = None,
    review_priority: Optional[str] = None,
) -> pd.DataFrame:
    """Filter the demand risk summary DataFrame by store, item, quadrant, or priority.

    Supports canonical short names ('High Velocity - Volatile') as well as full
    database labels ('High Velocity - Volatile (Requires Inventory Review)').
    Uses non-regex literal substring matching to avoid regex metacharacter errors.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Demand risk summary DataFrame.
    store : int, optional
        Store ID.
    item : int, optional
        Item ID.
    risk_category : str, optional
        Full or canonical risk category name (e.g., 'High Velocity - Volatile').
    review_priority : str, optional
        Review priority ('CRITICAL', 'ELEVATED', 'STANDARD', 'LOW').

    Returns
    -------
    pd.DataFrame
        Filtered demand risk DataFrame.
    """
    if summary_df.empty:
        return summary_df.copy()

    res = summary_df.copy()

    if store is not None:
        res = res[res["store"] == int(store)]

    if item is not None:
        res = res[res["item"] == int(item)]

    if risk_category is not None and risk_category != "All":
        canonical_full = CANONICAL_RISK_QUADRANTS.get(risk_category, risk_category)
        res = res[
            (res["risk_category"] == canonical_full)
            | (res["risk_category"] == risk_category)
            | (res["risk_category"].str.contains(risk_category, case=False, na=False, regex=False))
        ]

    if review_priority is not None and review_priority != "All":
        res = res[res["review_priority"].astype(str).str.upper() == str(review_priority).upper()]

    return res.reset_index(drop=True)


def train_and_cache_validation_predictions(
    sales_df: pd.DataFrame,
    use_cache: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train Ridge regression once on training split and compute validation predictions.

    Preserves strict chronological boundaries:
    - Train split: date < 2017-07-01
    - Validation split: 2017-07-01 <= date <= 2017-09-30 (Q3 2017)
    - Test split: strictly untouched

    Parameters
    ----------
    sales_df : pd.DataFrame
        Raw or full sales DataFrame.
    use_cache : bool, default True
        Whether to cache the resulting DataFrames across dashboard runs.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        - val_predictions_df: validation split rows with ['date', 'store', 'item', 'sales', 'predicted_sales']
        - sku_residuals_df: SKU-level residual dispersion metrics (val_residual_rmse, val_residual_std)
    """
    cache_key = "val_predictions_and_residuals"
    if use_cache and cache_key in _DATA_CACHE:
        val_df_cached, res_df_cached = _DATA_CACHE[cache_key]
        return val_df_cached.copy(), res_df_cached.copy()

    # 1. Feature Engineering
    feat_df = create_forecasting_features(sales_df, drop_na=True)

    # 2. Chronological Splits
    train_df, val_df, _ = split_chronological_data(
        feat_df,
        train_end=DEFAULT_TRAIN_END,
        val_start=DEFAULT_VAL_START,
        val_end=DEFAULT_VAL_END,
        test_start=DEFAULT_TEST_START,
        test_end=DEFAULT_TEST_END,
    )

    # 3. Fit Ridge Pipeline on Train Only
    feature_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    pipeline = build_forecasting_pipeline(Ridge(alpha=1.0, random_state=42))
    pipeline.fit(train_df[feature_cols], train_df["sales"])

    # 4. Predict on Validation Split Only
    preds = pipeline.predict(val_df[feature_cols])

    val_predictions_df = val_df[["date", "store", "item", "sales"]].copy()
    val_predictions_df["predicted_sales"] = preds

    # 5. Calculate SKU-level validation residual variability
    sku_residuals_df = calculate_sku_residual_variability(
        val_df=val_df,
        val_predictions=preds,
    )

    if use_cache:
        _DATA_CACHE[cache_key] = (val_predictions_df, sku_residuals_df)

    return val_predictions_df.copy(), sku_residuals_df.copy()


def get_sku_demand_and_variability(
    summary_df: pd.DataFrame,
    sku_residuals_df: Optional[pd.DataFrame],
    store: int,
    item: int,
) -> Dict[str, Any]:
    """Look up empirical daily demand, historical std, and validation RMSE for a SKU.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Demand risk summary DataFrame.
    sku_residuals_df : pd.DataFrame, optional
        SKU residual dispersion metrics DataFrame.
    store : int
        Target store ID.
    item : int
        Target item ID.

    Returns
    -------
    dict
        Dictionary containing mean daily sales, historical sales std, forecast RMSE,
        risk quadrant, and sample sizes.
    """
    row_sum = summary_df[(summary_df["store"] == store) & (summary_df["item"] == item)]
    if row_sum.empty:
        raise ValueError(f"SKU (store={store}, item={item}) not found in summary DataFrame.")

    sku_sum = row_sum.iloc[0]
    daily_rate = float(sku_sum["mean_daily_sales"])
    hist_std = float(sku_sum["std_daily_sales"])
    quadrant = str(sku_sum["risk_category"])
    priority = str(sku_sum["review_priority"])

    val_rmse = hist_std  # fallback if residuals df not provided
    sample_size = 92

    if sku_residuals_df is not None and not sku_residuals_df.empty:
        row_res = sku_residuals_df[
            (sku_residuals_df["store"] == store) & (sku_residuals_df["item"] == item)
        ]
        if not row_res.empty:
            val_rmse = float(row_res.iloc[0]["val_residual_rmse"])
            sample_size = int(row_res.iloc[0]["val_observation_count"])

    return {
        "store": store,
        "item": item,
        "daily_demand_rate": daily_rate,
        "historical_demand_std": hist_std,
        "forecast_residual_rmse": val_rmse,
        "risk_category": quadrant,
        "review_priority": priority,
        "validation_sample_size": sample_size,
    }


def evaluate_scenario_adapter(
    store: int,
    item: int,
    daily_demand_rate: float,
    demand_variability: float,
    lead_time_days: int,
    service_level: float = 0.95,
    lead_time_std_days: float = 0.0,
    min_order_quantity: int = 1,
    review_period_days: int = 1,
    on_hand: int = 0,
    on_order: int = 0,
    backorders: int = 0,
    variability_basis: str = "historical_demand_variance",
    residual_sample_size: Optional[int] = None,
) -> ReplenishmentRecommendation:
    """Validate user inputs and call the core replenishment engine.

    Directly delegates calculation to `stockwise.inventory.evaluate_replenishment_scenario`
    without reimplementing inventory formulas.

    Parameters
    ----------
    store : int
        Store ID.
    item : int
        Item ID.
    daily_demand_rate : float
        Expected daily demand rate in units/day.
    demand_variability : float
        Dispersion measure (historical std or forecast RMSE).
    lead_time_days : int
        Supplier fulfillment lead time in days.
    service_level : float, default 0.95
        Cycle service level fraction (0.0 < SL < 1.0).
    lead_time_std_days : float, default 0.0
        Standard deviation of supplier lead time in days.
    min_order_quantity : int, default 1
        Minimum batch order size.
    review_period_days : int, default 1
        Review cycle interval in days.
    on_hand : int, default 0
        Physical units on hand (>= 0).
    on_order : int, default 0
        Units on order (>= 0).
    backorders : int, default 0
        Backordered units (>= 0).
    variability_basis : str, default 'historical_demand_variance'
        Methodological basis for variability.
    residual_sample_size : int, optional
        Validation sample count if basis is forecast error.

    Returns
    -------
    ReplenishmentRecommendation
        Calculated replenishment recommendation with full explainability.
    """
    # 1. Input parameter validation
    if lead_time_days < 1:
        raise InventoryPolicyError(f"lead_time_days must be an integer >= 1, received: {lead_time_days}")

    if not (0.0 < service_level < 1.0):
        raise InventoryPolicyError(
            f"service_level must be strictly between 0.0 and 1.0, received: {service_level}"
        )

    if lead_time_std_days < 0.0:
        raise InventoryPolicyError("Lead time standard deviation cannot be negative.")

    if min_order_quantity < 1:
        raise InventoryPolicyError(f"min_order_quantity must be an integer >= 1, received: {min_order_quantity}")

    if review_period_days < 1:
        raise InventoryPolicyError(f"Review period must be an integer >= 1, received: {review_period_days}")

    if on_hand < 0 or on_order < 0 or backorders < 0:
        raise InventoryPolicyError("Inventory stock quantities (on_hand, on_order, backorders) cannot be negative.")

    # 2. Build Policy & Position Dataclasses
    policy = ReplenishmentPolicy(
        lead_time_days=int(lead_time_days),
        service_level=float(service_level),
        lead_time_std_days=float(lead_time_std_days),
        min_order_quantity=int(min_order_quantity),
        review_period_days=int(review_period_days),
    )

    current_inventory = InventoryPosition(
        on_hand=int(on_hand),
        on_order=int(on_order),
        backorders=int(backorders),
    )

    # 3. Call core inventory module
    return evaluate_replenishment_scenario(
        store=int(store),
        item=int(item),
        daily_demand_rate=float(daily_demand_rate),
        demand_variability=float(demand_variability),
        policy=policy,
        current_inventory=current_inventory,
        variability_basis=variability_basis,
        residual_sample_size=residual_sample_size,
    )


def format_replenishment_results(
    recommendation: ReplenishmentRecommendation,
) -> Dict[str, Any]:
    """Format replenishment recommendation results for clean UI rendering.

    Extracts presentation values and audit trail narratives from the
    ReplenishmentRecommendation dataclass without recalculating any formulas.

    Parameters
    ----------
    recommendation : ReplenishmentRecommendation
        Output from evaluate_replenishment_scenario.

    Returns
    -------
    dict
        Structured formatting payload ready for Streamlit metrics, tables, and alerts.
    """
    expl = recommendation.explanation
    inputs = expl["inputs_used"]
    meaning = expl["output_meaning"]

    status_color_map = {
        "REORDER_IMMEDIATE": "red",
        "ADEQUATE": "green",
        "SURPLUS": "blue",
    }

    raw_deficit = max(0.0, float(recommendation.order_up_to_level - recommendation.current_inventory_position))
    moq = int(inputs["min_order_quantity"])
    batches = int(np.ceil(raw_deficit / moq)) if raw_deficit > 0 else 0

    if recommendation.reorder_required:
        moq_explanation = (
            f"The unrounded replenishment gap (OUTL - IP) is {raw_deficit:.1f} units. "
            f"Accounting for the Minimum Order Quantity constraint of {moq} units, "
            f"the recommended purchase is rounded up to {recommendation.suggested_order_quantity} units "
            f"({batches} batch{'es' if batches > 1 else ''} of {moq})."
        )
    else:
        moq_explanation = (
            f"Net inventory position ({recommendation.current_inventory_position} units) exceeds reorder point "
            f"({recommendation.reorder_point:.1f} units). No purchase order is required."
        )

    return {
        "headline_metrics": {
            "store": recommendation.store,
            "item": recommendation.item,
            "stock_status": recommendation.stock_status,
            "status_color": status_color_map.get(recommendation.stock_status, "gray"),
            "reorder_required": recommendation.reorder_required,
            "suggested_order_quantity": recommendation.suggested_order_quantity,
            "net_inventory_position": recommendation.current_inventory_position,
            "reorder_point": round(recommendation.reorder_point, 1),
            "safety_stock": round(recommendation.safety_stock, 1),
            "lead_time_demand": round(recommendation.lead_time_demand, 1),
            "order_up_to_level": round(recommendation.order_up_to_level, 1),
            "unrounded_deficit": round(raw_deficit, 2),
        },
        "action_recommendation": expl["action_recommendation"],
        "moq_explanation": moq_explanation,
        "parameter_summary": [
            {"Parameter": "Daily Demand Rate (d)", "Value": f"{recommendation.daily_demand_rate:.2f} units/day"},
            {"Parameter": "Demand Dispersion (σ)", "Value": f"{inputs['demand_variability']:.2f} ({recommendation.variability_basis})"},
            {"Parameter": "Supplier Lead Time (L)", "Value": f"{inputs['lead_time_days']} days (std: {inputs['lead_time_std_days']} days)"},
            {"Parameter": "Cycle Service Level (CSL)", "Value": f"{inputs['service_level'] * 100:.1f}% (Z = {inputs['z_score']:.2f})"},
            {"Parameter": "Review Period (R)", "Value": f"{inputs['review_period_days']} day(s)"},
            {"Parameter": "Minimum Order Quantity (MOQ)", "Value": f"{inputs['min_order_quantity']} units"},
            {"Parameter": "Hypothetical On-Hand Stock", "Value": f"{inputs['current_on_hand']} units"},
            {"Parameter": "Hypothetical On-Order Stock", "Value": f"{inputs['current_on_order']} units"},
            {"Parameter": "Hypothetical Backorders", "Value": f"{inputs['current_backorders']} units"},
            {"Parameter": "Net Inventory Position (IP)", "Value": f"{recommendation.current_inventory_position} units"},
            {"Parameter": "Unrounded Replenishment Gap (OUTL - IP)", "Value": f"{raw_deficit:.2f} units"},
            {"Parameter": "Suggested Order (Batched to MOQ)", "Value": f"{recommendation.suggested_order_quantity} units ({batches} x {moq})"},
        ],
        "output_meaning": meaning,
        "assumptions": expl["assumptions"],
        "missing_information": expl["missing_information"],
    }


def build_demand_explorer_chart(
    filtered_df: pd.DataFrame,
    val_preds_df: Optional[pd.DataFrame] = None,
    show_ma7: bool = True,
    date_column: str = "date",
    sales_column: str = "sales",
) -> go.Figure:
    """Build interactive Plotly time series chart for Demand Explorer."""
    fig = go.Figure()

    if filtered_df.empty:
        fig.add_annotation(
            text="No sales observations match the selected filters.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray"),
        )
        fig.update_layout(template="plotly_white", height=420)
        return fig

    df = filtered_df.sort_values(by=date_column).copy()
    dates = pd.to_datetime(df[date_column])

    # 1. Historical Actuals Trace
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=df[sales_column],
            mode="lines",
            name="Observed Daily Sales (Actuals)",
            line=dict(color="#1f77b4", width=1.5),
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Actual Sales:</b> %{y:.0f} units<extra></extra>",
        )
    )

    # 2. 7-Day Moving Average
    if show_ma7 and len(df) >= 7:
        df["ma_7"] = df[sales_column].rolling(window=7, min_periods=1).mean()
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df["ma_7"],
                mode="lines",
                name="7-Day Rolling Average",
                line=dict(color="#64748b", width=2, dash="dot"),
                hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>7-Day Mean:</b> %{y:.1f} units<extra></extra>",
            )
        )

    # 3. Optional Validation Forecasts Trace (Q3 2017)
    if val_preds_df is not None and not val_preds_df.empty:
        v_df = val_preds_df.sort_values(by=date_column).copy()
        v_dates = pd.to_datetime(v_df[date_column])
        min_date, max_date = dates.min(), dates.max()
        v_mask = (v_dates >= min_date) & (v_dates <= max_date)
        v_plot = v_df.loc[v_mask]

        if not v_plot.empty:
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(v_plot[date_column]),
                    y=v_plot["predicted_sales"],
                    mode="lines",
                    name="Validation Forecast (Ridge Regression: Q3 2017)",
                    line=dict(color="#ef4444", width=2.2, dash="dash"),
                    hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Predicted:</b> %{y:.1f} units (Validation Split)<extra></extra>",
                )
            )

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=40, b=40),
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(title="Date", showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(title="Daily Sales (Units)", showgrid=True, gridcolor="#f1f5f9"),
    )
    return fig


def build_model_comparison_chart(comparison_df: pd.DataFrame) -> go.Figure:
    """Build grouped bar chart comparing Validation MAE and RMSE across candidate models."""
    fig = go.Figure()
    if comparison_df.empty:
        return fig

    df = comparison_df.sort_values("Validation MAE", ascending=True)

    fig.add_trace(
        go.Bar(
            y=df["Model"],
            x=df["Validation MAE"],
            name="Validation MAE (Lower is Better)",
            orientation="h",
            marker=dict(color="#3b82f6"),
            text=df["Validation MAE"].round(2).astype(str),
            textposition="auto",
        )
    )

    fig.add_trace(
        go.Bar(
            y=df["Model"],
            x=df["Validation RMSE"],
            name="Validation RMSE (Lower is Better)",
            orientation="h",
            marker=dict(color="#6366f1"),
            text=df["Validation RMSE"].round(2).astype(str),
            textposition="auto",
        )
    )

    fig.update_layout(
        barmode="group",
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(title="Error (Units / Day)", showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def build_inventory_risk_scatter(
    summary_df: pd.DataFrame,
    reference_summary_df: Optional[pd.DataFrame] = None,
    highlight_store: Optional[int] = None,
    highlight_item: Optional[int] = None,
) -> go.Figure:
    """Build interactive scatter plot of Demand Velocity vs Volatility colored by quadrant."""
    fig = go.Figure()

    if summary_df.empty:
        fig.add_annotation(
            text="No store-item series match the selected filter criteria.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray"),
        )
        fig.update_layout(template="plotly_white", height=450)
        return fig

    quadrant_colors = {
        "High Velocity - Volatile (Requires Inventory Review)": "#ef4444",
        "High Velocity - Stable (Steady Fast-Mover)": "#10b981",
        "Low Velocity - Stable (Low Maintenance)": "#3b82f6",
        "Lumpy / Erratic Demand (Intermittent)": "#f59e0b",
    }

    # Plot points by category for legend grouping
    for category, color in quadrant_colors.items():
        sub = summary_df[summary_df["risk_category"] == category]
        if not sub.empty:
            fig.add_trace(
                go.Scatter(
                    x=sub["mean_daily_sales"],
                    y=sub["std_daily_sales"],
                    mode="markers",
                    name=category.split(" (")[0],
                    marker=dict(color=color, size=7, opacity=0.75),
                    customdata=sub[["store", "item", "risk_category", "review_priority", "demand_risk_score"]].values,
                    hovertemplate=(
                        "<b>Store:</b> %{customdata[0]} | <b>Item:</b> %{customdata[1]}<br>"
                        "<b>Velocity (Mean):</b> %{x:.1f} units/day<br>"
                        "<b>Volatility (Std):</b> %{y:.1f} units/day<br>"
                        "<b>Risk Score:</b> %{customdata[4]}<br>"
                        "<b>Priority:</b> %{customdata[3]}<extra></extra>"
                    ),
                )
            )

    # Median threshold lines (computed from reference if provided, otherwise summary_df)
    ref_df = reference_summary_df if reference_summary_df is not None and not reference_summary_df.empty else summary_df
    med_vel = float(ref_df["mean_daily_sales"].median())
    med_vol = float(ref_df["std_daily_sales"].median())

    fig.add_vline(
        x=med_vel,
        line=dict(color="#94a3b8", dash="dash", width=1.5),
        annotation_text=f"Catalog Median Velocity ({med_vel:.1f})",
        annotation_position="top right",
    )
    fig.add_hline(
        y=med_vol,
        line=dict(color="#94a3b8", dash="dot", width=1.5),
        annotation_text=f"Catalog Median Volatility ({med_vol:.1f})",
        annotation_position="bottom right",
    )

    # Highlight selected SKU if provided
    if highlight_store is not None and highlight_item is not None:
        sel = summary_df[(summary_df["store"] == highlight_store) & (summary_df["item"] == highlight_item)]
        if not sel.empty:
            fig.add_trace(
                go.Scatter(
                    x=sel["mean_daily_sales"],
                    y=sel["std_daily_sales"],
                    mode="markers",
                    name=f"Selected (Store {highlight_store}, Item {highlight_item})",
                    marker=dict(color="#000000", size=14, symbol="star", line=dict(color="#ffffff", width=1.5)),
                    hovertemplate=(
                        "<b>TARGET SKU: Store %{customdata[0]} | Item %{customdata[1]}</b><br>"
                        "<b>Mean:</b> %{x:.1f} units/day | <b>Std:</b> %{y:.1f} units/day<extra></extra>"
                    ),
                    customdata=sel[["store", "item"]].values,
                )
            )

    fig.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(title="Demand Velocity (Mean Daily Sales, units/day)", showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(title="Demand Volatility (Daily Sales Std, units/day)", showgrid=True, gridcolor="#f1f5f9"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig
