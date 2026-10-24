"""Inventory Intelligence Module for StockWise (Milestone 3).

Provides pure, testable inventory calculation functions, policy dataclasses,
demand-risk categorization, and structured explainability for retail replenishment.

Important Methodology Notes:
----------------------------
1. Pure Sales Data Limitation:
   The underlying dataset contains historical sales transactions, NOT operational
   inventory balances, supplier lead times, stockouts, or holding costs.
   All replenishment metrics are parameterized scenario evaluations, not verified
   purchase orders.

2. One-Day-Ahead Forecasting Limitation:
   The forecasting model provides daily refreshed 1-day-ahead predictions.
   Lead-time demand calculations (d_L = daily_demand * lead_time_days) assume that
   the predicted daily demand rate remains constant over the lead-time window.

3. Risk Scoring as Prioritization Heuristic:
   Demand risk scores rank SKUs by empirical demand velocity and volatility
   percentiles to help managers prioritize reviews. They are NOT calibrated
   probabilities of stockout.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import norm


class InventoryPolicyError(ValueError):
    """Exception raised when inventory policy inputs or scenario parameters are invalid."""

    pass


@dataclass(frozen=True)
class ReplenishmentPolicy:
    """Explicit parameters governing a replenishment scenario.

    Parameters
    ----------
    lead_time_days : int
        Supplier fulfillment lead time in integer calendar days (must be >= 1).
    service_level : float, default 0.95
        Target cycle service level as a fraction, e.g. 0.95 for 95% (must be 0 < SL < 1).
    lead_time_std_days : float, default 0.0
        Standard deviation of supplier fulfillment lead time in days (must be >= 0.0).
    min_order_quantity : int, default 1
        Minimum order quantity / batch size constraint (must be >= 1).
    review_period_days : int, default 1
        Review cycle interval in days (default 1 for daily review, must be >= 1).
    """

    lead_time_days: int
    service_level: float = 0.95
    lead_time_std_days: float = 0.0
    min_order_quantity: int = 1
    review_period_days: int = 1

    def __post_init__(self) -> None:
        """Validate policy input ranges."""
        if not isinstance(self.lead_time_days, (int, np.integer)) or self.lead_time_days < 1:
            raise InventoryPolicyError(
                f"lead_time_days must be an integer >= 1, received: {self.lead_time_days}"
            )
        if not (0.0 < self.service_level < 1.0):
            raise InventoryPolicyError(
                f"service_level must be strictly between 0.0 and 1.0 (exclusive), received: {self.service_level}"
            )
        if self.lead_time_std_days < 0.0:
            raise InventoryPolicyError(
                f"lead_time_std_days cannot be negative, received: {self.lead_time_std_days}"
            )
        if not isinstance(self.min_order_quantity, (int, np.integer)) or self.min_order_quantity < 1:
            raise InventoryPolicyError(
                f"min_order_quantity must be an integer >= 1, received: {self.min_order_quantity}"
            )
        if not isinstance(self.review_period_days, (int, np.integer)) or self.review_period_days < 1:
            raise InventoryPolicyError(
                f"review_period_days must be an integer >= 1, received: {self.review_period_days}"
            )


@dataclass(frozen=True)
class InventoryPosition:
    """User-supplied operational inventory state for a store-item combination.

    Parameters
    ----------
    on_hand : int
        Physical units currently in stock (must be >= 0).
    on_order : int, default 0
        Units ordered from supplier but not yet received (must be >= 0).
    backorders : int, default 0
        Committed/unfulfilled customer orders (must be >= 0).
    """

    on_hand: int
    on_order: int = 0
    backorders: int = 0

    def __post_init__(self) -> None:
        """Validate stock quantities are non-negative."""
        if self.on_hand < 0:
            raise InventoryPolicyError(f"on_hand inventory cannot be negative, received: {self.on_hand}")
        if self.on_order < 0:
            raise InventoryPolicyError(f"on_order inventory cannot be negative, received: {self.on_order}")
        if self.backorders < 0:
            raise InventoryPolicyError(f"backorders cannot be negative, received: {self.backorders}")

    @property
    def net_effective_stock(self) -> int:
        """Net inventory position = on_hand + on_order - backorders."""
        return self.on_hand + self.on_order - self.backorders


@dataclass(frozen=True)
class ReplenishmentRecommendation:
    """Calculated replenishment scenario results with full decision explainability."""

    store: int
    item: int
    daily_demand_rate: float
    lead_time_demand: float
    safety_stock: float
    reorder_point: float
    order_up_to_level: float
    current_inventory_position: int
    reorder_required: bool
    suggested_order_quantity: int
    stock_status: str  # "REORDER_IMMEDIATE", "ADEQUATE", "SURPLUS"
    variability_basis: str  # "historical_demand_variance" or "forecast_residual_error"
    explanation: Dict[str, Any]


def calculate_lead_time_demand(
    daily_demand: float,
    lead_time_days: int | float,
) -> float:
    """Calculate expected demand over supplier lead time.

    Applies defensive non-negative clipping to daily demand:
    d_L = max(0.0, daily_demand) * lead_time_days

    Parameters
    ----------
    daily_demand : float
        Expected daily demand rate in units/day.
    lead_time_days : int or float
        Supplier fulfillment lead time in days (must be > 0).

    Returns
    -------
    float
        Expected lead-time demand in units.

    Raises
    ------
    InventoryPolicyError
        If lead_time_days <= 0 or is NaN.
    """
    if math.isnan(daily_demand):
        raise InventoryPolicyError("daily_demand cannot be NaN.")
    if math.isnan(lead_time_days) or lead_time_days <= 0:
        raise InventoryPolicyError(
            f"lead_time_days must be strictly positive (> 0), received: {lead_time_days}"
        )

    # Defensive non-negative demand clipping
    effective_daily_demand = max(0.0, float(daily_demand))
    return float(round(effective_daily_demand * lead_time_days, 4))


def calculate_safety_stock(
    service_level: float,
    lead_time_days: int | float,
    demand_std: float,
    daily_demand_mean: float = 0.0,
    lead_time_std_days: float = 0.0,
) -> float:
    """Calculate safety stock buffer using classical inventory theory.

    When lead time is deterministic (lead_time_std_days == 0.0):
        SS = Z * demand_std * sqrt(lead_time_days)
    When lead time is variable (lead_time_std_days > 0.0):
        SS = Z * sqrt(lead_time_days * (demand_std ** 2) + (daily_demand_mean ** 2) * (lead_time_std_days ** 2))

    Where Z is the inverse cumulative distribution function of the standard normal
    distribution at the specified cycle service level (norm.ppf(service_level)).

    Parameters
    ----------
    service_level : float
        Target cycle service level fraction (0.0 < service_level < 1.0).
    lead_time_days : int or float
        Average supplier fulfillment lead time in days (> 0).
    demand_std : float
        Daily demand dispersion in units/day (standard deviation of daily sales
        or standard deviation of forecast residuals; must be >= 0.0).
    daily_demand_mean : float, default 0.0
        Average daily demand rate in units/day (must be >= 0.0).
    lead_time_std_days : float, default 0.0
        Standard deviation of lead time in days (must be >= 0.0).

    Returns
    -------
    float
        Required safety stock buffer in units.

    Raises
    ------
    InventoryPolicyError
        If service_level is outside (0, 1), lead_time_days <= 0, or dispersions are negative.
    """
    if math.isnan(service_level) or not (0.0 < service_level < 1.0):
        raise InventoryPolicyError(
            f"service_level must be strictly between 0.0 and 1.0, received: {service_level}"
        )
    if math.isnan(lead_time_days) or lead_time_days <= 0:
        raise InventoryPolicyError(
            f"lead_time_days must be strictly positive (> 0), received: {lead_time_days}"
        )
    if math.isnan(demand_std) or demand_std < 0.0:
        raise InventoryPolicyError(f"demand_std cannot be negative or NaN, received: {demand_std}")
    if math.isnan(daily_demand_mean) or daily_demand_mean < 0.0:
        raise InventoryPolicyError(
            f"daily_demand_mean cannot be negative or NaN, received: {daily_demand_mean}"
        )
    if math.isnan(lead_time_std_days) or lead_time_std_days < 0.0:
        raise InventoryPolicyError(
            f"lead_time_std_days cannot be negative or NaN, received: {lead_time_std_days}"
        )

    z = float(norm.ppf(service_level))

    if lead_time_std_days == 0.0:
        # Deterministic lead time formula
        variance_lead_time = float(lead_time_days) * (demand_std ** 2)
    else:
        # Combined demand and lead time variability formula
        variance_lead_time = (float(lead_time_days) * (demand_std ** 2)) + (
            (daily_demand_mean ** 2) * (lead_time_std_days ** 2)
        )

    safety_stock = z * math.sqrt(variance_lead_time)
    return float(round(max(0.0, safety_stock), 4))


def calculate_reorder_point(
    lead_time_demand: float,
    safety_stock: float,
) -> float:
    """Calculate reorder point threshold (ROP = d_L + SS).

    Parameters
    ----------
    lead_time_demand : float
        Expected demand over lead time (must be >= 0.0).
    safety_stock : float
        Safety stock buffer (must be >= 0.0).

    Returns
    -------
    float
        Reorder point threshold in units.

    Raises
    ------
    InventoryPolicyError
        If inputs are negative or NaN.
    """
    if math.isnan(lead_time_demand) or lead_time_demand < 0.0:
        raise InventoryPolicyError(
            f"lead_time_demand cannot be negative or NaN, received: {lead_time_demand}"
        )
    if math.isnan(safety_stock) or safety_stock < 0.0:
        raise InventoryPolicyError(
            f"safety_stock cannot be negative or NaN, received: {safety_stock}"
        )

    return float(round(lead_time_demand + safety_stock, 4))


def evaluate_replenishment_scenario(
    store: int,
    item: int,
    daily_demand_rate: float,
    demand_variability: float,
    policy: ReplenishmentPolicy,
    current_inventory: InventoryPosition,
    variability_basis: str = "historical_demand_variance",
    residual_sample_size: Optional[int] = None,
) -> ReplenishmentRecommendation:
    """Evaluate replenishment status for a store-item combination under a specific policy.

    Implements a continuous-review (s, S) replenishment policy:
    1. Reorder Point (s) = d_L + SS
    2. Order-Up-To Level (S) = ROP + (review_period_days * daily_demand_rate)
    3. Net Inventory Position (IP) = on_hand + on_order - backorders
    4. If IP <= ROP:
         Deficit = OUTL - IP
         Order Quantity = ceil(Deficit / MOQ) * MOQ
       Else:
         Order Quantity = 0

    Parameters
    ----------
    store : int
        Store identifier.
    item : int
        Item/SKU identifier.
    daily_demand_rate : float
        Forecasted or recent average daily demand rate (units/day).
    demand_variability : float
        Dispersion measure (historical sales std or forecast residual RMSE).
    policy : ReplenishmentPolicy
        Replenishment constraints and service level target.
    current_inventory : InventoryPosition
        User-supplied on-hand, on-order, and backordered quantities.
    variability_basis : str, default 'historical_demand_variance'
        Methodological basis for variability ('historical_demand_variance' or
        'forecast_residual_error').
    residual_sample_size : int, optional
        Number of residual observations if variability_basis is 'forecast_residual_error'.

    Returns
    -------
    ReplenishmentRecommendation
        Comprehensive recommendation with mathematical audit trail.
    """
    # 1. Calculate Core Inventory Parameters
    lead_time_demand = calculate_lead_time_demand(
        daily_demand=daily_demand_rate,
        lead_time_days=policy.lead_time_days,
    )

    safety_stock = calculate_safety_stock(
        service_level=policy.service_level,
        lead_time_days=policy.lead_time_days,
        demand_std=demand_variability,
        daily_demand_mean=max(0.0, float(daily_demand_rate)),
        lead_time_std_days=policy.lead_time_std_days,
    )

    reorder_point = calculate_reorder_point(
        lead_time_demand=lead_time_demand,
        safety_stock=safety_stock,
    )

    # Order-Up-To Level (target max inventory covering review cycle)
    effective_daily_rate = max(0.0, float(daily_demand_rate))
    cycle_demand = policy.review_period_days * effective_daily_rate
    order_up_to_level = float(round(reorder_point + cycle_demand, 4))

    # 2. Evaluate Current Inventory Position
    net_position = current_inventory.net_effective_stock
    reorder_required = net_position <= reorder_point

    if reorder_required:
        net_deficit = order_up_to_level - net_position
        # Batch to Minimum Order Quantity (MOQ)
        moq = policy.min_order_quantity
        raw_order = max(0.0, net_deficit)
        suggested_order_quantity = int(math.ceil(raw_order / moq) * moq)
        stock_status = "REORDER_IMMEDIATE"
    else:
        suggested_order_quantity = 0
        if net_position > order_up_to_level * 1.5:
            stock_status = "SURPLUS"
        else:
            stock_status = "ADEQUATE"

    # 3. Construct Explainability Payload
    z_score = float(round(norm.ppf(policy.service_level), 4))
    explanation = {
        "inputs_used": {
            "store": store,
            "item": item,
            "daily_demand_rate": round(effective_daily_rate, 4),
            "demand_variability": round(demand_variability, 4),
            "variability_basis": variability_basis,
            "residual_sample_size": residual_sample_size,
            "lead_time_days": policy.lead_time_days,
            "lead_time_std_days": policy.lead_time_std_days,
            "service_level": policy.service_level,
            "z_score": z_score,
            "min_order_quantity": policy.min_order_quantity,
            "review_period_days": policy.review_period_days,
            "current_on_hand": current_inventory.on_hand,
            "current_on_order": current_inventory.on_order,
            "current_backorders": current_inventory.backorders,
            "net_effective_stock": net_position,
        },
        "output_meaning": {
            "lead_time_demand": f"Expected demand across the {policy.lead_time_days}-day supplier lead time ({lead_time_demand:.1f} units).",
            "safety_stock": f"Protective stock buffer against demand spikes ({safety_stock:.1f} units) for {policy.service_level * 100:.1f}% cycle service level.",
            "reorder_point": f"Stock threshold triggering an order ({reorder_point:.1f} units).",
            "order_up_to_level": f"Target inventory level covering lead time and {policy.review_period_days}-day review cycle ({order_up_to_level:.1f} units).",
            "suggested_order_quantity": f"Recommended purchase batch ({suggested_order_quantity} units) accounting for MOQ {policy.min_order_quantity}.",
            "stock_status": stock_status,
        },
        "assumptions": [
            f"Daily demand rate ({effective_daily_rate:.2f} units/day) remains constant over the {policy.lead_time_days}-day lead time.",
            "Demand fluctuations or forecast errors are approximately normally distributed around the mean.",
            f"Supplier lead time is {'fixed at ' + str(policy.lead_time_days) + ' days' if policy.lead_time_std_days == 0 else 'variable with standard deviation ' + str(policy.lead_time_std_days) + ' days'}.",
            f"Target cycle service level is {policy.service_level * 100:.1f}%, accepting a {(1 - policy.service_level) * 100:.1f}% stockout risk per order cycle.",
            "Current on-hand inventory and supplier delivery status are hypothetical inputs provided for scenario analysis.",
        ],
        "missing_information": [
            "Actual on-hand inventory balances and warehouse location data.",
            "Historical supplier fulfillment records and real delivery delay variance.",
            "Purchase order cost, transportation freight rates, and unit inventory holding cost.",
            "Shelf space, pallet constraints, and product expiration / shelf-life limits.",
            "Confirmed historical stockout occurrences or unobserved lost sales.",
        ],
        "action_recommendation": (
            f"Place purchase order for {suggested_order_quantity} units immediately. "
            f"Net inventory ({net_position}) has breached reorder point ({reorder_point:.1f})."
            if reorder_required
            else (
                f"Maintain current stock monitoring. Net inventory ({net_position}) exceeds "
                f"reorder point ({reorder_point:.1f}). No purchase order required."
            )
        ),
    }

    return ReplenishmentRecommendation(
        store=store,
        item=item,
        daily_demand_rate=effective_daily_rate,
        lead_time_demand=lead_time_demand,
        safety_stock=safety_stock,
        reorder_point=reorder_point,
        order_up_to_level=order_up_to_level,
        current_inventory_position=net_position,
        reorder_required=reorder_required,
        suggested_order_quantity=suggested_order_quantity,
        stock_status=stock_status,
        variability_basis=variability_basis,
        explanation=explanation,
    )


def summarize_demand_and_risk(
    sales_df: pd.DataFrame,
    forecast_df: Optional[pd.DataFrame] = None,
    recent_history_days: int = 28,
    store_column: str = "store",
    item_column: str = "item",
    date_column: str = "date",
    sales_column: str = "sales",
    forecast_column: str = "predicted_sales",
) -> pd.DataFrame:
    """Summarize historical sales patterns and demand-risk categories by store-item.

    Computes empirical demand velocity (mean daily sales), demand volatility (standard
    deviation), and relative dispersion (coefficient of variation).
    Ranks store-item pairs into empirical velocity/volatility quadrants and assigns
    review priorities.

    IMPORTANT METHODOLOGICAL NOTE:
    Demand risk scores and quadrant labels are prioritization heuristics to help
    retail managers allocate review attention, NOT calibrated stockout probabilities.
    No historical stockout records exist in this sales dataset.

    Parameters
    ----------
    sales_df : pd.DataFrame
        Historical sales dataset.
    forecast_df : pd.DataFrame, optional
        Predicted sales DataFrame containing out-of-sample forecasts.
    recent_history_days : int, default 28
        Number of recent days in sales_df used to calculate recent velocity/volatility.
    store_column : str, default 'store'
        Store identifier column name.
    item_column : str, default 'item'
        Item/SKU identifier column name.
    date_column : str, default 'date'
        Date column name.
    sales_column : str, default 'sales'
        Sales target column name.
    forecast_column : str, default 'predicted_sales'
        Column name in forecast_df containing predicted daily sales.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per (store, item) containing:
        - store, item
        - mean_daily_sales
        - std_daily_sales
        - cv_daily_sales
        - total_period_sales
        - forecast_daily_demand (if forecast_df provided)
        - demand_velocity_percentile (0.0 to 100.0)
        - demand_volatility_percentile (0.0 to 100.0)
        - demand_risk_score (0.0 to 100.0)
        - risk_category (Quadrant label)
        - review_priority ("CRITICAL", "ELEVATED", "STANDARD", "LOW")

    Raises
    ------
    InventoryPolicyError
        If required columns are missing, DataFrame is empty, or inputs are invalid.
    """
    required_cols = [date_column, store_column, item_column, sales_column]
    missing = [c for c in required_cols if c not in sales_df.columns]
    if missing:
        raise InventoryPolicyError(f"Missing required columns in sales_df: {missing}")

    if sales_df.empty:
        raise InventoryPolicyError("sales_df cannot be empty.")

    if recent_history_days <= 0:
        raise InventoryPolicyError(
            f"recent_history_days must be strictly positive, received: {recent_history_days}"
        )

    df = sales_df.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    # Filter to recent history window based on maximum date present
    max_date = df[date_column].max()
    min_recent_date = max_date - pd.Timedelta(days=recent_history_days - 1)
    recent_df = df[df[date_column] >= min_recent_date].copy()

    if recent_df.empty:
        raise InventoryPolicyError(
            f"No data points found in recent history window >= {min_recent_date}."
        )

    # Compute group statistics
    grouped = recent_df.groupby([store_column, item_column])[sales_column]
    summary = grouped.agg(
        mean_daily_sales="mean",
        std_daily_sales="std",
        total_period_sales="sum",
        observed_days="count",
    ).reset_index()

    # Fill NaN std (in case of single observation or perfectly constant sales)
    summary["std_daily_sales"] = summary["std_daily_sales"].fillna(0.0)

    # Coefficient of Variation (handle zero mean safely)
    summary["cv_daily_sales"] = np.where(
        summary["mean_daily_sales"] > 0,
        summary["std_daily_sales"] / summary["mean_daily_sales"],
        0.0,
    )

    # Attach forecast demand if forecast_df is provided
    if forecast_df is not None:
        f_cols = [store_column, item_column]
        if forecast_column in forecast_df.columns:
            f_grouped = forecast_df.groupby(f_cols)[forecast_column].mean().reset_index()
            f_grouped.rename(columns={forecast_column: "forecast_daily_demand"}, inplace=True)
            # Clip negative predictions defensively
            f_grouped["forecast_daily_demand"] = f_grouped["forecast_daily_demand"].clip(lower=0.0)
            summary = summary.merge(f_grouped, on=f_cols, how="left")
            summary["forecast_daily_demand"] = summary["forecast_daily_demand"].fillna(
                summary["mean_daily_sales"]
            )
        else:
            summary["forecast_daily_demand"] = summary["mean_daily_sales"]
    else:
        summary["forecast_daily_demand"] = summary["mean_daily_sales"]

    # Calculate empirical percentiles (0 to 100) using average rank for ties
    summary["demand_velocity_percentile"] = (
        summary["mean_daily_sales"].rank(pct=True, method="average") * 100.0
    ).round(2)
    summary["demand_volatility_percentile"] = (
        summary["std_daily_sales"].rank(pct=True, method="average") * 100.0
    ).round(2)

    # Prioritization Heuristic Score (equal weighting of volume and volatility)
    summary["demand_risk_score"] = (
        0.5 * summary["demand_velocity_percentile"]
        + 0.5 * summary["demand_volatility_percentile"]
    ).round(2)

    # Quadrant Classification
    # Strictly above median (> 50.0) defines the top half, ensuring equal partitioning
    is_high_velocity = summary["demand_velocity_percentile"] > 50.0
    is_high_volatility = summary["demand_volatility_percentile"] > 50.0

    conditions = [
        is_high_velocity & is_high_volatility,      # Quadrant I: High Vel, High Vol
        is_high_velocity & (~is_high_volatility),   # Quadrant II: High Vel, Low Vol
        (~is_high_velocity) & (~is_high_volatility),# Quadrant III: Low Vel, Low Vol
        (~is_high_velocity) & is_high_volatility,   # Quadrant IV: Low Vel, High Vol
    ]

    categories = [
        "High Velocity - Volatile (Requires Inventory Review)",
        "High Velocity - Stable (Steady Fast-Mover)",
        "Low Velocity - Stable (Low Maintenance)",
        "Lumpy / Erratic Demand (Intermittent)",
    ]

    priorities = [
        "CRITICAL",   # High volume + high uncertainty
        "ELEVATED",   # High volume revenue drivers
        "STANDARD",   # Low maintenance base
        "LOW",        # Low volume erratic
    ]

    summary["risk_category"] = np.select(conditions, categories, default="Standard")
    summary["review_priority"] = np.select(conditions, priorities, default="STANDARD")

    # Round statistics for clean consumption
    summary["mean_daily_sales"] = summary["mean_daily_sales"].round(4)
    summary["std_daily_sales"] = summary["std_daily_sales"].round(4)
    summary["cv_daily_sales"] = summary["cv_daily_sales"].round(4)
    summary["forecast_daily_demand"] = summary["forecast_daily_demand"].round(4)

    return summary.sort_values("demand_risk_score", ascending=False).reset_index(drop=True)


def calculate_sku_residual_variability(
    val_df: pd.DataFrame,
    val_predictions: np.ndarray | pd.Series,
    store_column: str = "store",
    item_column: str = "item",
    sales_column: str = "sales",
) -> pd.DataFrame:
    """Calculate SKU-specific out-of-sample residual error dispersion on validation data.

    Calculates the Root Mean Squared Error (RMSE) and residual standard deviation
    per (store, item) pair exclusively on the validation period (e.g. Q3 2017).
    This provides an empirical, store-item-specific variability metric for
    forecast-error-based safety stock.

    IMPORTANT METHODOLOGICAL LIMITATIONS:
    1. Validation-Period Specificity:
       These residuals reflect model errors during the validation window (e.g.,
       92 days in summer Q3 2017). They may not fully capture off-peak or holiday
       error dispersion.
    2. Sample Size:
       Each store-item pair typically has N ~ 92 daily observations in the validation
       split. For low-demand items, small sample sizes can lead to noisy error estimates.
    3. Untouched Test Set Preservation:
       These metrics are computed on the validation split so the final test split
       remains completely untouched and un-tuned.

    Parameters
    ----------
    val_df : pd.DataFrame
        Validation split DataFrame.
    val_predictions : array-like
        Out-of-sample predictions corresponding to val_df rows.
    store_column : str, default 'store'
        Store identifier column name.
    item_column : str, default 'item'
        Item identifier column name.
    sales_column : str, default 'sales'
        Actual sales column name.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - store, item
        - val_residual_rmse: float
        - val_residual_std: float
        - val_residual_mean: float (bias)
        - val_observation_count: int
    """
    if len(val_df) != len(val_predictions):
        raise InventoryPolicyError(
            f"val_df length ({len(val_df)}) does not match val_predictions length ({len(val_predictions)})."
        )

    df = val_df[[store_column, item_column, sales_column]].copy()
    df["predicted"] = np.asarray(val_predictions, dtype=float)
    df["residual"] = df[sales_column].astype(float) - df["predicted"]

    def compute_metrics(group: pd.DataFrame) -> pd.Series:
        res = group["residual"].to_numpy()
        rmse = float(np.sqrt(np.mean(res ** 2)))
        std = float(np.std(res, ddof=1)) if len(res) > 1 else 0.0
        bias = float(np.mean(res))
        count = int(len(res))
        return pd.Series({
            "val_residual_rmse": round(rmse, 4),
            "val_residual_std": round(std, 4),
            "val_residual_mean": round(bias, 4),
            "val_observation_count": count,
        })

    metrics_df = df.groupby([store_column, item_column]).apply(
        compute_metrics, include_groups=False
    ).reset_index()

    return metrics_df
