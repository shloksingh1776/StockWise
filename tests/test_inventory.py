"""Unit tests for the stockwise inventory intelligence module."""

import numpy as np
import pandas as pd
import pytest

from stockwise.inventory import (
    InventoryPolicyError,
    InventoryPosition,
    ReplenishmentPolicy,
    ReplenishmentRecommendation,
    calculate_lead_time_demand,
    calculate_reorder_point,
    calculate_safety_stock,
    calculate_sku_residual_variability,
    evaluate_replenishment_scenario,
    summarize_demand_and_risk,
)


# -------------------------------------------------------------------------
# Lead-Time Demand Tests
# -------------------------------------------------------------------------

def test_lead_time_demand_positive():
    """Test standard lead-time demand calculation (d_L = daily_demand * lead_time)."""
    demand = calculate_lead_time_demand(daily_demand=25.0, lead_time_days=7)
    assert demand == 175.0

    demand_float_lt = calculate_lead_time_demand(daily_demand=12.5, lead_time_days=3.5)
    assert demand_float_lt == 43.75


def test_lead_time_demand_negative_clipping():
    """Test defensive clipping when daily demand prediction is negative."""
    # Negative demand can occur with unconstrained linear models
    demand = calculate_lead_time_demand(daily_demand=-4.3, lead_time_days=7)
    assert demand == 0.0


def test_lead_time_demand_invalid_inputs():
    """Test error handling when lead_time_days is non-positive or NaN."""
    with pytest.raises(InventoryPolicyError, match="lead_time_days must be strictly positive"):
        calculate_lead_time_demand(daily_demand=10.0, lead_time_days=0)

    with pytest.raises(InventoryPolicyError, match="lead_time_days must be strictly positive"):
        calculate_lead_time_demand(daily_demand=10.0, lead_time_days=-3)

    with pytest.raises(InventoryPolicyError, match="daily_demand cannot be NaN"):
        calculate_lead_time_demand(daily_demand=float("nan"), lead_time_days=5)


# -------------------------------------------------------------------------
# Safety Stock Tests
# -------------------------------------------------------------------------

def test_safety_stock_deterministic_lead_time():
    """Test safety stock under deterministic lead time with known normal Z-scores."""
    # 95% service level -> Z ~ 1.6448536
    # lead_time = 4, demand_std = 10.0 -> sqrt(4) * 10 * Z = 20 * 1.6448536 = 32.8971
    ss_95 = calculate_safety_stock(service_level=0.95, lead_time_days=4, demand_std=10.0)
    assert np.isclose(ss_95, 32.8971, atol=1e-3)

    # 90% service level -> Z ~ 1.2815516
    ss_90 = calculate_safety_stock(service_level=0.90, lead_time_days=4, demand_std=10.0)
    assert np.isclose(ss_90, 25.6310, atol=1e-3)

    # 98% service level -> Z ~ 2.0537489
    ss_98 = calculate_safety_stock(service_level=0.98, lead_time_days=4, demand_std=10.0)
    assert np.isclose(ss_98, 41.0750, atol=1e-3)


def test_safety_stock_zero_variability():
    """Test safety stock when demand has zero variability."""
    ss = calculate_safety_stock(service_level=0.95, lead_time_days=7, demand_std=0.0)
    assert ss == 0.0


def test_safety_stock_stochastic_lead_time():
    """Test safety stock when lead time also has standard deviation."""
    # L = 4, sigma_D = 10.0, D_mean = 50.0, sigma_L = 1.0, SL = 0.95 (Z = 1.6448536)
    # Var = 4 * (10^2) + (50^2) * (1^2) = 400 + 2500 = 2900
    # Combined sigma = sqrt(2900) ~ 53.851648
    # SS = 1.6448536 * 53.851648 = 88.5780
    ss = calculate_safety_stock(
        service_level=0.95,
        lead_time_days=4,
        demand_std=10.0,
        daily_demand_mean=50.0,
        lead_time_std_days=1.0,
    )
    assert np.isclose(ss, 88.5780, atol=1e-3)


def test_safety_stock_invalid_inputs():
    """Test error handling for out-of-range service level or negative dispersions."""
    # Service level <= 0 or >= 1
    with pytest.raises(InventoryPolicyError, match="service_level must be strictly between 0.0 and 1.0"):
        calculate_safety_stock(service_level=0.0, lead_time_days=5, demand_std=5.0)

    with pytest.raises(InventoryPolicyError, match="service_level must be strictly between 0.0 and 1.0"):
        calculate_safety_stock(service_level=1.0, lead_time_days=5, demand_std=5.0)

    # Negative dispersions
    with pytest.raises(InventoryPolicyError, match="demand_std cannot be negative"):
        calculate_safety_stock(service_level=0.95, lead_time_days=5, demand_std=-2.0)

    with pytest.raises(InventoryPolicyError, match="lead_time_std_days cannot be negative"):
        calculate_safety_stock(
            service_level=0.95, lead_time_days=5, demand_std=5.0, lead_time_std_days=-1.0
        )


# -------------------------------------------------------------------------
# Reorder Point Tests
# -------------------------------------------------------------------------

def test_reorder_point_calculation():
    """Test ROP = d_L + SS calculation and input validation."""
    rop = calculate_reorder_point(lead_time_demand=150.0, safety_stock=35.0)
    assert rop == 185.0

    with pytest.raises(InventoryPolicyError, match="lead_time_demand cannot be negative"):
        calculate_reorder_point(lead_time_demand=-10.0, safety_stock=20.0)

    with pytest.raises(InventoryPolicyError, match="safety_stock cannot be negative"):
        calculate_reorder_point(lead_time_demand=100.0, safety_stock=-5.0)


# -------------------------------------------------------------------------
# Dataclasses and Policy Validation Tests
# -------------------------------------------------------------------------

def test_replenishment_policy_validation():
    """Test ReplenishmentPolicy rejects invalid parameters."""
    # Valid policy
    policy = ReplenishmentPolicy(lead_time_days=7, service_level=0.95, min_order_quantity=10)
    assert policy.lead_time_days == 7

    # Non-positive lead time
    with pytest.raises(InventoryPolicyError, match="lead_time_days must be an integer >= 1"):
        ReplenishmentPolicy(lead_time_days=0)

    # Invalid service level
    with pytest.raises(InventoryPolicyError, match="service_level must be strictly between"):
        ReplenishmentPolicy(lead_time_days=5, service_level=1.2)

    # Invalid MOQ
    with pytest.raises(InventoryPolicyError, match="min_order_quantity must be an integer >= 1"):
        ReplenishmentPolicy(lead_time_days=5, min_order_quantity=0)


def test_inventory_position_validation_and_net_stock():
    """Test InventoryPosition net calculation and non-negative constraints."""
    pos = InventoryPosition(on_hand=100, on_order=50, backorders=20)
    assert pos.net_effective_stock == 130

    with pytest.raises(InventoryPolicyError, match="on_hand inventory cannot be negative"):
        InventoryPosition(on_hand=-5)

    with pytest.raises(InventoryPolicyError, match="on_order inventory cannot be negative"):
        InventoryPosition(on_hand=10, on_order=-1)

    with pytest.raises(InventoryPolicyError, match="backorders cannot be negative"):
        InventoryPosition(on_hand=10, backorders=-2)


# -------------------------------------------------------------------------
# Scenario Evaluation & MOQ Tests
# -------------------------------------------------------------------------

def test_evaluate_replenishment_adequate_stock():
    """Test scenario where stock exceeds ROP so no reorder is placed."""
    policy = ReplenishmentPolicy(lead_time_days=5, service_level=0.95, min_order_quantity=10)
    # daily_demand = 20, lead_time = 5 -> d_L = 100
    # SS = Z * 10 * sqrt(5) ~ 1.64485 * 22.3606 ~ 36.78
    # ROP ~ 136.78
    inv = InventoryPosition(on_hand=150, on_order=0, backorders=0)  # IP = 150 > 136.78

    rec = evaluate_replenishment_scenario(
        store=1,
        item=1,
        daily_demand_rate=20.0,
        demand_variability=10.0,
        policy=policy,
        current_inventory=inv,
    )

    assert not rec.reorder_required
    assert rec.suggested_order_quantity == 0
    assert rec.stock_status == "ADEQUATE"


def test_evaluate_replenishment_trigger_and_moq_batching():
    """Test reorder trigger when IP <= ROP with MOQ batch rounding."""
    policy = ReplenishmentPolicy(
        lead_time_days=5,
        service_level=0.95,
        min_order_quantity=25,
        review_period_days=2,
    )
    # daily_demand = 20 -> d_L = 100, SS ~ 36.78 -> ROP ~ 136.78
    # cycle_demand = 2 * 20 = 40 -> OUTL = 176.78
    # on_hand = 50, on_order = 0 -> IP = 50 (below ROP)
    # Deficit = 176.78 - 50 = 126.78
    # Batch ceil(126.78 / 25) * 25 = 6 * 25 = 150 units
    inv = InventoryPosition(on_hand=50, on_order=0, backorders=0)

    rec = evaluate_replenishment_scenario(
        store=1,
        item=1,
        daily_demand_rate=20.0,
        demand_variability=10.0,
        policy=policy,
        current_inventory=inv,
    )

    assert rec.reorder_required
    assert rec.stock_status == "REORDER_IMMEDIATE"
    assert rec.suggested_order_quantity == 150
    assert "reorder_point" in rec.explanation["output_meaning"]
    assert "assumptions" in rec.explanation
    assert "missing_information" in rec.explanation


def test_evaluate_replenishment_on_order_prevents_unnecessary_order():
    """Test that incoming on-order shipments prevent double-ordering."""
    policy = ReplenishmentPolicy(lead_time_days=5, service_level=0.95)
    # On-hand is low (30), but 120 units are already on order -> IP = 150 > ROP (~136.78)
    inv = InventoryPosition(on_hand=30, on_order=120, backorders=0)

    rec = evaluate_replenishment_scenario(
        store=1,
        item=1,
        daily_demand_rate=20.0,
        demand_variability=10.0,
        policy=policy,
        current_inventory=inv,
    )

    assert not rec.reorder_required
    assert rec.suggested_order_quantity == 0


def test_evaluate_replenishment_surplus_detection():
    """Test surplus classification when stock significantly exceeds order-up-to level."""
    policy = ReplenishmentPolicy(lead_time_days=3, service_level=0.90)
    # High inventory (500 units) relative to demand (~20 units/day)
    inv = InventoryPosition(on_hand=500, on_order=0, backorders=0)

    rec = evaluate_replenishment_scenario(
        store=1,
        item=1,
        daily_demand_rate=10.0,
        demand_variability=5.0,
        policy=policy,
        current_inventory=inv,
    )

    assert not rec.reorder_required
    assert rec.stock_status == "SURPLUS"


def test_evaluate_replenishment_negative_net_inventory_position():
    """Test replenishment order sizing when backorders exceed on-hand and on-order stock (net position < 0)."""
    policy = ReplenishmentPolicy(
        lead_time_days=5,
        service_level=0.95,
        min_order_quantity=20,
        review_period_days=1,
    )
    # daily_demand = 10.0 -> lead_time_demand = 50.0
    # SS with demand_std = 5.0 -> Z * 5 * sqrt(5) ~ 1.64485 * 11.1803 ~ 18.39
    # ROP = 50.0 + 18.39 = 68.39
    # cycle_demand = 1 * 10 = 10.0 -> OUTL = 78.39
    # on_hand = 10, on_order = 0, backorders = 35 -> net_effective_stock = 10 - 35 = -25
    inv = InventoryPosition(on_hand=10, on_order=0, backorders=35)
    assert inv.net_effective_stock == -25

    rec = evaluate_replenishment_scenario(
        store=1,
        item=1,
        daily_demand_rate=10.0,
        demand_variability=5.0,
        policy=policy,
        current_inventory=inv,
    )

    assert rec.reorder_required
    assert rec.stock_status == "REORDER_IMMEDIATE"
    assert rec.current_inventory_position == -25

    # Deficit = OUTL - net_position = 78.39 - (-25) = 103.39
    # Sizing with MOQ = 20: ceil(103.39 / 20) * 20 = 6 * 20 = 120 units
    expected_deficit = rec.order_up_to_level - (-25)
    expected_order = int(np.ceil(expected_deficit / 20) * 20)
    assert rec.suggested_order_quantity == expected_order
    assert rec.suggested_order_quantity == 120
    assert rec.explanation["inputs_used"]["net_effective_stock"] == -25
    assert rec.explanation["inputs_used"]["current_backorders"] == 35


# -------------------------------------------------------------------------
# Demand & Risk Summarization Tests
# -------------------------------------------------------------------------

@pytest.fixture
def synthetic_risk_panel() -> pd.DataFrame:
    """Create panel of 4 distinct items with controlled velocity and volatility."""
    records = []
    dates = pd.date_range("2017-01-01", periods=60, freq="D")
    rng = np.random.default_rng(42)

    for dt in dates:
        # Item 1: High Velocity, High Volatility (mean ~ 100, noise ~ 35)
        records.append({"date": dt, "store": 1, "item": 1, "sales": max(0.0, 100.0 + rng.normal(0, 35))})
        # Item 2: High Velocity, Low Volatility (mean ~ 100, noise ~ 2)
        records.append({"date": dt, "store": 1, "item": 2, "sales": max(0.0, 100.0 + rng.normal(0, 2))})
        # Item 3: Low Velocity, Low Volatility (mean ~ 10, noise ~ 1)
        records.append({"date": dt, "store": 1, "item": 3, "sales": max(0.0, 10.0 + rng.normal(0, 1))})
        # Item 4: Low Velocity, High Volatility (mean ~ 10, noise ~ 15)
        records.append({"date": dt, "store": 1, "item": 4, "sales": max(0.0, 10.0 + rng.normal(0, 15))})

    return pd.DataFrame(records)


def test_summarize_demand_and_risk_quadrants(synthetic_risk_panel: pd.DataFrame):
    """Test that store-item pairs are properly mapped to velocity/volatility quadrants."""
    summary = summarize_demand_and_risk(synthetic_risk_panel, recent_history_days=30)

    assert len(summary) == 4
    assert set(summary["store"]) == {1}
    assert set(summary["item"]) == {1, 2, 3, 4}

    item_1 = summary[summary["item"] == 1].iloc[0]
    item_2 = summary[summary["item"] == 2].iloc[0]
    item_3 = summary[summary["item"] == 3].iloc[0]
    item_4 = summary[summary["item"] == 4].iloc[0]

    # Item 1: High Velocity & High Volatility
    assert "High Velocity - Volatile" in item_1["risk_category"]
    assert item_1["review_priority"] == "CRITICAL"

    # Item 2: High Velocity & Low Volatility
    assert "High Velocity - Stable" in item_2["risk_category"]
    assert item_2["review_priority"] == "ELEVATED"

    # Item 3: Low Velocity & Low Volatility
    assert "Low Velocity - Stable" in item_3["risk_category"]
    assert item_3["review_priority"] == "STANDARD"

    # Item 4: Low Velocity & High Volatility
    assert "Lumpy / Erratic Demand" in item_4["risk_category"]
    assert item_4["review_priority"] == "LOW"


def test_summarize_demand_and_risk_zero_sales():
    """Test that zero-sales SKUs have zero CV and don't produce division-by-zero errors."""
    dates = pd.date_range("2017-01-01", periods=10, freq="D")
    df = pd.DataFrame({"date": dates, "store": 1, "item": 1, "sales": 0.0})

    summary = summarize_demand_and_risk(df, recent_history_days=10)
    assert len(summary) == 1
    assert summary["mean_daily_sales"].iloc[0] == 0.0
    assert summary["cv_daily_sales"].iloc[0] == 0.0
    assert not np.isnan(summary["cv_daily_sales"].iloc[0])


def test_summarize_demand_and_risk_percentile_ties():
    """Test that identical sales across multiple items handle rank ties gracefully."""
    dates = pd.date_range("2017-01-01", periods=10, freq="D")
    records = []
    for item in [1, 2, 3]:
        for dt in dates:
            records.append({"date": dt, "store": 1, "item": item, "sales": 50.0})
    df = pd.DataFrame(records)

    summary = summarize_demand_and_risk(df, recent_history_days=10)
    assert len(summary) == 3
    # All percentiles should be equal and non-null
    assert summary["demand_velocity_percentile"].nunique() == 1
    assert not summary["demand_risk_score"].isna().any()


def test_summarize_demand_and_risk_validation_errors():
    """Test schema validation in demand summary."""
    df_missing = pd.DataFrame({"date": ["2017-01-01"], "sales": [10]})
    with pytest.raises(InventoryPolicyError, match="Missing required columns"):
        summarize_demand_and_risk(df_missing)

    with pytest.raises(InventoryPolicyError, match="sales_df cannot be empty"):
        summarize_demand_and_risk(pd.DataFrame(columns=["date", "store", "item", "sales"]))

    valid_df = pd.DataFrame({
        "date": ["2017-01-01"],
        "store": [1],
        "item": [1],
        "sales": [10],
    })
    with pytest.raises(InventoryPolicyError, match="recent_history_days must be strictly positive"):
        summarize_demand_and_risk(valid_df, recent_history_days=0)


# -------------------------------------------------------------------------
# SKU Residual Variability Tests
# -------------------------------------------------------------------------

def test_calculate_sku_residual_variability():
    """Test calculation of out-of-sample RMSE and residual std per SKU."""
    val_df = pd.DataFrame({
        "store": [1, 1, 1, 2, 2],
        "item": [1, 1, 1, 1, 1],
        "sales": [10.0, 12.0, 14.0, 20.0, 22.0],
    })
    # Store 1 residuals: [10-11, 12-11, 14-11] = [-1, 1, 3] -> Sq = [1, 1, 9] -> Mean = 11/3 -> RMSE = sqrt(11/3) ~ 1.9149
    # Store 2 residuals: [20-20, 22-20] = [0, 2] -> Sq = [0, 4] -> Mean = 2 -> RMSE = sqrt(2) ~ 1.4142
    val_preds = np.array([11.0, 11.0, 11.0, 20.0, 20.0])

    metrics = calculate_sku_residual_variability(val_df, val_preds)

    assert len(metrics) == 2
    s1 = metrics[(metrics["store"] == 1) & (metrics["item"] == 1)].iloc[0]
    assert np.isclose(s1["val_residual_rmse"], np.sqrt(11.0 / 3.0), atol=1e-3)
    assert s1["val_observation_count"] == 3

    s2 = metrics[(metrics["store"] == 2) & (metrics["item"] == 1)].iloc[0]
    assert np.isclose(s2["val_residual_rmse"], np.sqrt(2.0), atol=1e-3)
    assert s2["val_observation_count"] == 2


def test_calculate_sku_residual_variability_mismatched_length():
    """Test error when val_df length does not match prediction length."""
    val_df = pd.DataFrame({"store": [1], "item": [1], "sales": [10.0]})
    val_preds = np.array([10.0, 20.0])
    with pytest.raises(InventoryPolicyError, match="does not match val_predictions length"):
        calculate_sku_residual_variability(val_df, val_preds)
