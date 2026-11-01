"""Unit and integration tests for StockWise dashboard utilities (Milestone 4).

Tests data loaders, caching behavior, store-item filtering, overview KPI
aggregation, demand risk preparation, replenishment adapter calculation parity,
and scenario formatting.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from stockwise.dashboard_utils import (
    HOLDOUT_TEST_BENCHMARK,
    build_demand_explorer_chart,
    build_inventory_risk_scatter,
    build_model_comparison_chart,
    compute_overview_kpis,
    evaluate_scenario_adapter,
    filter_demand_risk_data,
    filter_sales_data,
    format_replenishment_results,
    get_date_range,
    get_default_data_path,
    get_default_reports_dir,
    get_holdout_test_metrics,
    get_project_root,
    get_sku_demand_and_variability,
    get_store_and_item_options,
    load_dashboard_sales_data,
    load_model_comparison,
    load_saved_replenishment_scenarios,
    prepare_demand_risk_data,
)
from stockwise.inventory import (
    InventoryPolicyError,
    InventoryPosition,
    ReplenishmentPolicy,
    evaluate_replenishment_scenario,
)


@pytest.fixture
def sample_sales_df() -> pd.DataFrame:
    """Create a minimal synthetic multi-store multi-item sales DataFrame."""
    dates = pd.date_range("2017-01-01", periods=10, freq="D")
    records = []
    for d in dates:
        records.append({"date": d, "store": 1, "item": 1, "sales": 10.0})
        records.append({"date": d, "store": 1, "item": 2, "sales": 20.0})
        records.append({"date": d, "store": 2, "item": 1, "sales": 30.0})
        records.append({"date": d, "store": 2, "item": 2, "sales": 40.0})
    return pd.DataFrame(records)


def test_get_project_root_and_paths() -> None:
    """Verify project root resolution and default directory paths."""
    root = get_project_root()
    assert root.is_dir()
    assert (root / "src" / "stockwise").is_dir()
    assert (root / "reports").is_dir()

    data_path = get_default_data_path()
    assert data_path.name == "train.csv"

    reports_dir = get_default_reports_dir()
    assert reports_dir.name == "reports"


def test_load_dashboard_sales_data_real_file() -> None:
    """Verify loading real sales dataset from data/raw/train.csv with caching."""
    df1 = load_dashboard_sales_data(use_cache=True)
    assert not df1.empty
    assert "date" in df1.columns
    assert "sales" in df1.columns
    assert "store" in df1.columns
    assert "item" in df1.columns
    assert pd.api.types.is_datetime64_any_dtype(df1["date"])

    # Test cache reuse
    df2 = load_dashboard_sales_data(use_cache=True)
    assert len(df1) == len(df2)


def test_load_reports_artifacts() -> None:
    """Verify loading pre-computed model comparison and replenishment scenario reports."""
    mc = load_model_comparison()
    assert not mc.empty
    assert "Model" in mc.columns
    assert "Validation MAE" in mc.columns
    assert "Ridge Regression" in mc["Model"].values

    # Test sorting by Validation MAE
    assert mc["Validation MAE"].is_monotonic_increasing

    # Test non-existent file handling
    with pytest.raises(FileNotFoundError):
        load_model_comparison("non_existent_file.csv")

    sc = load_saved_replenishment_scenarios()
    assert not sc.empty
    assert "store" in sc.columns
    assert "item" in sc.columns
    assert "lead_time_demand" in sc.columns

    with pytest.raises(FileNotFoundError):
        load_saved_replenishment_scenarios("non_existent_scenarios.csv")


def test_get_holdout_test_metrics() -> None:
    """Verify untouched holdout test metrics structure and immutability."""
    metrics = get_holdout_test_metrics()
    assert "test_period" in metrics
    assert "ridge_regression" in metrics
    assert "seasonal_naive_baseline" in metrics
    assert metrics["ridge_regression"]["mae"] == 6.7529
    assert metrics["seasonal_naive_baseline"]["mae"] == 9.0763

    # Ensure returned dict is a copy
    metrics["test_period"] = "modified"
    assert HOLDOUT_TEST_BENCHMARK["test_period"] != "modified"


def test_get_store_and_item_options(sample_sales_df: pd.DataFrame) -> None:
    """Verify unique store and item extraction."""
    stores, items = get_store_and_item_options(sample_sales_df)
    assert stores == [1, 2]
    assert items == [1, 2]

    # Test missing columns error
    with pytest.raises(ValueError, match="must contain 'store' and 'item' columns"):
        get_store_and_item_options(pd.DataFrame({"col_a": [1, 2]}))


def test_get_date_range(sample_sales_df: pd.DataFrame) -> None:
    """Verify extraction of min and max timestamps."""
    start_dt, end_dt = get_date_range(sample_sales_df)
    assert start_dt == pd.Timestamp("2017-01-01")
    assert end_dt == pd.Timestamp("2017-01-10")

    with pytest.raises(ValueError, match="Cannot extract date range from an empty DataFrame"):
        get_date_range(pd.DataFrame())

    with pytest.raises(ValueError, match="not found in DataFrame"):
        get_date_range(sample_sales_df, date_column="non_existent")


def test_filter_sales_data(sample_sales_df: pd.DataFrame) -> None:
    """Verify filtering by store, item, and date ranges."""
    # 1. Filter by store
    res_store = filter_sales_data(sample_sales_df, store=1)
    assert set(res_store["store"].unique()) == {1}
    assert len(res_store) == 20

    # 2. Filter by store and item
    res_sku = filter_sales_data(sample_sales_df, store=1, item=2)
    assert len(res_sku) == 10
    assert (res_sku["item"] == 2).all()

    # 3. Filter by date range (string and timestamp)
    res_date = filter_sales_data(
        sample_sales_df,
        store=1,
        item=1,
        start_date="2017-01-03",
        end_date=pd.Timestamp("2017-01-06"),
    )
    assert len(res_date) == 4
    assert res_date["date"].min() == pd.Timestamp("2017-01-03")
    assert res_date["date"].max() == pd.Timestamp("2017-01-06")

    # 4. Filter with empty result
    res_empty = filter_sales_data(sample_sales_df, store=999)
    assert res_empty.empty
    assert list(res_empty.columns) == list(sample_sales_df.columns)

    # 5. Filter on empty dataframe
    res_from_empty = filter_sales_data(pd.DataFrame(), store=1)
    assert res_from_empty.empty


def test_compute_overview_kpis(sample_sales_df: pd.DataFrame) -> None:
    """Verify executive overview KPIs and dataset boundary metadata."""
    kpis = compute_overview_kpis(sample_sales_df)
    assert kpis["total_sales"] == 1000  # (10+20+30+40)*10 = 1000
    assert kpis["avg_daily_sales"] == 25.0
    assert kpis["total_stores"] == 2
    assert kpis["total_items"] == 2
    assert kpis["total_series"] == 4
    assert kpis["total_days"] == 10
    assert kpis["start_date"] == "2017-01-01"
    assert kpis["end_date"] == "2017-01-10"
    assert "limitations" in kpis
    assert len(kpis["limitations"]["what_dataset_cannot_establish"]) > 0

    # Empty DataFrame KPIs
    empty_kpis = compute_overview_kpis(pd.DataFrame())
    assert empty_kpis["total_sales"] == 0
    assert empty_kpis["avg_daily_sales"] == 0.0

    # With summary_df review priorities
    mock_summary = pd.DataFrame({
        "review_priority": ["CRITICAL", "CRITICAL", "ELEVATED", "STANDARD", "LOW"]
    })
    kpis_with_summary = compute_overview_kpis(sample_sales_df, summary_df=mock_summary)
    assert kpis_with_summary["critical_risk_count"] == 2
    assert kpis_with_summary["elevated_risk_count"] == 1
    assert kpis_with_summary["standard_risk_count"] == 1
    assert kpis_with_summary["low_risk_count"] == 1


def test_prepare_and_filter_demand_risk_data(sample_sales_df: pd.DataFrame) -> None:
    """Verify demand-risk segmentation wrapper and filtering."""
    summary = prepare_demand_risk_data(sample_sales_df, recent_history_days=5, use_cache=False)
    assert len(summary) == 4
    assert "mean_daily_sales" in summary.columns
    assert "std_daily_sales" in summary.columns
    assert "demand_risk_score" in summary.columns
    assert "risk_category" in summary.columns
    assert "review_priority" in summary.columns

    # Filter by store
    f_store = filter_demand_risk_data(summary, store=1)
    assert len(f_store) == 2
    assert set(f_store["store"]) == {1}

    # Filter by priority
    f_prio = filter_demand_risk_data(summary, review_priority="CRITICAL")
    assert all(f_prio["review_priority"] == "CRITICAL")


def test_get_sku_demand_and_variability(sample_sales_df: pd.DataFrame) -> None:
    """Verify SKU parameter extraction for replenishment planning."""
    summary = prepare_demand_risk_data(sample_sales_df, recent_history_days=5, use_cache=False)
    residuals_df = pd.DataFrame([
        {"store": 1, "item": 1, "val_residual_rmse": 3.45, "val_observation_count": 90}
    ])

    sku_meta = get_sku_demand_and_variability(summary, residuals_df, store=1, item=1)
    assert sku_meta["store"] == 1
    assert sku_meta["item"] == 1
    assert sku_meta["daily_demand_rate"] == 10.0
    assert sku_meta["forecast_residual_rmse"] == 3.45
    assert sku_meta["validation_sample_size"] == 90

    # Non-existent SKU error
    with pytest.raises(ValueError, match="not found in summary DataFrame"):
        get_sku_demand_and_variability(summary, residuals_df, store=99, item=99)


def test_evaluate_scenario_adapter_parity_with_core_inventory() -> None:
    """Verify exact calculation parity between evaluate_scenario_adapter and core evaluate_replenishment_scenario."""
    store = 1
    item = 1
    daily_rate = 50.0
    dispersion = 12.0
    lead_time = 7
    service_level = 0.95
    moq = 10
    review_period = 1
    on_hand = 100
    on_order = 50
    backorders = 10

    # 1. Via adapter
    rec_adapter = evaluate_scenario_adapter(
        store=store,
        item=item,
        daily_demand_rate=daily_rate,
        demand_variability=dispersion,
        lead_time_days=lead_time,
        service_level=service_level,
        min_order_quantity=moq,
        review_period_days=review_period,
        on_hand=on_hand,
        on_order=on_order,
        backorders=backorders,
    )

    # 2. Directly via core inventory module
    policy = ReplenishmentPolicy(
        lead_time_days=lead_time,
        service_level=service_level,
        min_order_quantity=moq,
        review_period_days=review_period,
    )
    current_inv = InventoryPosition(on_hand=on_hand, on_order=on_order, backorders=backorders)
    rec_core = evaluate_replenishment_scenario(
        store=store,
        item=item,
        daily_demand_rate=daily_rate,
        demand_variability=dispersion,
        policy=policy,
        current_inventory=current_inv,
    )

    # Assert exact numerical parity
    assert rec_adapter.lead_time_demand == rec_core.lead_time_demand
    assert rec_adapter.safety_stock == rec_core.safety_stock
    assert rec_adapter.reorder_point == rec_core.reorder_point
    assert rec_adapter.order_up_to_level == rec_core.order_up_to_level
    assert rec_adapter.current_inventory_position == rec_core.current_inventory_position
    assert rec_adapter.suggested_order_quantity == rec_core.suggested_order_quantity
    assert rec_adapter.reorder_required == rec_core.reorder_required
    assert rec_adapter.stock_status == rec_core.stock_status


def test_evaluate_scenario_adapter_input_validation() -> None:
    """Verify defensive validation on invalid planner inputs."""
    # Invalid lead time (< 1)
    with pytest.raises(InventoryPolicyError, match="lead_time_days must be"):
        evaluate_scenario_adapter(
            store=1, item=1, daily_demand_rate=20.0, demand_variability=5.0, lead_time_days=0
        )

    # Invalid service level (<= 0 or >= 1)
    with pytest.raises(InventoryPolicyError, match="service_level must be"):
        evaluate_scenario_adapter(
            store=1, item=1, daily_demand_rate=20.0, demand_variability=5.0, lead_time_days=7, service_level=1.0
        )

    # Invalid MOQ (< 1)
    with pytest.raises(InventoryPolicyError, match="min_order_quantity must be"):
        evaluate_scenario_adapter(
            store=1, item=1, daily_demand_rate=20.0, demand_variability=5.0, lead_time_days=7, min_order_quantity=0
        )

    # Negative stock quantities
    with pytest.raises(InventoryPolicyError, match="cannot be negative"):
        evaluate_scenario_adapter(
            store=1, item=1, daily_demand_rate=20.0, demand_variability=5.0, lead_time_days=7, on_hand=-5
        )


def test_format_replenishment_results() -> None:
    """Verify formatting payload structure for UI presentation."""
    rec = evaluate_scenario_adapter(
        store=1,
        item=1,
        daily_demand_rate=50.0,
        demand_variability=10.0,
        lead_time_days=5,
        service_level=0.95,
        on_hand=50,  # Below ROP -> reorder immediate
    )

    formatted = format_replenishment_results(rec)
    assert "headline_metrics" in formatted
    assert "action_recommendation" in formatted
    assert "parameter_summary" in formatted
    assert "assumptions" in formatted
    assert "missing_information" in formatted

    hl = formatted["headline_metrics"]
    assert hl["stock_status"] == "REORDER_IMMEDIATE"
    assert hl["status_color"] == "red"
    assert hl["reorder_required"] is True
    assert hl["suggested_order_quantity"] > 0
    assert len(formatted["parameter_summary"]) == 12


def test_build_demand_explorer_chart(sample_sales_df: pd.DataFrame) -> None:
    """Verify Plotly time series chart creation with actuals and moving averages."""
    # Non-empty chart
    fig = build_demand_explorer_chart(sample_sales_df, show_ma7=True)
    assert len(fig.data) >= 2  # actuals + 7-day MA

    # With validation predictions trace
    mock_val_preds = pd.DataFrame({
        "date": sample_sales_df["date"].head(5),
        "predicted_sales": [12.0, 15.0, 18.0, 22.0, 25.0],
    })
    fig_with_val = build_demand_explorer_chart(sample_sales_df, val_preds_df=mock_val_preds)
    assert len(fig_with_val.data) >= 3

    # Empty chart handling
    fig_empty = build_demand_explorer_chart(pd.DataFrame())
    assert len(fig_empty.data) == 0
    assert len(fig_empty.layout.annotations) > 0


def test_build_model_comparison_chart() -> None:
    """Verify model comparison bar chart creation."""
    df = load_model_comparison()
    fig = build_model_comparison_chart(df)
    assert len(fig.data) == 2  # MAE and RMSE bars

    # Empty dataframe handling
    fig_empty = build_model_comparison_chart(pd.DataFrame())
    assert len(fig_empty.data) == 0


def test_build_inventory_risk_scatter(sample_sales_df: pd.DataFrame) -> None:
    """Verify inventory risk scatter plot construction with median lines and selected marker."""
    summary = prepare_demand_risk_data(sample_sales_df, recent_history_days=5, use_cache=False)
    fig = build_inventory_risk_scatter(summary, highlight_store=1, highlight_item=1)
    assert len(fig.data) > 0

    # Empty handling with descriptive annotation
    fig_empty = build_inventory_risk_scatter(pd.DataFrame())
    assert len(fig_empty.data) == 0
    assert len(fig_empty.layout.annotations) > 0


# =============================================================================
# Milestone 4 Stage 3 Audit Regression Tests
# =============================================================================

def test_filter_demand_risk_store_2_and_quadrants() -> None:
    """Regression test for Step 2: Store 2 risk quadrant filtering bug.

    Ensures selecting Store 2 with canonical short name or full label returns
    all 33 matching rows, compose correctly with review priority, and handles
    empty results gracefully.
    """
    sales_df = load_dashboard_sales_data(use_cache=True)
    summary = prepare_demand_risk_data(sales_df, recent_history_days=90, cutoff_date="2017-07-01")

    # 1. Store 2 with All categories -> exactly 50 series
    res_all = filter_demand_risk_data(summary, store=2, risk_category="All")
    assert len(res_all) == 50
    assert (res_all["store"] == 2).all()

    # 2. Store 2 with canonical short category name -> exactly 33 rows
    res_short = filter_demand_risk_data(summary, store=2, risk_category="High Velocity - Volatile")
    assert len(res_short) == 33
    assert (res_short["store"] == 2).all()
    assert (res_short["review_priority"] == "CRITICAL").all()

    # 3. Store 2 with full category label containing parentheses -> exactly 33 rows
    res_full = filter_demand_risk_data(
        summary, store=2, risk_category="High Velocity - Volatile (Requires Inventory Review)"
    )
    assert len(res_full) == 33
    assert (res_full["store"] == 2).all()

    # 4. Store 2 with combined quadrant + review priority filters
    res_prio = filter_demand_risk_data(
        summary, store=2, risk_category="High Velocity - Volatile", review_priority="CRITICAL"
    )
    assert len(res_prio) == 33

    # 5. Store 2 with other categories
    res_stable_fast = filter_demand_risk_data(summary, store=2, risk_category="High Velocity - Stable")
    assert len(res_stable_fast) == 1

    res_stable_slow = filter_demand_risk_data(summary, store=2, risk_category="Low Velocity - Stable")
    assert len(res_stable_slow) == 16

    # 6. Store 2 with Lumpy/Erratic Demand (genuinely 0 items for Store 2 in this quadrant)
    res_lumpy = filter_demand_risk_data(summary, store=2, risk_category="Lumpy / Erratic Demand")
    assert len(res_lumpy) == 0
    assert isinstance(res_lumpy, pd.DataFrame)
    assert "store" in res_lumpy.columns

    # 7. Total items for Store 2 across all 4 quadrants reconciles to 50
    assert len(res_short) + len(res_stable_fast) + len(res_stable_slow) + len(res_lumpy) == 50


def test_risk_quadrant_counts_reconciliation() -> None:
    """Regression test for Step 3: Reconcile risk counts across overview, table, and guide.

    Verifies that the segmentation result across all 500 store-item series
    yields exact counts matching the authoritative Milestone 3 baseline
    (238 Critical, 12 Elevated, 238 Standard, 12 Low = 500 total).
    """
    sales_df = load_dashboard_sales_data(use_cache=True)
    summary = prepare_demand_risk_data(sales_df, recent_history_days=90, cutoff_date="2017-07-01")

    assert len(summary) == 500

    counts = summary["review_priority"].value_counts().to_dict()
    assert counts.get("CRITICAL", 0) == 238
    assert counts.get("ELEVATED", 0) == 12
    assert counts.get("STANDARD", 0) == 238
    assert counts.get("LOW", 0) == 12
    assert sum(counts.values()) == 500

    # Verify overview KPI aggregation returns exact matching counts
    kpis = compute_overview_kpis(sales_df, summary_df=summary)
    assert kpis["critical_risk_count"] == 238
    assert kpis["elevated_risk_count"] == 12
    assert kpis["standard_risk_count"] == 238
    assert kpis["low_risk_count"] == 12
    assert kpis["total_series"] == 500


def test_replenishment_audit_example_arithmetic() -> None:
    """Regression test for Step 4: Verify arithmetic for the displayed planner example.

    Inputs:
    - Daily demand: 20.31 units/day
    - Historical std: 5.86 units/day
    - Lead time: 7 days
    - Service level: 95% (Z = 1.6449)
    - Review period: 1 day
    - On-hand stock: 71 units
    - On-order: 0 units
    - Backorders: 0 units
    - MOQ: 10 units

    Expected:
    - Lead-time demand: ~142.17 units
    - Safety stock: ~25.5 units
    - Reorder point: ~167.7 units
    - Order-up-to level: ~188.0 units
    - Net inventory position: 71 units
    - Unrounded replenishment gap (Deficit): ~117.0 units
    - Suggested order: 120 units (batched to MOQ 10)
    """
    rec = evaluate_scenario_adapter(
        store=1,
        item=1,
        daily_demand_rate=20.31,
        demand_variability=5.86,
        lead_time_days=7,
        service_level=0.95,
        review_period_days=1,
        min_order_quantity=10,
        on_hand=71,
        on_order=0,
        backorders=0,
    )

    assert round(rec.lead_time_demand, 2) == 142.17
    assert round(rec.safety_stock, 1) == 25.5
    assert round(rec.reorder_point, 1) == 167.7
    assert round(rec.order_up_to_level, 1) == 188.0
    assert rec.current_inventory_position == 71
    assert rec.reorder_required is True
    assert rec.suggested_order_quantity == 120
    assert rec.stock_status == "REORDER_IMMEDIATE"

    # Verify formatted UI payload
    formatted = format_replenishment_results(rec)
    hl = formatted["headline_metrics"]
    assert hl["unrounded_deficit"] == 116.98
    assert "moq_explanation" in formatted
    assert "117.0" in formatted["moq_explanation"] or "116.9" in formatted["moq_explanation"]
    assert "120" in formatted["moq_explanation"]


def test_replenishment_on_order_backorders_surplus_and_zero_demand() -> None:
    """Verify on-order, backorders, surplus stock, zero demand, and MOQ batching variations."""
    # 1. On-order inventory prevents reorder when net IP > ROP
    rec_on_order = evaluate_scenario_adapter(
        store=1,
        item=1,
        daily_demand_rate=20.31,
        demand_variability=5.86,
        lead_time_days=7,
        service_level=0.95,
        on_hand=71,
        on_order=150,  # IP = 71 + 150 = 221 > ROP 167.7
        backorders=0,
        min_order_quantity=10,
    )
    assert rec_on_order.current_inventory_position == 221
    assert rec_on_order.reorder_required is False
    assert rec_on_order.suggested_order_quantity == 0
    assert rec_on_order.stock_status in ["ADEQUATE", "SURPLUS"]

    # 2. Backorders reduce net inventory position and increase order size
    rec_backorders = evaluate_scenario_adapter(
        store=1,
        item=1,
        daily_demand_rate=20.31,
        demand_variability=5.86,
        lead_time_days=7,
        service_level=0.95,
        on_hand=71,
        on_order=0,
        backorders=30,  # IP = 71 - 30 = 41
        min_order_quantity=10,
    )
    assert rec_backorders.current_inventory_position == 41
    # Deficit = 187.982 - 41 = 146.982 -> rounds to 150
    assert rec_backorders.suggested_order_quantity == 150

    # 3. Surplus stock detection (IP > 1.5 * OUTL)
    # OUTL ~ 188.0, 1.5 * 188.0 = 282.0
    rec_surplus = evaluate_scenario_adapter(
        store=1,
        item=1,
        daily_demand_rate=20.31,
        demand_variability=5.86,
        lead_time_days=7,
        service_level=0.95,
        on_hand=350,
        min_order_quantity=10,
    )
    assert rec_surplus.stock_status == "SURPLUS"
    assert rec_surplus.suggested_order_quantity == 0

    # 4. Zero demand rate edge case
    rec_zero = evaluate_scenario_adapter(
        store=1,
        item=1,
        daily_demand_rate=0.0,
        demand_variability=0.0,
        lead_time_days=7,
        service_level=0.95,
        on_hand=10,
    )
    assert rec_zero.lead_time_demand == 0.0
    assert rec_zero.safety_stock == 0.0
    assert rec_zero.reorder_point == 0.0
    assert rec_zero.suggested_order_quantity == 0

    # 5. MOQ batching variations for unrounded deficit ~116.98
    for moq, expected_order in [(10, 120), (25, 125), (50, 150), (100, 200)]:
        r = evaluate_scenario_adapter(
            store=1,
            item=1,
            daily_demand_rate=20.31,
            demand_variability=5.86,
            lead_time_days=7,
            service_level=0.95,
            min_order_quantity=moq,
            on_hand=71,
        )
        assert r.suggested_order_quantity == expected_order


