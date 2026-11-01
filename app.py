"""StockWise — AI Inventory & Demand Intelligence Platform.

Interactive Streamlit analytics dashboard (Milestone 4) providing multi-store
retail demand exploration, model benchmark evaluation, demand-risk quadrant
segmentation, and continuous-review replenishment scenario simulation.
"""

from __future__ import annotations

import datetime
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure src directory is in sys.path so app can be launched directly from any directory
_src_dir = str(Path(__file__).resolve().parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stockwise.dashboard_utils import (
    CANONICAL_RISK_QUADRANTS,
    build_demand_explorer_chart,
    build_inventory_risk_scatter,
    build_model_comparison_chart,
    compute_overview_kpis,
    evaluate_scenario_adapter,
    filter_demand_risk_data,
    filter_sales_data,
    format_replenishment_results,
    get_date_range,
    get_holdout_test_metrics,
    get_sku_demand_and_variability,
    get_store_and_item_options,
    load_dashboard_sales_data,
    load_model_comparison,
    load_saved_replenishment_scenarios,
    prepare_demand_risk_data,
    train_and_cache_validation_predictions,
)
from stockwise.inventory import InventoryPolicyError

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="StockWise — AI Inventory & Demand Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Cached Data Loaders
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading retail sales dataset (913k records)...")
def cached_sales_data() -> pd.DataFrame:
    """Load and cache the sales dataset."""
    return load_dashboard_sales_data()


@st.cache_data(show_spinner=False)
def cached_model_comparison() -> pd.DataFrame:
    """Load and cache model comparison metrics."""
    return load_model_comparison()


@st.cache_data(show_spinner="Computing demand velocity and risk segmentation...")
def cached_demand_risk(sales_df: pd.DataFrame) -> pd.DataFrame:
    """Compute and cache demand risk quadrant segmentation using pre-validation historical data."""
    return prepare_demand_risk_data(sales_df, recent_history_days=90, cutoff_date="2017-07-01")


@st.cache_data(show_spinner="Fitting Ridge model on train split to compute Q3 2017 validation forecasts...")
def cached_validation_forecasts(sales_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute and cache out-of-sample validation predictions and residual metrics."""
    return train_and_cache_validation_predictions(sales_df)


# -----------------------------------------------------------------------------
# Main Dashboard Application
# -----------------------------------------------------------------------------
def main() -> None:
    """Main dashboard rendering function."""
    # Header Banner
    st.title("📦 StockWise: AI Inventory & Demand Intelligence")
    st.markdown(
        "**Retail Demand Analytics, Forecast Benchmarking, and Replenishment Optimization**  \n"
        "*Empirical demand intelligence for multi-store retail supply chains.*"
    )

    # Global Scope & Limitation Callout
    st.info(
        "**Benchmark Scope Notice:** This application analyzes 5 years of historical retail point-of-sale "
        "transactions (500 store-item pairs). It records historical sales volume, **not** physical warehouse inventory "
        "balances, supplier delivery delay records, purchase orders, holding/shortage costs, or historical stockouts. "
        "Replenishment recommendations are parameterized scenario simulations.",
        icon="ℹ️",
    )

    # Data Ingestion with Graceful Error Handling
    try:
        sales_df = cached_sales_data()
    except FileNotFoundError as err:
        st.error(f"Failed to load dataset: {err}. Please ensure data/raw/train.csv exists.")
        st.stop()
    except Exception as err:
        st.error(f"Unexpected error loading sales data: {err}")
        st.stop()

    try:
        model_comp_df = cached_model_comparison()
    except Exception:
        model_comp_df = pd.DataFrame()

    try:
        summary_df = cached_demand_risk(sales_df)
    except Exception as err:
        st.warning(f"Could not compute demand risk summary: {err}")
        summary_df = pd.DataFrame()

    stores, items = get_store_and_item_options(sales_df)
    min_date, max_date = get_date_range(sales_df)

    # Sidebar Navigation & Context
    with st.sidebar:
        st.header("StockWise Intelligence")
        st.markdown(
            "An enterprise retail inventory decision platform built upon classical inventory control theory "
            "and chronological machine learning forecasting."
        )
        st.divider()

        st.subheader("Catalog Dimensions")
        st.markdown(
            f"- **Stores:** {len(stores):,} retail locations\n"
            f"- **SKUs / Items:** {len(items):,} products\n"
            f"- **Total Time Series:** {len(stores) * len(items):,} combinations\n"
            f"- **Historical Range:** {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}\n"
            f"- **Total Observations:** {len(sales_df):,} daily records"
        )
        st.divider()

        st.subheader("Methodology Pillars")
        st.markdown(
            "1. **Strict Chronological Splits:** Training, validation, and untouched test holdouts.\n"
            "2. **Dual Safety-Stock Framework:** Sizing buffers using historical demand variance or forecast error RMSE.\n"
            "3. **Continuous Review $(s, S)$ Policy:** Order-up-to level replenishment batched to MOQ.\n"
            "4. **Transparent Risk Heuristics:** Velocity/volatility segmentation (prioritization, not stockout probabilities)."
        )
        st.divider()
        st.caption("StockWise v1.0 • Milestone 4 Interactive Analytics")

    # Top-Level Tabs
    tab_overview, tab_explorer, tab_eval, tab_risk, tab_planner = st.tabs(
        [
            "📊 Executive Overview",
            "📈 Demand Explorer",
            "🎯 Forecast Evaluation",
            "⚠️ Inventory Risk",
            "📦 Replenishment Planner",
        ]
    )

    # =========================================================================
    # TAB 1: EXECUTIVE OVERVIEW
    # =========================================================================
    with tab_overview:
        st.header("Executive Catalog Overview")
        st.markdown("High-level catalog volume, store coverage, and empirical demand indicators.")

        kpis = compute_overview_kpis(sales_df, summary_df=summary_df)

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Recorded Sales", f"{kpis['total_sales']:,} units")
        col2.metric("Mean Store-Item Daily Sales", f"{kpis['avg_daily_sales']:.1f} units/day")
        col3.metric("Stores Covered", f"{kpis['total_stores']}")
        col4.metric("SKUs Covered", f"{kpis['total_items']}")
        col5.metric("Active Series", f"{kpis['total_series']}")

        st.divider()

        sub1, sub2, sub3, sub4 = st.columns(4)
        sub1.metric(
            "Critical Review SKUs",
            f"{kpis['critical_risk_count']}",
            help="Quadrant I: High velocity & high volatility items requiring active monitoring.",
        )
        sub2.metric(
            "Elevated Volume SKUs",
            f"{kpis['elevated_risk_count']}",
            help="Quadrant II: High velocity steady fast-movers (high sales volume).",
        )
        sub3.metric(
            "Standard Maintenance SKUs",
            f"{kpis['standard_risk_count']}",
            help="Quadrant III: Predictable low-maintenance baseline items.",
        )
        sub4.metric(
            "Erratic / Lumpy SKUs",
            f"{kpis['low_risk_count']}",
            help="Quadrant IV: Slow-moving items with intermittent demand spikes.",
        )

        st.divider()

        col_left, col_right = st.columns([1.1, 1.0])

        with col_left:
            st.subheader("Demand Risk Quadrants Distribution")
            if not summary_df.empty:
                quad_counts = summary_df["risk_category"].value_counts().reset_index()
                quad_counts.columns = ["Quadrant Category", "SKU Count"]
                quad_counts["Review Priority"] = quad_counts["Quadrant Category"].map({
                    "High Velocity - Volatile (Requires Inventory Review)": "CRITICAL",
                    "High Velocity - Stable (Steady Fast-Mover)": "ELEVATED",
                    "Low Velocity - Stable (Low Maintenance)": "STANDARD",
                    "Lumpy / Erratic Demand (Intermittent)": "LOW",
                })
                st.dataframe(quad_counts, width="stretch", hide_index=True)
            else:
                st.info("Demand risk summary table not available.")

            st.subheader("Selected Model Performance")
            holdout = get_holdout_test_metrics()
            ridge_res = holdout["ridge_regression"]
            base_res = holdout["seasonal_naive_baseline"]
            comp_table = pd.DataFrame([
                {
                    "Model": ridge_res["model_name"],
                    "Holdout Test MAE": f"{ridge_res['mae']:.2f} units",
                    "Holdout Test RMSE": f"{ridge_res['rmse']:.2f} units",
                    "Holdout Test R²": f"{ridge_res['r2']:.4f}",
                    "Status": "Validation-Selected Model",
                },
                {
                    "Model": base_res["model_name"],
                    "Holdout Test MAE": f"{base_res['mae']:.2f} units",
                    "Holdout Test RMSE": f"{base_res['rmse']:.2f} units",
                    "Holdout Test R²": f"{base_res['r2']:.4f}",
                    "Status": "Retail Baseline Benchmark",
                },
            ])
            st.dataframe(comp_table, width="stretch", hide_index=True)
            st.caption(f"Operational Benchmark: Ridge regression achieves **{holdout['mae_reduction_pct']}% lower test MAE vs baseline**.")

        with col_right:
            st.subheader("Operational Reality & Dataset Boundaries")
            limits = kpis["limitations"]
            with st.container(border=True):
                st.markdown("#### ✅ What this Dataset Establishes")
                for item_str in limits["what_dataset_establishes"]:
                    st.markdown(f"- {item_str}")

                st.markdown("#### ⚠️ What this Dataset CANNOT Establish")
                for item_str in limits["what_dataset_cannot_establish"]:
                    st.markdown(f"- {item_str}")

            st.caption(
                "Because physical operational balances are not present in transactional sales logs, "
                "the Replenishment Planner accepts hypothetical user inputs to evaluate policy outcomes."
            )

    # =========================================================================
    # TAB 2: DEMAND EXPLORER
    # =========================================================================
    with tab_explorer:
        st.header("Demand Explorer")
        st.markdown("Filter, visualize, and analyze historical daily demand patterns and out-of-sample predictions.")

        fc1, fc2, fc3, fc4 = st.columns([1, 1, 1.5, 1.2])
        with fc1:
            selected_store = st.selectbox("Store Identifier", options=stores, index=0, key="exp_store")
        with fc2:
            selected_item = st.selectbox("Item / SKU Identifier", options=items, index=0, key="exp_item")
        with fc3:
            date_filter = st.date_input(
                "Date Range Filter",
                value=(min_date.date(), max_date.date()),
                min_value=min_date.date(),
                max_value=max_date.date(),
                key="exp_date_range",
            )
        with fc4:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            show_val_overlay = st.checkbox(
                "Overlay Q3 2017 Validation Forecast",
                value=True,
                help="Overlay out-of-sample Ridge predictions evaluated on the historical Q3 2017 validation split.",
            )

        # Resolve date boundaries safely
        if isinstance(date_filter, (tuple, list)) and len(date_filter) == 2:
            start_d, end_d = date_filter[0], date_filter[1]
        elif isinstance(date_filter, (tuple, list)) and len(date_filter) == 1:
            start_d, end_d = date_filter[0], max_date.date()
        else:
            start_d, end_d = min_date.date(), max_date.date()

        filtered_series = filter_sales_data(
            sales_df,
            store=selected_store,
            item=selected_item,
            start_date=start_d,
            end_date=end_d,
        )

        val_preds_for_sku: Optional[pd.DataFrame] = None
        if show_val_overlay:
            try:
                val_df_all, _ = cached_validation_forecasts(sales_df)
                val_preds_for_sku = val_df_all[
                    (val_df_all["store"] == selected_store) & (val_df_all["item"] == selected_item)
                ]
            except Exception as err:
                st.warning(f"Could not load validation forecasts: {err}")

        # Summary Metrics for Selected Series
        if not filtered_series.empty:
            m1, m2, m3, m4 = st.columns(4)
            tot = int(filtered_series["sales"].sum())
            mean_d = float(filtered_series["sales"].mean())
            std_d = float(filtered_series["sales"].std())
            cv_d = (std_d / mean_d) if mean_d > 0 else 0.0

            m1.metric("Selected Range Sales", f"{tot:,} units")
            m2.metric("Mean Daily Sales", f"{mean_d:.2f} units/day")
            m3.metric("Daily Volatility (Std)", f"{std_d:.2f} units/day")
            m4.metric("Coeff. of Variation (CV)", f"{cv_d:.2f}")

        # Plotly Time Series Chart
        chart = build_demand_explorer_chart(
            filtered_df=filtered_series,
            val_preds_df=val_preds_for_sku,
            show_ma7=True,
        )
        st.plotly_chart(chart, width="stretch")

        st.caption(
            "**Chart Legend & Methodology Note:** Solid blue line indicates historical point-of-sale observations. "
            "Dashed red line displays out-of-sample Ridge regression forecasts evaluated strictly on the historical Q3 2017 validation split (not a live future forecast). "
            "Historical actuals and validation predictions are strictly segregated."
        )

    # =========================================================================
    # TAB 3: FORECAST EVALUATION
    # =========================================================================
    with tab_eval:
        st.header("Forecast Engine Evaluation & Methodology")
        st.markdown(
            "Evaluation of candidate models using strict chronological partitioning to eliminate data leakage."
        )

        # Methodology Description Card
        with st.container(border=True):
            st.subheader("Chronological Partitioning Protocol")
            st.markdown(
                "- **Training Split (2013-01-29 to 2017-06-30 • 807,000 records):** Historical window used to train models and fit transformers.\n"
                "- **Validation Split (2017-07-01 to 2017-09-30 • 46,000 records • Q3 2017):** Out-of-sample tuning and model comparison window.\n"
                "- **Final Test Split (2017-10-01 to 2017-12-31 • 46,000 records • Q4 2017):** Untouched holdout set used exclusively for verified reporting.\n"
                "- **Strict Data Leakage Prevention:** Rolling averages are lagged by 1 day so target sales never enter feature calculation."
            )

        if not model_comp_df.empty:
            st.subheader("Validation Set Model Comparison (Q3 2017)")
            st.dataframe(model_comp_df, width="stretch", hide_index=True)

            bar_fig = build_model_comparison_chart(model_comp_df)
            st.plotly_chart(bar_fig, width="stretch")
        else:
            st.info("Validation model comparison table not found in reports/model_comparison.csv.")

        st.divider()

        # Accessible Metric Guide
        st.subheader("Retail Metric Interpretation Guide")
        g1, g2, g3 = st.columns(3)
        with g1:
            with st.container(border=True):
                st.markdown("#### Mean Absolute Error (MAE)")
                st.markdown(
                    "Measures the average magnitude of daily forecasting errors in actual sales units. "
                    "An MAE of **7.13 units** means daily predictions deviate by approximately 7 units on average."
                )
        with g2:
            with st.container(border=True):
                st.markdown("#### Root Mean Squared Error (RMSE)")
                st.markdown(
                    "Penalizes large outlier forecast errors more heavily by squaring residuals before averaging. "
                    "RMSE is the mathematically appropriate metric for sizing safety stock buffers."
                )
        with g3:
            with st.container(border=True):
                st.markdown("#### Coefficient of Determination ($R^2$)")
                st.markdown(
                    "Quantifies the fraction of total sales variance captured by model features (variance explained, "
                    "not point accuracy). An $R^2$ of **0.927** indicates the model captures 92.7% of demand fluctuations around the mean."
                )

        st.divider()

        # Final Holdout Benchmark Card
        st.subheader("Fixed Final Holdout Test Set Benchmark (Q4 2017)")
        st.warning(
            "**Policy Integrity Guarantee:** The final holdout test set (October 1 to December 31, 2017) "
            "is preserved strictly as an un-tuned performance benchmark. It is **never** used to tune model hyperparameters, "
            "select replenishment thresholds, or simulate replenishment decisions.",
            icon="🔒",
        )
        holdout_info = get_holdout_test_metrics()
        ht1, ht2, ht3, ht4 = st.columns(4)
        ht1.metric("Selected Model Test MAE", f"{holdout_info['ridge_regression']['mae']:.2f} units")
        ht2.metric("Selected Model Test RMSE", f"{holdout_info['ridge_regression']['rmse']:.2f} units")
        ht3.metric("Selected Model Test R²", f"{holdout_info['ridge_regression']['r2']:.4f}")
        ht4.metric("Test Error Reduction", f"{holdout_info['mae_reduction_pct']}% lower MAE vs Baseline")

    # =========================================================================
    # TAB 4: INVENTORY RISK
    # =========================================================================
    with tab_risk:
        st.header("Inventory Demand Risk Matrix")
        st.markdown(
            "Segment all 500 store-item combinations by empirical demand velocity and volatility "
            "to allocate supervisory attention."
        )

        st.info(
            "**Heuristic Clarification:** Quadrant classifications and demand risk scores are transparent prioritization "
            "heuristics ranking items by volume and uncertainty. They are **not** calibrated probabilities of stockout, "
            "as historical stockout events are not observed in this dataset.",
            icon="ℹ️",
        )

        if not summary_df.empty:
            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                risk_store_filter = st.selectbox(
                    "Filter by Store",
                    options=["All"] + stores,
                    index=0,
                    key="risk_store",
                )
            with rc2:
                canonical_categories = ["All"] + list(CANONICAL_RISK_QUADRANTS.keys())
                risk_cat_filter = st.selectbox(
                    "Filter by Risk Quadrant",
                    options=canonical_categories,
                    index=0,
                    key="risk_cat",
                )
            with rc3:
                risk_prio_filter = st.selectbox(
                    "Filter by Review Priority",
                    options=["All", "CRITICAL", "ELEVATED", "STANDARD", "LOW"],
                    index=0,
                    key="risk_prio",
                )

            filtered_risk = filter_demand_risk_data(
                summary_df,
                store=None if risk_store_filter == "All" else int(risk_store_filter),
                risk_category=None if risk_cat_filter == "All" else str(risk_cat_filter),
                review_priority=None if risk_prio_filter == "All" else str(risk_prio_filter),
            )

            # Scatter Plot using filtered items with catalog reference medians
            if not filtered_risk.empty:
                scatter_fig = build_inventory_risk_scatter(
                    summary_df=filtered_risk,
                    reference_summary_df=summary_df,
                    highlight_store=None if risk_store_filter == "All" else int(risk_store_filter),
                )
                st.plotly_chart(scatter_fig, width="stretch")
            else:
                st.info("No store-item series match the selected filter criteria.")

            # Quadrant Reference Guide with dynamic counts reconciled with summary_df
            q1_count = int((summary_df["review_priority"] == "CRITICAL").sum())
            q2_count = int((summary_df["review_priority"] == "ELEVATED").sum())
            q3_count = int((summary_df["review_priority"] == "STANDARD").sum())
            q4_count = int((summary_df["review_priority"] == "LOW").sum())

            with st.expander("Quadrant Categorization Guide & Operational Definitions", expanded=False):
                st.markdown(
                    f"1. **Quadrant I: High Velocity - Volatile (Critical Priority):** {q1_count} high-volume series with high demand dispersion. Primary focus for inventory review.\n"
                    f"2. **Quadrant II: High Velocity - Stable (Elevated Priority):** {q2_count} fast-moving steady products with high sales volume and low safety buffer requirements.\n"
                    f"3. **Quadrant III: Low Velocity - Stable (Standard Priority):** {q3_count} predictable baseline products requiring standard replenishment review.\n"
                    f"4. **Quadrant IV: Lumpy / Erratic Demand (Low Volume Priority):** {q4_count} slow-moving items with intermittent volume spikes. Require larger batch sizes."
                )

            if not filtered_risk.empty:
                st.subheader(f"Segmented Series Table ({len(filtered_risk)} matching SKUs)")
                display_cols = [
                    "store",
                    "item",
                    "mean_daily_sales",
                    "std_daily_sales",
                    "cv_daily_sales",
                    "demand_risk_score",
                    "risk_category",
                    "review_priority",
                ]
                st.dataframe(
                    filtered_risk[display_cols],
                    width="stretch",
                    hide_index=True,
                )
        else:
            st.info("Demand risk data is not currently available.")

    # =========================================================================
    # TAB 5: REPLENISHMENT SCENARIO PLANNER
    # =========================================================================
    with tab_planner:
        st.header("Replenishment Scenario Planner")
        st.markdown(
            "Continuous-review $(s, S)$ replenishment policy evaluation with dual safety-stock sizing and full decision explainability."
        )

        st.warning(
            "**Hypothetical Input Notice:** The sales dataset contains historical sales, not current stock levels. "
            "Please provide your operational assumptions below to simulate replenishment orders.",
            icon="⚠️",
        )

        # SKU Selector
        ps1, ps2, ps3 = st.columns([1, 1, 2])
        with ps1:
            plan_store = st.selectbox("Store Location", options=stores, index=0, key="plan_store")
        with ps2:
            plan_item = st.selectbox("Target Item / SKU", options=items, index=0, key="plan_item")

        # Load SKU Empirical Metadata
        try:
            _, sku_residuals_df = cached_validation_forecasts(sales_df)
        except Exception:
            sku_residuals_df = None

        try:
            sku_meta = get_sku_demand_and_variability(
                summary_df=summary_df,
                sku_residuals_df=sku_residuals_df,
                store=plan_store,
                item=plan_item,
            )
        except Exception as err:
            st.error(f"Failed to lookup SKU demand parameters: {err}")
            st.stop()

        with ps3:
            st.markdown(
                f"**SKU Profile:** Store {plan_store} • Item {plan_item}  \n"
                f"- **Empirical Demand Velocity:** `{sku_meta['daily_demand_rate']:.2f}` units/day  \n"
                f"- **Historical Sales Volatility (σ_D):** `{sku_meta['historical_demand_std']:.2f}` units/day  \n"
                f"- **Validation Forecast Error (RMSE):** `{sku_meta['forecast_residual_rmse']:.2f}` units/day  \n"
                f"- **Risk Classification:** `{sku_meta['risk_category'].split(' (')[0]}` ({sku_meta['review_priority']})"
            )

        st.divider()

        # Simulation Input Panels
        col_policy, col_inv = st.columns(2)

        with col_policy:
            st.subheader("1. Policy & Supplier Constraints")
            lead_time = st.slider("Supplier Lead Time (days)", min_value=1, max_value=60, value=7, step=1)
            service_level_pct = st.slider(
                "Cycle Service Level (CSL %)",
                min_value=80.0,
                max_value=99.0,
                value=95.0,
                step=0.5,
                help="Target cycle service level fraction (theoretical target under normal error assumptions; not an empirical stockout prevention guarantee).",
            )
            service_level = service_level_pct / 100.0

            lt_variability = st.slider(
                "Lead Time Std. Deviation (days)",
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=0.5,
                help="0.0 represents deterministic supplier lead time; > 0 represents stochastic delivery variance.",
            )

            moq = st.number_input("Minimum Order Quantity (MOQ)", min_value=1, max_value=5000, value=10, step=5)
            review_period = st.number_input("Review Period (days)", min_value=1, max_value=30, value=1, step=1)

            variability_choice = st.radio(
                "Safety Stock Buffer Basis",
                options=["Historical Sales Variance (Default)", "Forecast Residual Error (RMSE)"],
                index=0,
                help="Historical variance uses empirical daily sales std; forecast residual error sizes buffers using out-of-sample ML error dispersion.",
            )
            variability_basis_key = (
                "forecast_residual_error"
                if "Forecast" in variability_choice
                else "historical_demand_variance"
            )

        with col_inv:
            st.subheader("2. Hypothetical Operational Inventory")
            # Default hypothetical stock to 50% of expected lead time demand
            default_on_hand = int(round(sku_meta["daily_demand_rate"] * lead_time * 0.5))

            on_hand = st.number_input(
                "Hypothetical Physical On-Hand Stock (units)",
                min_value=0,
                max_value=100000,
                value=default_on_hand,
                step=10,
                help="Physical inventory assumed to be on store shelves or backroom for scenario planning.",
            )
            on_order = st.number_input(
                "Hypothetical Stock On-Order with Supplier (units)",
                min_value=0,
                max_value=100000,
                value=0,
                step=10,
                help="Units assumed to be in transit from supplier.",
            )
            backorders = st.number_input(
                "Hypothetical Customer Backorders (units)",
                min_value=0,
                max_value=100000,
                value=0,
                step=5,
                help="Hypothetical customer demand awaiting fulfillment.",
            )

            dispersion_used = (
                sku_meta["forecast_residual_rmse"]
                if variability_basis_key == "forecast_residual_error"
                else sku_meta["historical_demand_std"]
            )

            st.caption(
                f"Selected variability metric: **{dispersion_used:.2f} units/day** "
                f"({variability_basis_key.replace('_', ' ').title()})"
            )

        # Run Scenario Evaluation
        try:
            recommendation = evaluate_scenario_adapter(
                store=plan_store,
                item=plan_item,
                daily_demand_rate=sku_meta["daily_demand_rate"],
                demand_variability=dispersion_used,
                lead_time_days=lead_time,
                service_level=service_level,
                lead_time_std_days=lt_variability,
                min_order_quantity=moq,
                review_period_days=review_period,
                on_hand=on_hand,
                on_order=on_order,
                backorders=backorders,
                variability_basis=variability_basis_key,
                residual_sample_size=sku_meta["validation_sample_size"],
            )
            results = format_replenishment_results(recommendation)
        except InventoryPolicyError as err:
            st.error(f"Policy parameter validation error: {err}")
            st.stop()
        except Exception as err:
            st.error(f"Unexpected simulation error: {err}")
            st.stop()

        st.divider()

        # Display Results
        st.subheader("Replenishment Decision & Parameters")

        hl = results["headline_metrics"]
        status = hl["stock_status"]

        if status == "REORDER_IMMEDIATE":
            st.error(f"🚨 **ORDER ACTION REQUIRED:** {results['action_recommendation']}", icon="🚨")
        elif status == "ADEQUATE":
            st.success(f"✅ **STOCK ADEQUATE:** {results['action_recommendation']}", icon="✅")
        else:
            st.info(f"ℹ️ **SURPLUS INVENTORY:** {results['action_recommendation']}", icon="📦")

        # Metric Cards
        rc1, rc2, rc3, rc4, rc5 = st.columns(5)
        rc1.metric(
            "Net Inventory Position",
            f"{hl['net_inventory_position']:,} units",
            help="On-Hand + On-Order - Backorders",
        )
        rc2.metric(
            "Reorder Point (ROP)",
            f"{hl['reorder_point']} units",
            help="Lead Time Demand + Safety Stock Buffer",
        )
        rc3.metric(
            "Safety Stock Buffer",
            f"{hl['safety_stock']} units",
            help=f"Protective buffer for nominal {service_level * 100:.1f}% CSL target",
        )
        rc4.metric(
            "Order-Up-To Level (OUTL)",
            f"{hl['order_up_to_level']} units",
            help="Target inventory level covering lead time and review cycle",
        )
        rc5.metric(
            "Suggested Order Qty",
            f"{hl['suggested_order_quantity']:,} units",
            help=f"Recommended purchase batch rounded to MOQ {moq}",
        )

        st.info(f"📋 **Replenishment Sizing Analysis:** {results['moq_explanation']}")

        st.divider()

        # Audit Trail and Explainability
        col_audit, col_assump = st.columns(2)

        with col_audit:
            st.markdown("#### Scenario Audit Trail")
            param_df = pd.DataFrame(results["parameter_summary"])
            st.dataframe(param_df, width="stretch", hide_index=True)

        with col_assump:
            st.markdown("#### Policy Assumptions & Missing Information")
            with st.expander("Underlying Inventory Model Assumptions", expanded=True):
                for a in results["assumptions"]:
                    st.markdown(f"- {a}")

            with st.expander("Unobserved Real-World Operational Factors", expanded=False):
                for m in results["missing_information"]:
                    st.markdown(f"- {m}")

        st.caption(
            "Scenario Evaluation Notice: All calculations reflect a parameterized simulation based on your hypothetical inputs. "
            "This does not represent an actual purchase order or an empirically guaranteed stockout-prevention action."
        )


# -----------------------------------------------------------------------------
# Application Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
