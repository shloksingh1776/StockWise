"""StockWise - AI Inventory and Demand Intelligence package.

Milestone 1: Data Foundation and Exploratory Analysis.
Milestone 2: Demand Forecasting Engine.
Milestone 3: Inventory Intelligence.
"""

from stockwise.data_loader import (
    DataValidationError,
    generate_data_quality_report,
    load_and_validate_data,
    load_raw_data,
    validate_sales_data,
)
from stockwise.forecasting import (
    CATEGORICAL_FEATURES,
    DEFAULT_TEST_END,
    DEFAULT_TEST_START,
    DEFAULT_TRAIN_END,
    DEFAULT_VAL_END,
    DEFAULT_VAL_START,
    LAG_DAYS,
    NUMERICAL_FEATURES,
    ROLLING_WINDOWS,
    ForecastingError,
    SeasonalNaiveForecaster,
    build_forecasting_pipeline,
    calculate_metrics,
    create_forecasting_features,
    evaluate_models_on_splits,
    get_model_specifications,
    plot_actual_vs_predicted,
    plot_model_comparison,
    plot_residuals_analysis,
    split_chronological_data,
)
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

__all__ = [
    # Milestone 1: Data Loader
    "DataValidationError",
    "load_raw_data",
    "validate_sales_data",
    "generate_data_quality_report",
    "load_and_validate_data",
    # Milestone 2: Forecasting Engine
    "ForecastingError",
    "SeasonalNaiveForecaster",
    "create_forecasting_features",
    "split_chronological_data",
    "build_forecasting_pipeline",
    "calculate_metrics",
    "get_model_specifications",
    "evaluate_models_on_splits",
    "plot_model_comparison",
    "plot_actual_vs_predicted",
    "plot_residuals_analysis",
    "LAG_DAYS",
    "ROLLING_WINDOWS",
    "CATEGORICAL_FEATURES",
    "NUMERICAL_FEATURES",
    "DEFAULT_TRAIN_END",
    "DEFAULT_VAL_START",
    "DEFAULT_VAL_END",
    "DEFAULT_TEST_START",
    "DEFAULT_TEST_END",
    # Milestone 3: Inventory Intelligence
    "InventoryPolicyError",
    "ReplenishmentPolicy",
    "InventoryPosition",
    "ReplenishmentRecommendation",
    "calculate_lead_time_demand",
    "calculate_safety_stock",
    "calculate_reorder_point",
    "evaluate_replenishment_scenario",
    "summarize_demand_and_risk",
    "calculate_sku_residual_variability",
]
