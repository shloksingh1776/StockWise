"""Unit tests for the stockwise forecasting module."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from stockwise.forecasting import (
    CATEGORICAL_FEATURES,
    ForecastingError,
    NUMERICAL_FEATURES,
    SeasonalNaiveForecaster,
    build_forecasting_pipeline,
    calculate_metrics,
    create_forecasting_features,
    split_chronological_data,
)
from sklearn.linear_model import LinearRegression, Ridge


@pytest.fixture
def synthetic_panel_df() -> pd.DataFrame:
    """Generate a clean synthetic panel dataset for testing.

    2 stores, 2 items, 400 consecutive days = 1,600 total observations.
    Sales follow a known deterministic formula:
    sales = store * 10 + item * 5 + day_index.
    """
    records = []
    dates = pd.date_range("2016-01-01", periods=400, freq="D")
    for store in [1, 2]:
        for item in [1, 2]:
            for day_idx, dt in enumerate(dates):
                records.append({
                    "date": dt,
                    "store": store,
                    "item": item,
                    "sales": float(store * 10 + item * 5 + day_idx),
                })
    df = pd.DataFrame(records)
    # Shuffle order to ensure functions properly sort by (store, item, date)
    return df.sample(frac=1.0, random_state=42).reset_index(drop=True)


def test_lag_features_correctness(synthetic_panel_df: pd.DataFrame):
    """Test that lag features (1, 7, 14, 28) correctly reference past sales for each entity."""
    df_feat = create_forecasting_features(
        synthetic_panel_df,
        lag_days=[1, 7, 14, 28],
        rolling_windows=[7, 28],
        drop_na=False,
    )

    # Filter to store 1, item 1
    s1_i1 = df_feat[(df_feat["store"] == 1) & (df_feat["item"] == 1)].sort_values("date").reset_index(drop=True)

    # For day index 30 (date = 2017-01-31), sales was 10 + 5 + 30 = 45.0
    day_30 = s1_i1.iloc[30]
    assert day_30["sales"] == 45.0
    assert day_30["lag_1"] == s1_i1.iloc[29]["sales"]
    assert day_30["lag_7"] == s1_i1.iloc[23]["sales"]
    assert day_30["lag_14"] == s1_i1.iloc[16]["sales"]
    assert day_30["lag_28"] == s1_i1.iloc[2]["sales"]


def test_rolling_features_no_target_leakage(synthetic_panel_df: pd.DataFrame):
    """Test that rolling features are shifted by 1 so the target day's sales never enters its rolling average."""
    df_feat = create_forecasting_features(
        synthetic_panel_df,
        lag_days=[1, 7],
        rolling_windows=[7, 28],
        drop_na=False,
    )

    s1_i1 = df_feat[(df_feat["store"] == 1) & (df_feat["item"] == 1)].sort_values("date").reset_index(drop=True)

    # At day index 7 (the 8th day):
    # Past 7 days are indices 0, 1, 2, 3, 4, 5, 6.
    expected_rolling_7 = s1_i1.iloc[0:7]["sales"].mean()
    actual_rolling_7 = s1_i1.iloc[7]["rolling_mean_7"]
    assert np.isclose(actual_rolling_7, expected_rolling_7)

    # Strict Leakage Proof:
    # If we alter day 7's sales in an identical dataset, rolling_mean_7 at day 7 MUST NOT CHANGE
    df_modified = synthetic_panel_df.copy()
    mask = (df_modified["store"] == 1) & (df_modified["item"] == 1) & (df_modified["date"] == s1_i1.iloc[7]["date"])
    df_modified.loc[mask, "sales"] = 999999.0

    df_feat_modified = create_forecasting_features(
        df_modified,
        lag_days=[1, 7],
        rolling_windows=[7, 28],
        drop_na=False,
    )
    s1_i1_mod = df_feat_modified[(df_feat_modified["store"] == 1) & (df_feat_modified["item"] == 1)].sort_values("date").reset_index(drop=True)

    assert s1_i1_mod.iloc[7]["rolling_mean_7"] == actual_rolling_7
    # But day 8's rolling average should reflect the change
    assert s1_i1_mod.iloc[8]["rolling_mean_7"] != s1_i1.iloc[8]["rolling_mean_7"]


def test_missing_history_handling(synthetic_panel_df: pd.DataFrame):
    """Test explicit handling of initial rows lacking sufficient history."""
    # When drop_na=False, initial rows have NaN for lag_28 and rolling_mean_28
    df_retained = create_forecasting_features(synthetic_panel_df, drop_na=False)
    assert df_retained["lag_28"].isna().sum() == 2 * 2 * 28  # 4 series * 28 initial days = 112 NaNs

    # When drop_na=True, these incomplete initial rows are explicitly removed
    df_dropped = create_forecasting_features(synthetic_panel_df, drop_na=True)
    assert df_dropped["lag_28"].isna().sum() == 0
    assert len(df_dropped) == len(synthetic_panel_df) - (2 * 2 * 28)
    assert len(df_dropped) == 4 * (400 - 28)  # 1,488 rows


def test_chronological_split_boundaries():
    """Test date-based chronological splitting boundaries and validation."""
    dates = pd.date_range("2017-01-01", "2017-12-31", freq="D")
    df = pd.DataFrame({
        "date": dates,
        "store": 1,
        "item": 1,
        "sales": np.arange(len(dates), dtype=float),
    })

    train_df, val_df, test_df = split_chronological_data(
        df,
        train_end="2017-07-01",
        val_start="2017-07-01",
        val_end="2017-09-30",
        test_start="2017-10-01",
        test_end="2017-12-31",
    )

    # Verify strict date bounds
    assert train_df["date"].max() == pd.Timestamp("2017-06-30")
    assert val_df["date"].min() == pd.Timestamp("2017-07-01")
    assert val_df["date"].max() == pd.Timestamp("2017-09-30")
    assert test_df["date"].min() == pd.Timestamp("2017-10-01")
    assert test_df["date"].max() == pd.Timestamp("2017-12-31")

    # Verify no date overlap across partitions
    train_dates = set(train_df["date"])
    val_dates = set(val_df["date"])
    test_dates = set(test_df["date"])
    assert train_dates.isdisjoint(val_dates)
    assert val_dates.isdisjoint(test_dates)
    assert train_dates.isdisjoint(test_dates)


def test_split_chronological_data_invalid_boundaries():
    """Test that invalid chronological order raises ForecastingError."""
    df = pd.DataFrame({"date": pd.date_range("2017-01-01", "2017-02-01"), "sales": 1})
    with pytest.raises(ForecastingError, match="Invalid chronological split order"):
        split_chronological_data(df, train_end="2017-05-01", val_start="2017-03-01", val_end="2017-04-01")


def test_seasonal_naive_baseline(synthetic_panel_df: pd.DataFrame):
    """Test Weekly Seasonal Naive forecaster correctly predicts using lag_7."""
    df_feat = create_forecasting_features(synthetic_panel_df, drop_na=True)

    baseline = SeasonalNaiveForecaster(lag_feature_name="lag_7")
    baseline.fit(df_feat)
    predictions = baseline.predict(df_feat)

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(df_feat)
    np.testing.assert_array_equal(predictions, df_feat["lag_7"].to_numpy())


def test_seasonal_naive_missing_lag_feature():
    """Test error when required lag feature is absent."""
    df_bad = pd.DataFrame({"sales": [10, 20]})
    baseline = SeasonalNaiveForecaster(lag_feature_name="lag_7")
    with pytest.raises(ForecastingError, match="missing from input columns"):
        baseline.fit(df_bad)


def test_calculate_metrics():
    """Test accuracy of MAE, RMSE, R-squared, and negative predictions calculation."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])

    metrics = calculate_metrics(y_true, y_pred)
    # Errors: [-2, 2, -3] -> Abs errors: [2, 2, 3] -> MAE = 7/3 = 2.3333
    assert np.isclose(metrics["MAE"], 7.0 / 3.0, atol=1e-3)
    # Sq errors: [4, 4, 9] -> Mean = 17/3 -> RMSE = sqrt(17/3) = 2.3805
    assert np.isclose(metrics["RMSE"], np.sqrt(17.0 / 3.0), atol=1e-3)
    # Actual mean = 20, SS_tot = (10-20)^2 + 0 + (30-20)^2 = 200. SS_res = 17 -> R2 = 1 - 17/200 = 0.915
    assert np.isclose(metrics["R2"], 0.915, atol=1e-3)
    assert metrics["negative_count"] == 0
    assert metrics["min_pred"] == 12.0
    assert metrics["max_pred"] == 33.0

    # Test negative prediction detection
    y_pred_neg = np.array([-5.0, 10.0, 25.0])
    metrics_neg = calculate_metrics(y_true, y_pred_neg)
    assert metrics_neg["negative_count"] == 1
    assert metrics_neg["min_pred"] == -5.0


def test_feature_generation_reproducibility(synthetic_panel_df: pd.DataFrame):
    """Test that feature generation produces identical results across runs."""
    df_1 = create_forecasting_features(synthetic_panel_df, drop_na=True)
    df_2 = create_forecasting_features(synthetic_panel_df, drop_na=True)

    pd.testing.assert_frame_equal(df_1, df_2)


def test_pipeline_fit_predict_synthetic(synthetic_panel_df: pd.DataFrame):
    """Test scikit-learn Pipeline construction, fitting, and inference on synthetic data."""
    df_feat = create_forecasting_features(synthetic_panel_df, drop_na=True)

    # Split synthetic data chronologically (train on full year 2016, validate on 2017)
    train_df = df_feat[df_feat["date"] < "2017-01-01"]
    val_df = df_feat[df_feat["date"] >= "2017-01-01"]

    feature_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    pipeline = build_forecasting_pipeline(LinearRegression())

    pipeline.fit(train_df[feature_cols], train_df["sales"])
    preds = pipeline.predict(val_df[feature_cols])

    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(val_df)
    assert not np.isnan(preds).any()


def test_pipeline_unknown_category_handling(synthetic_panel_df: pd.DataFrame):
    """Test that pipeline handles unseen categorical values gracefully via handle_unknown='ignore'."""
    df_feat = create_forecasting_features(synthetic_panel_df, drop_na=True)
    train_df = df_feat[df_feat["date"] < "2017-01-01"]

    # Introduce intentionally unseen categories in validation set
    val_unseen = df_feat[df_feat["date"] >= "2017-01-01"].copy()
    val_unseen["store"] = 999  # Unseen store category
    val_unseen["item"] = 999   # Unseen item category

    feature_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    pipeline = build_forecasting_pipeline(LinearRegression())
    pipeline.fit(train_df[feature_cols], train_df["sales"])

    # Expected behavior: OneHotEncoder emits UserWarning but encodes as zeros without crashing
    with pytest.warns(UserWarning, match="Found unknown categories"):
        preds = pipeline.predict(val_unseen[feature_cols])

    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(val_unseen)
    assert not np.isnan(preds).any()
