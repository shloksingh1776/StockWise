"""Unit tests for the stockwise data_loader module."""

import pytest
import pandas as pd
from pathlib import Path

from stockwise.data_loader import (
    DataValidationError,
    load_raw_data,
    validate_sales_data,
    generate_data_quality_report,
    load_and_validate_data,
)


@pytest.fixture
def valid_sample_csv(tmp_path: Path) -> Path:
    """Create a small valid synthetic CSV file for testing."""
    file_path = tmp_path / "valid_sample.csv"
    content = (
        "date,store,item,sales\n"
        "2023-01-01,1,1,10\n"
        "2023-01-01,1,2,20\n"
        "2023-01-02,1,1,15\n"
        "2023-01-02,2,1,25\n"
        "2023-01-03,2,2,30\n"
    )
    file_path.write_text(content, encoding="utf-8")
    return file_path


def test_load_raw_data_success(valid_sample_csv: Path):
    """Test successful loading and date parsing of a valid CSV."""
    df = load_raw_data(valid_sample_csv)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert list(df.columns) == ["date", "store", "item", "sales"]
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["date"].iloc[0] == pd.Timestamp("2023-01-01")


def test_load_raw_data_file_not_found():
    """Test FileNotFoundError is raised when target file is missing."""
    with pytest.raises(FileNotFoundError):
        load_raw_data("non_existent_file_path.csv")


def test_load_raw_data_empty_file(tmp_path: Path):
    """Test DataValidationError when CSV file is completely empty."""
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("", encoding="utf-8")

    with pytest.raises(DataValidationError, match="empty|Failed to read"):
        load_raw_data(empty_csv)


def test_load_raw_data_missing_required_columns(tmp_path: Path):
    """Test DataValidationError when one or more required columns are missing."""
    bad_csv = tmp_path / "missing_columns.csv"
    bad_csv.write_text("date,store,item\n2023-01-01,1,1\n", encoding="utf-8")

    with pytest.raises(DataValidationError, match="Missing required column.*sales"):
        load_raw_data(bad_csv)


def test_load_raw_data_invalid_date_format(tmp_path: Path):
    """Test DataValidationError when date column contains unparseable values."""
    bad_dates_csv = tmp_path / "bad_dates.csv"
    bad_dates_csv.write_text(
        "date,store,item,sales\n"
        "2023-01-01,1,1,10\n"
        "not-a-valid-date,1,2,15\n",
        encoding="utf-8",
    )

    with pytest.raises(DataValidationError, match="unparseable date values"):
        load_raw_data(bad_dates_csv)


def test_validate_sales_data_clean_dataset(valid_sample_csv: Path):
    """Test validation summary on clean data without warnings."""
    df = load_raw_data(valid_sample_csv)
    summary = validate_sales_data(df)

    assert summary["dimensions"]["total_rows"] == 5
    assert summary["dimensions"]["total_columns"] == 4
    assert summary["data_quality"]["total_missing_values"] == 0
    assert summary["data_quality"]["exact_duplicates"] == 0
    assert summary["data_quality"]["observation_duplicates"] == 0
    assert summary["data_quality"]["negative_sales"] == 0
    assert summary["data_quality"]["warnings_count"] == 0
    assert len(summary["warnings"]) == 0


def test_validate_sales_data_missing_values():
    """Test detection and warning generation for missing values."""
    df = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
        "store": [1, None, 2],
        "item": [1, 2, None],
        "sales": [10.0, 20.0, 30.0],
    })

    summary = validate_sales_data(df)
    assert summary["data_quality"]["total_missing_values"] == 2
    assert summary["data_quality"]["missing_values"]["store"] == 1
    assert summary["data_quality"]["missing_values"]["item"] == 1
    assert any("Missing values detected" in w for w in summary["warnings"])


def test_validate_sales_data_duplicates():
    """Test detection of exact row duplicates and composite key duplicates."""
    df = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-01"]),
        "store": [1, 1, 1],
        "item": [1, 1, 1],
        "sales": [10, 10, 20],  # row 0 and 1 are exact duplicates; row 2 shares same key
    })

    summary = validate_sales_data(df)
    assert summary["data_quality"]["exact_duplicates"] == 1
    assert summary["data_quality"]["observation_duplicates"] == 2
    assert any("duplicate row" in w for w in summary["warnings"])
    assert any("duplicate observation" in w for w in summary["warnings"])


def test_validate_sales_data_negative_and_invalid_numeric_sales():
    """Test detection of negative sales and non-numeric entries in sales column."""
    df = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]),
        "store": [1, 1, 1, 1],
        "item": [1, 1, 1, 1],
        "sales": [15, -5, -2, "invalid_num"],
    })

    summary = validate_sales_data(df)
    assert summary["data_quality"]["negative_sales"] == 2
    assert summary["data_quality"]["invalid_numeric_sales"] == 1
    assert any("negative value" in w for w in summary["warnings"])
    assert any("non-numeric value" in w for w in summary["warnings"])


def test_validate_sales_data_statistics_accuracy():
    """Test that summary statistics are dynamically and accurately calculated."""
    df = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]),
        "store": [1, 1, 2, 2, 3],
        "item": [10, 20, 10, 30, 40],
        "sales": [10, 20, 30, 40, 50],
    })

    summary = validate_sales_data(df)

    # Date range checks
    assert summary["date_range"]["min_date"] == "2023-01-01"
    assert summary["date_range"]["max_date"] == "2023-01-05"
    assert summary["date_range"]["total_days"] == 5

    # Entity counts
    assert summary["entities"]["unique_stores"] == 3
    assert summary["entities"]["unique_items"] == 4

    # Sales statistics
    assert summary["sales_statistics"]["total_sales"] == 150.0
    assert summary["sales_statistics"]["mean_sales"] == 30.0
    assert summary["sales_statistics"]["median_sales"] == 30.0
    assert summary["sales_statistics"]["min_sales"] == 10.0
    assert summary["sales_statistics"]["max_sales"] == 50.0
    assert summary["sales_statistics"]["zero_sales_count"] == 0


def test_generate_data_quality_report(valid_sample_csv: Path):
    """Test formatting of summary into readable text report."""
    df = load_raw_data(valid_sample_csv)
    summary = validate_sales_data(df)
    report = generate_data_quality_report(summary)

    assert "STOCKWISE DATA QUALITY & VALIDATION REPORT" in report
    assert "Rows:               5" in report
    assert "Unique Stores:      2" in report
    assert "Unique Items:       2" in report
    assert "STATUS: PASSED" in report


def test_load_and_validate_pipeline(valid_sample_csv: Path):
    """Test the combined load_and_validate_data pipeline function."""
    df, summary = load_and_validate_data(valid_sample_csv)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert summary["dimensions"]["total_rows"] == 5
    assert summary["data_quality"]["warnings_count"] == 0
