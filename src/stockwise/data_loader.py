"""Data loading and validation module for StockWise.

Provides functions to load retail sales CSV data, perform strict schema
validation, and calculate comprehensive data-quality metrics and warnings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import pandas as pd


REQUIRED_COLUMNS: List[str] = ["date", "store", "item", "sales"]


class DataValidationError(ValueError):
    """Exception raised when dataset schema or fundamental validation fails."""

    pass


def load_raw_data(
    file_path: Union[str, Path],
    date_column: str = "date",
) -> pd.DataFrame:
    """Load retail sales data from a CSV file and perform schema and date validation.

    Parameters
    ----------
    file_path : str or Path
        Path to the target CSV file.
    date_column : str, default 'date'
        Name of the date column to parse and validate.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame with parsed datetime values in `date_column`. The source
        data is not altered otherwise.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    DataValidationError
        If required columns are missing, the file is empty, or dates cannot be parsed.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found at: {path.resolve()}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise DataValidationError(f"Failed to read CSV file '{path.name}': {exc}") from exc

    if df.empty:
        raise DataValidationError(f"CSV file '{path.name}' is empty (0 rows).")

    # Check for required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise DataValidationError(
            f"Dataset schema validation failed. Missing required column(s): {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    # Validate and parse date column
    # We parse with errors='coerce' to detect unparseable dates
    parsed_dates = pd.to_datetime(df[date_column], errors="coerce")
    invalid_date_mask = parsed_dates.isna() & df[date_column].notna()

    if invalid_date_mask.any():
        sample_invalid = df.loc[invalid_date_mask, date_column].head(3).tolist()
        raise DataValidationError(
            f"Date validation failed in column '{date_column}'. Found unparseable date values: "
            f"{sample_invalid}"
        )

    # Return a copy with parsed dates without modifying caller data in-place
    df_clean = df.copy()
    df_clean[date_column] = parsed_dates
    return df_clean


def validate_sales_data(
    df: pd.DataFrame,
    date_column: str = "date",
    store_column: str = "store",
    item_column: str = "item",
    sales_column: str = "sales",
) -> Dict[str, Any]:
    """Perform data-quality checks and return summary metrics and warnings.

    This function does NOT modify the input DataFrame. It distinguishes between
    critical schema failures and data quality observations (e.g. missing values,
    duplicates, or negative sales) that are reported as warnings.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to inspect.
    date_column : str
        Name of the date column.
    store_column : str
        Name of the store identifier column.
    item_column : str
        Name of the item/product identifier column.
    sales_column : str
        Name of the sales quantity column.

    Returns
    -------
    dict
        Structured dictionary containing dimensions, date range, entity counts,
        sales statistics, data quality checks, and warning messages.

    Raises
    ------
    DataValidationError
        If required columns are missing from the DataFrame.
    """
    missing_cols = [
        col for col in [date_column, store_column, item_column, sales_column]
        if col not in df.columns
    ]
    if missing_cols:
        raise DataValidationError(
            f"Cannot validate data: missing required column(s): {missing_cols}"
        )

    warnings: List[str] = []

    # 1. Dataset Dimensions
    total_rows = int(len(df))
    total_columns = int(len(df.columns))

    # 2. Missing Values Check
    missing_counts = df.isnull().sum().to_dict()
    total_missing = sum(missing_counts.values())
    if total_missing > 0:
        cols_with_missing = {k: v for k, v in missing_counts.items() if v > 0}
        warnings.append(
            f"Missing values detected in {len(cols_with_missing)} column(s): {cols_with_missing}"
        )

    # 3. Duplicate Observations Check
    exact_duplicates_count = int(df.duplicated().sum())
    if exact_duplicates_count > 0:
        warnings.append(
            f"Found {exact_duplicates_count} fully duplicate row(s) across all columns."
        )

    composite_key = [date_column, store_column, item_column]
    observation_duplicates_count = int(df.duplicated(subset=composite_key).sum())
    if observation_duplicates_count > 0:
        warnings.append(
            f"Found {observation_duplicates_count} duplicate observation(s) on composite key {composite_key}."
        )

    # 4. Numeric and Sales Validation
    sales_series = pd.to_numeric(df[sales_column], errors="coerce")
    invalid_numeric_sales_count = int(sales_series.isna().sum() - df[sales_column].isna().sum())
    if invalid_numeric_sales_count > 0:
        warnings.append(
            f"Found {invalid_numeric_sales_count} non-numeric value(s) in '{sales_column}' column."
        )

    negative_sales_mask = sales_series < 0
    negative_sales_count = int(negative_sales_mask.sum())
    if negative_sales_count > 0:
        warnings.append(
            f"Found {negative_sales_count} negative value(s) in '{sales_column}' column."
        )

    zero_sales_count = int((sales_series == 0).sum())

    # 5. Date Coverage
    dates_converted = pd.to_datetime(df[date_column], errors="coerce")
    valid_dates = dates_converted.dropna()
    if not valid_dates.empty:
        min_date = valid_dates.min().strftime("%Y-%m-%d")
        max_date = valid_dates.max().strftime("%Y-%m-%d")
        total_days = int((valid_dates.max() - valid_dates.min()).days + 1)
    else:
        min_date = None
        max_date = None
        total_days = 0
        warnings.append("No valid date values found in the dataset.")

    # 6. Entity Counts
    unique_stores = int(df[store_column].nunique(dropna=True))
    unique_items = int(df[item_column].nunique(dropna=True))

    # 7. Sales Statistics (calculated dynamically on valid numeric values)
    valid_sales = sales_series.dropna()
    if not valid_sales.empty:
        total_sales = float(valid_sales.sum())
        mean_sales = float(valid_sales.mean())
        median_sales = float(valid_sales.median())
        std_sales = float(valid_sales.std())
        min_sales = float(valid_sales.min())
        max_sales = float(valid_sales.max())
    else:
        total_sales = 0.0
        mean_sales = 0.0
        median_sales = 0.0
        std_sales = 0.0
        min_sales = 0.0
        max_sales = 0.0

    return {
        "dimensions": {
            "total_rows": total_rows,
            "total_columns": total_columns,
        },
        "date_range": {
            "min_date": min_date,
            "max_date": max_date,
            "total_days": total_days,
        },
        "entities": {
            "unique_stores": unique_stores,
            "unique_items": unique_items,
        },
        "sales_statistics": {
            "total_sales": total_sales,
            "mean_sales": round(mean_sales, 4),
            "median_sales": round(median_sales, 4),
            "std_sales": round(std_sales, 4),
            "min_sales": min_sales,
            "max_sales": max_sales,
            "zero_sales_count": zero_sales_count,
        },
        "data_quality": {
            "missing_values": missing_counts,
            "total_missing_values": total_missing,
            "exact_duplicates": exact_duplicates_count,
            "observation_duplicates": observation_duplicates_count,
            "negative_sales": negative_sales_count,
            "invalid_numeric_sales": invalid_numeric_sales_count,
            "warnings_count": len(warnings),
        },
        "warnings": warnings,
    }


def generate_data_quality_report(summary: Dict[str, Any]) -> str:
    """Format the validation summary dictionary into a human-readable text report.

    Parameters
    ----------
    summary : dict
        Summary dictionary returned by `validate_sales_data`.

    Returns
    -------
    str
        Formatted text report.
    """
    dims = summary["dimensions"]
    dates = summary["date_range"]
    entities = summary["entities"]
    sales = summary["sales_statistics"]
    quality = summary["data_quality"]
    warnings = summary["warnings"]

    lines = [
        "=" * 60,
        "          STOCKWISE DATA QUALITY & VALIDATION REPORT",
        "=" * 60,
        f"Rows:               {dims['total_rows']:,}",
        f"Columns:            {dims['total_columns']}",
        "-" * 60,
        "DATE COVERAGE",
        f"Start Date:         {dates['min_date']}",
        f"End Date:           {dates['max_date']}",
        f"Total Days:         {dates['total_days']:,}",
        "-" * 60,
        "ENTITIES",
        f"Unique Stores:      {entities['unique_stores']}",
        f"Unique Items:       {entities['unique_items']}",
        "-" * 60,
        "SALES METRICS",
        f"Total Units Sold:   {sales['total_sales']:,.0f}",
        f"Mean Daily Sales:   {sales['mean_sales']:,.2f}",
        f"Median Sales:       {sales['median_sales']:,.2f}",
        f"Std Deviation:      {sales['std_sales']:,.2f}",
        f"Min Sales:          {sales['min_sales']}",
        f"Max Sales:          {sales['max_sales']}",
        f"Zero-Sales Count:   {sales['zero_sales_count']:,}",
        "-" * 60,
        "DATA QUALITY & INTEGRITY",
        f"Missing Values:     {quality['total_missing_values']:,}",
        f"Exact Duplicates:   {quality['exact_duplicates']:,}",
        f"Key Duplicates:     {quality['observation_duplicates']:,}",
        f"Negative Sales:     {quality['negative_sales']:,}",
        f"Invalid Numerics:   {quality['invalid_numeric_sales']:,}",
    ]

    if warnings:
        lines.append("-" * 60)
        lines.append(f"WARNINGS ({len(warnings)} detected):")
        for idx, warn in enumerate(warnings, 1):
            lines.append(f"  [{idx}] {warn}")
    else:
        lines.append("-" * 60)
        lines.append("STATUS: PASSED (No data quality warnings)")

    lines.append("=" * 60)
    return "\n".join(lines)


def load_and_validate_data(
    file_path: Union[str, Path]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Convenience pipeline to load CSV data and run validation in one step.

    Parameters
    ----------
    file_path : str or Path
        Path to the target CSV file.

    Returns
    -------
    tuple of (pd.DataFrame, dict)
        The loaded DataFrame with parsed dates and the quality summary dict.
    """
    df = load_raw_data(file_path)
    summary = validate_sales_data(df)
    return df, summary
