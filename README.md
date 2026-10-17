# StockWise — AI Inventory and Demand Intelligence

StockWise is an AI-driven inventory optimization and demand intelligence platform designed to forecast multi-store retail demand, detect seasonal trends, and guide stock replenishment decisions.

This repository tracks the project's milestones, starting with the core data foundation and exploratory data analysis.

---

## Milestone 1: Data Foundation and Exploratory Analysis

### Objectives & Deliverables
* **Data Ingestion & Strict Validation Module (`src/stockwise/data_loader.py`):**
  * Safe CSV loading using Pandas without altering source data.
  * Validation of required schema columns: `date`, `store`, `item`, `sales`.
  * Date parsing and malformed date detection (raising `DataValidationError`).
  * Detection of missing values, duplicate rows, duplicate observations on composite keys `(date, store, item)`, non-numeric values, and negative sales quantities.
  * Calculation of summary statistics: dimensions, date coverage, store counts, item counts, and sales metrics (mean, median, standard deviation, min, max, zero-sales count).
  * Separation of critical validation failures (exceptions) from data quality observations (warnings).
* **Automated Unit Tests (`tests/test_data_loader.py`):**
  * 12 isolated unit tests covering schema conformity, date handling, missing data detection, duplicate detection, negative sales alerts, and statistical calculation accuracy using synthetic in-memory fixtures.
* **Exploratory Data Analysis Notebook (`notebooks/01_exploratory_data_analysis.ipynb`):**
  * Executed analysis of the full 913,000-row retail sales dataset.
  * Visualizations covering macro demand trends (30-day moving average), annual seasonality curves, store volume rankings, top 10 product lines, and day-of-the-week demand cycles.

---

## Project Structure

```
stockwise/
├── data/
│   ├── raw/                  # Raw dataset directory (.gitkeep, train.csv)
│   └── processed/            # Processed data directory (.gitkeep)
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb  # Executed EDA notebook
├── reports/                  # Generated figures and reports
├── src/
│   └── stockwise/
│       ├── __init__.py       # Package entry point exposing core loader functions
│       └── data_loader.py    # Data ingestion, schema validation & quality metrics
├── tests/
│   └── test_data_loader.py   # Pytest suite with synthetic fixtures
├── .gitignore                # Protects .venv, local data, and secrets
├── pytest.ini                # Pytest configuration (pythonpath = src)
├── requirements.txt          # Pinned project dependencies
└── README.md                 # Project documentation
```

---

## Environment & Setup

### 1. Python Virtual Environment
StockWise uses Python 3.13 in a dedicated virtual environment at `.venv`.

**Activate on Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Activate on macOS/Linux (Bash):**
```bash
source .venv/bin/activate
```

### 2. Dependencies
All required dependencies (`pandas`, `numpy`, `matplotlib`, `seaborn`, `pytest`, `jupyter`, `ipykernel`, etc.) are recorded in `requirements.txt`.
To install or restore dependencies into an active environment:
```powershell
pip install -r requirements.txt
```

---

## Dataset Placement

The raw retail sales dataset must be placed at:
```
data/raw/train.csv
```

### Expected Schema
| Column | Type | Description |
| :--- | :--- | :--- |
| `date` | Date (`YYYY-MM-DD`) | Date of transaction |
| `store` | Integer (`1` – `10`) | Store identifier |
| `item` | Integer (`1` – `50`) | Product / SKU identifier |
| `sales` | Integer | Units sold on that day |

*Note: The raw dataset is tracked locally and excluded from Git commits via `.gitignore`.*

---

## Running Automated Tests

Run the test suite using `pytest`:

```powershell
# Run all tests
pytest

# Run tests with verbose output
pytest -v
```

All 12 unit tests use synthetic fixtures and temporary directories, executing in under 1 second without re-reading the entire 913,000-row dataset.

---

## Running the EDA Notebook

Launch Jupyter to inspect or interactively rerun the exploratory analysis notebook:

```powershell
jupyter lab
# or
jupyter notebook
```

Navigate to `notebooks/01_exploratory_data_analysis.ipynb`.

The notebook can also be re-executed programmatically via the command line:
```powershell
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_exploratory_data_analysis.ipynb
```

---

## Key Empirical Findings (Milestone 1 Summary)

1. **Dataset Completeness:**
   * 913,000 observations forming a balanced rectangular panel: exactly 10 stores $\times$ 50 items $\times$ 1,826 consecutive calendar days (January 1, 2013 to December 31, 2017).
   * 0 missing values, 0 duplicate observations, and 0 negative sales records. Exactly 1 recorded zero-sales event (`2014-01-15`, Store 6, Item 4).
2. **Temporal Trends & Seasonality:**
   * Consistent year-over-year compounding sales growth from 2013 through 2017.
   * Prominent annual inverted-U seasonal curve peaking in July and bottoming out in January/February.
3. **Store-Level Disparity:**
   * Tier 1 stores (Store 2: 6.12M units; Store 8: 5.86M units) vs. Tier 3 stores (Store 7: 3.32M units). Store 2 moves 1.84× the volume of Store 7.
4. **Product Distribution:**
   * Top-selling products: Item 15 (1.61M units) and Item 28 (1.60M units). The top 10 items account for 31.6% of all units sold.
5. **Intra-Week Cycles:**
   * Monotonic growth from Monday trough (~41.4 units/item/day) to Sunday peak (~62.1 units/item/day), a ~50.0% weekend demand acceleration.
