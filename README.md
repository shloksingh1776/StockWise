# StockWise — AI Inventory and Demand Intelligence

StockWise is an AI-driven inventory optimization and demand intelligence platform designed to forecast multi-store retail demand, detect seasonal trends, and guide stock replenishment decisions.

This repository tracks the project's milestones:
* **Milestone 1:** Data Foundation and Exploratory Analysis (Completed)
* **Milestone 2:** Demand Forecasting Engine (Completed)
* **Milestone 3:** Inventory Intelligence and Replenishment Optimization (Completed)

---

## Milestone 3: Inventory Intelligence

### Objectives & Deliverables
* **Reusable Inventory Intelligence Module (`src/stockwise/inventory.py`):**
  * **Demand Velocity & Volatility Summaries:** Aggregates empirical daily mean sales ($\bar{d}$), daily sales standard deviation ($\sigma_D$), and coefficient of variation ($CV$) by store-item pair.
  * **Demand-Based Risk Segmentation:** Ranks SKUs using an empirical percentile score ($0.5 \cdot p_{vel} + 0.5 \cdot p_{vol}$) into four operational quadrants:
    1. *Quadrant I (High Velocity - Volatile):* 238 SKUs with high volume and high volatility designated as **CRITICAL** priority.
    2. *Quadrant II (High Velocity - Stable):* 12 steady fast-movers with **ELEVATED** revenue impact and low safety buffer requirements.
    3. *Quadrant III (Low Velocity - Stable):* 238 predictable baseline items with **STANDARD** maintenance.
    4. *Quadrant IV (Lumpy / Erratic Demand):* 12 slow-movers with intermittent demand spikes.
    *(Note: Demand risk scores are transparent prioritization heuristics, strictly not calibrated stockout probabilities).*
  * **Dual Safety-Stock Framework:**
    * *Historical Sales Standard Deviation (Default):* $SS_{hist} = Z \cdot \sigma_D \cdot \sqrt{L}$
    * *Forecast Error Variability (RMSE):* $SS_{model} = Z \cdot \text{RMSE}_{s,i} \cdot \sqrt{L}$ (computed on out-of-sample validation residuals, yielding 15%–39% buffer reductions under theoretical normal-error assumptions).
    * *Stochastic Lead Time Support:* $SS = Z \cdot \sqrt{L \sigma_D^2 + \bar{d}^2 \sigma_L^2}$ when supplier delivery variance $\sigma_L > 0$.
  * **Replenishment Policy Engine:** Continuous-review $(s, S)$ order calculator evaluating net inventory position ($IP = \text{On-Hand} + \text{On-Order} - \text{Backorders}$), Reorder Point ($ROP = d_L + SS$), Order-Up-To Level ($OUTL$), and Minimum Order Quantity ($MOQ$) batch rounding.
  * **Structured Decision Explainability:** Returns an audit trail with inputs used, output meaning, mathematical formulas, underlying assumptions, missing operational data, and recommended manager actions.
* **Automated Unit Tests (`tests/test_inventory.py`):**
  * 21 focused unit tests (44 total tests across the project) verifying lead-time demand calculations, defensive negative prediction clipping, deterministic/stochastic safety stock formulas, boundary rejections, MOQ rounding, quadrant assignments, rank ties, SKU residual metrics, and negative net inventory position handling.
* **Pipeline Execution & Reporting Engine (`src/stockwise/run_inventory.py`):**
  * Evaluates demand segmentation across all 500 store-item series, calculates out-of-sample validation error metrics, evaluates 72 policy scenarios across representative SKUs, and exports artifacts to `reports/`.

---

## Milestone 2: Demand Forecasting Engine

### Objectives & Deliverables
* **Reusable Forecasting Module (`src/stockwise/forecasting.py`):**
  * **Feature Engineering:** Calendar attributes (`dayofweek`, `month`, `year`), entity indicators (`store`, `item`), lagged sales (1, 7, 14, and 28 days), and shifted rolling averages (7 and 28 days).
  * **Strict Leakage Prevention:** Rolling averages are shifted by 1 day so that target day sales never enter feature calculation for that day.
  * **Initial History Management:** Explicitly handles initial rows lacking 28-day history (dropping exactly 14,000 initial observations across the 500 store-item pairs from January 1–28, 2013).
  * **Chronological Splits:** Validates date order and partitions data strictly by date:
    * **Training Targets:** 2013-01-29 to 2017-06-30 (807,000 records).
    * **Validation Targets:** 2017-07-01 to 2017-09-30 (46,000 records, Q3 2017).
    * **Final Test Targets:** 2017-10-01 to 2017-12-31 (46,000 records, Q4 2017).
  * **Scikit-Learn Architecture:** Reusable `Pipeline` and `ColumnTransformer` with `OneHotEncoder(drop='first')` for categorical identifiers and `StandardScaler` fitted strictly on training targets.
  * **Candidate Models Evaluated:**
    1. *Weekly Seasonal Naive:* Strong retail baseline using 7-day lagged sales ($\hat{y}_t = y_{t-7}$).
    2. *Linear Regression (OLS):* Unregularized ordinary least squares.
    3. *Ridge Regression (L2):* Regularized regression ($\alpha = 1.0$) mitigating multicollinearity.
    4. *Lasso Regression (L1):* Sparse regression ($\alpha = 0.1$) enforcing feature sparsity.
    5. *Elastic Net Regression:* Joint L1/L2 penalty ($\alpha = 0.1, l_1 = 0.5$).
* **Automated Unit Tests (`tests/test_forecasting.py`):**
  * 11 focused unit tests verifying lag correctness, target leakage prevention via synthetic mutations, chronological boundary non-overlap, initial history drops, baseline mechanics, and metric formula accuracy.
* **Pipeline Execution & Reporting Engine (`src/stockwise/run_forecasting.py`):**
  * CLI runner executing the end-to-end pipeline, selecting the top model on validation data, evaluating on untouched test data, and generating figures and reports in `reports/`.

---

## Model Evaluation & Empirical Findings (Milestone 2)

### Validation Set Comparison (2017-07-01 to 2017-09-30)

| Model | Type | Validation MAE | Validation RMSE | Validation $R^2$ | Fit Time (s) | Inference Time (s) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **Ridge Regression** | ML (L2) | **7.1288** | **9.2449** | **0.9272** | 2.19s | 0.06s |
| **Linear Regression** | ML (OLS) | 7.1295 | 9.2450 | 0.9272 | 2.49s | 0.06s |
| **Lasso Regression** | ML (L1) | 7.3785 | 9.5815 | 0.9218 | 8.74s | 0.06s |
| **Elastic Net Regression** | ML (Joint) | 7.4817 | 9.7139 | 0.9197 | 7.73s | 0.07s |
| **Weekly Seasonal Naive** | Baseline | 9.3519 | 12.1765 | 0.8738 | 0.00s | 0.00s |

### Untouched Test Set Evaluation (2017-10-01 to 2017-12-31)

Ridge Regression was selected based on validation performance and evaluated on the final test split using the model trained on the training split through 2017-06-30 (fixed-model holdout protocol, preserving the exact selected parameters without retraining on validation data):

| Metric | Weekly Seasonal Naive | Ridge Regression (Selected) | Operational Improvement |
| :--- | :--- | :--- | :--- |
| **Test MAE** | 9.0763 units | **6.7529 units** | **25.6% error reduction** |
| **Test RMSE** | 12.0507 units | **8.7568 units** | **27.3% error reduction** |
| **Test $R^2$** | 0.8210 | **0.9055** | **+0.0845 explained variance** |
| **Negative Preds** | 0 | **35 (0.076%)** | Min prediction: -4.30 units |

### Key Technical Insights:
1. **Machine Learning vs. Seasonal Baseline:** Ridge regression slashes validation MAE by 23.8% and test MAE by 25.6% relative to the weekly seasonal-naive heuristic.
2. **Ridge vs. OLS:** While OLS and Ridge perform similarly on uncorrupted validation data, Ridge ($\alpha = 1.0$) prevents coefficient inflation caused by collinearity between `lag_7`, `lag_14`, and 7-day rolling averages.
3. **Negative Demand Predictions:**
   * Validation set (summer peak): **0** negative predictions across all models.
   * Test set (winter trough): **35 out of 46,000 predictions (0.076%)** yielded slightly negative values (down to -4.30 units) due to unconstrained linear extrapolation on low-volume SKUs.
   * *Production Safeguard:* In `src/stockwise/inventory.py`, a defensive floor (`np.maximum(0.0, preds)`) is applied before replenishment sizing to ensure negative orders can never occur.
4. **Residual Diagnostics & Heteroscedasticity:**
   * Residual errors display clear **heteroscedasticity**: error standard deviation scales from 6.10 units in the lowest fitted quartile to 12.37 units in the highest quartile.
   * Sizing safety stock using SKU-specific dispersion ($\text{RMSE}_{s,i}$) accurately reflects volume-dependent error variance rather than applying a single uniform buffer.

---

## Milestone 1: Data Foundation and Exploratory Analysis

### Objectives & Deliverables
* **Data Ingestion & Strict Validation Module (`src/stockwise/data_loader.py`):**
  * Safe CSV loading using Pandas without altering source data.
  * Validation of required schema columns: `date`, `store`, `item`, `sales`.
  * Date parsing and malformed date detection (raising `DataValidationError`).
  * Detection of missing values, duplicate rows, duplicate observations on composite keys `(date, store, item)`, non-numeric values, and negative sales quantities.
  * Calculation of summary statistics: dimensions, date coverage, store counts, item counts, and sales metrics (mean, median, standard deviation, min, max, zero-sales count).
* **Automated Unit Tests (`tests/test_data_loader.py`):** 12 isolated unit tests covering schema conformity, date handling, missing data detection, duplicate detection, and statistical calculation accuracy.
* **Exploratory Data Analysis Notebook (`notebooks/01_exploratory_data_analysis.ipynb`):** Executed analysis of the full 913,000-row retail sales dataset.

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
│   ├── actual_vs_predicted.png              # Time-series & scatter plot diagnostics
│   ├── demand_risk_matrix.png               # Demand velocity vs. volatility scatter
│   ├── inventory_replenishment_scenarios.csv# Evaluated replenishment scenarios
│   ├── milestone_2_report.md                # Milestone 2 forecasting report
│   ├── milestone_3_inventory_report.md      # Milestone 3 inventory intelligence report
│   ├── model_comparison.csv                 # Tabular validation comparison metrics
│   ├── model_comparison.png                 # Comparative MAE and RMSE bar charts
│   └── residuals_analysis.png               # Error distribution & residual diagnostics
├── src/
│   └── stockwise/
│       ├── __init__.py            # Package entry point exposing loader, forecaster & inventory
│       ├── data_loader.py         # Data ingestion, schema validation & quality metrics
│       ├── forecasting.py         # Feature engineering, chronological splits, pipelines
│       ├── inventory.py           # Inventory policies, safety stock, ROP, risk scoring
│       ├── run_forecasting.py     # End-to-end forecasting pipeline runner
│       └── run_inventory.py       # End-to-end inventory scenario runner
├── tests/
│   ├── test_data_loader.py   # Pytest suite for data loader (12 tests)
│   ├── test_forecasting.py   # Pytest suite for forecasting engine (11 tests)
│   └── test_inventory.py     # Pytest suite for inventory intelligence (21 tests)
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
All required dependencies (`pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`, `pytest`, `jupyter`, etc.) are recorded in `requirements.txt`.
To install dependencies into an active environment:
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

---

## Running Automated Tests

Run the complete 44-test suite using `pytest`:

```powershell
# Run all tests
pytest

# Run tests with verbose output
pytest -v
```

All 44 unit tests run in ~6–7 seconds using synthetic in-memory fixtures.

---

## Executing Pipelines

### 1. Demand Forecasting Pipeline (Milestone 2)
```powershell
# Directly from project root (Windows)
.\.venv\Scripts\python.exe src/stockwise/run_forecasting.py

# Or on macOS / Linux
python src/stockwise/run_forecasting.py
```

### 2. Inventory Intelligence Pipeline (Milestone 3)
```powershell
# Directly from project root (Windows)
.\.venv\Scripts\python.exe src/stockwise/run_inventory.py

# Or on macOS / Linux
python src/stockwise/run_inventory.py
```
This generates `reports/demand_risk_matrix.png`, `reports/inventory_replenishment_scenarios.csv`, and `reports/milestone_3_inventory_report.md` in ~10 seconds.

---

## Methodological Integrity & Dataset Limitations
1. **Historical Sales Limitation:** The dataset contains transaction sales, not inventory records. Current stock balances, lead times, purchase orders, and holding costs used in scenario analyses are explicitly parameterized user inputs.
2. **1-Day-Ahead Assumption:** The pipeline models 1-day-ahead rolling forecasts with daily refreshed observations. For lead times $L > 1$, lead-time demand is approximated assuming constant daily demand rate ($L \times \max(0, \hat{y}_{t+1})$). Multi-step direct and recursive horizons will be explored in future milestones.
3. **Preservation of Untouched Test Set:** Model-driven safety stock metrics ($\text{RMSE}_{s,i}$) are calculated exclusively on out-of-sample validation residuals (Q3 2017). The Q4 2017 test set was preserved untouched as a pure holdout evaluation.
4. **Theoretical Normal-Error Assumptions & Nominal Service Levels:** Safety-stock reductions (15%–39%) are calculated under theoretical standard normal error assumptions. Nominal cycle service levels (e.g., 95%) have not been empirically validated through discrete-event inventory simulation. Because residuals display heteroscedasticity and heavy tails (+0.97 excess kurtosis), realized cycle service levels may differ from nominal targets. No specific realized service level is claimed.
