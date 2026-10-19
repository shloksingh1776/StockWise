# StockWise — Milestone 2: Demand Forecasting Engine Report

## 1. Executive Summary

Milestone 2 establishes a machine-learning demand forecasting pipeline for StockWise across 10 retail store locations and 50 unique items (500 distinct store-item time series).

Candidate linear and regularized regression models were compared against an established retail benchmark: the **Weekly Seasonal Naive** baseline. All models were evaluated under strict chronological splits with feature engineering designed to eliminate target data leakage.

### Key Highlights
* **Selected Best Model:** **Ridge Regression**
* **Validation Performance:** MAE of **7.1288** units and RMSE of **9.2449** units ($R^2 = 0.9272$).
* **Improvement over Baseline:** 23.8% reduction in Validation MAE relative to Weekly Seasonal Naive (MAE 9.3519).
* **Final Untouched Test Performance:** Test MAE of **6.7529** units and Test RMSE of **8.7568** units ($R^2 = 0.9055$), outperforming Seasonal Naive test MAE (9.0763) by **25.6%**.
* **Negative Predictions Analysis:** Exactly 0 negative predictions on validation split across all models; 35 negative predictions (35 / 46,000 = 0.076%) on the test period due to unconstrained linear extrapolation during seasonal low-demand troughs.
* **Pipeline Runtime:** Completed full data ingestion, feature generation, 5-model training, evaluation, and chart generation in **28.96 seconds**.

---

## 2. Dataset Partitioning & Forecast Origin Assumption

### Chronological Splits
To prevent lookahead bias and reflect real-world operational forecasting, the dataset was partitioned chronologically:
* **Training Period:** 2013-01-29 to 2017-06-30 (**807,000 rows**)
* **Validation Period:** 2017-07-01 to 2017-09-30 (**46,000 rows**, Q3 2017)
* **Final Test Period:** 2017-10-01 to 2017-12-31 (**46,000 rows**, Q4 2017)

### Initial History Handling
Generating 28-day lag and rolling average features requires an initial history window of 28 consecutive days.
* Exactly **14,000 observations** (28 initial days $\times$ 500 store-item pairs) from January 1, 2013 to January 28, 2013 were dropped from the training set.
* Validation and Test splits contain zero missing feature values because full preceding historical records were preserved.

### Evaluation Assumption
The system models **1-day-ahead rolling forecasts** refreshed daily using observed sales from previous days.
On any target date $t$:
* Actual sales values up to day $t-1$ are known and used for lag features (`lag_1`, `lag_7`, `lag_14`, `lag_28`).
* Rolling window averages (`rolling_mean_7`, `rolling_mean_28`) are shifted by 1 day, guaranteeing that day $t$'s sales **never** enter its own predictor features.
* Day $t$ sales and all future dates remain strictly unseen.

---

## 3. Candidate Models & Architecture

All machine learning models are wrapped within scikit-learn `Pipeline` and `ColumnTransformer` constructs:
1. **OneHotEncoder (`drop='first'`):** Encodes categorical entity IDs (`store`, `item`) and calendar cycles (`dayofweek`, `month`) into 75 dummy indicators.
2. **StandardScaler:** Fits purely on training targets to scale numerical features (`year`, `lag_1`, `lag_7`, `lag_14`, `lag_28`, `rolling_mean_7`, `rolling_mean_28`).

### Evaluated Models:
1. **Weekly Seasonal Naive:** Baseline forecaster setting $\hat{y}_t = y_{t-7}$.
2. **Linear Regression (OLS):** Ordinary least squares solving the normal equations.
3. **Ridge Regression:** L2 regularized regression ($\alpha = 1.0$) mitigating multicollinearity.
4. **Lasso Regression:** L1 regularized regression ($\alpha = 0.1$) enforcing coefficient sparsity.
5. **Elastic Net Regression:** Joint L1/L2 penalty ($\alpha = 0.1$, $l_1 = 0.5$).

---

## 4. Validation Performance & Comparison

| Model | Type | Validation MAE | Validation RMSE | Validation R2 | Negative Preds | Min Pred | Max Pred | Fit Time (s) | Inference Time (s) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ridge Regression | ml | 7.1288 | 9.2449 | 0.9272 | 0 | 4.2956 | 184.7264 | 2.2200 | 0.0600 |
| Linear Regression | ml | 7.1295 | 9.2450 | 0.9272 | 0 | 4.3035 | 184.7311 | 2.6300 | 0.0500 |
| Lasso Regression | ml | 7.3785 | 9.5815 | 0.9218 | 0 | 9.7298 | 192.2916 | 8.1800 | 0.0500 |
| Elastic Net Regression | ml | 7.4817 | 9.7139 | 0.9197 | 0 | 10.9435 | 191.7188 | 7.7300 | 0.0600 |
| Weekly Seasonal Naive | baseline | 9.3519 | 12.1765 | 0.8738 | 0 | 6.0000 | 231.0000 | 0.0000 | 0.0000 |

### Observations:
1. **Ridge Regression and Linear Regression** achieved virtually identical top performance (Validation MAE ~7.13), heavily outperforming the Weekly Seasonal Naive baseline (MAE ~9.35).
2. **Ridge Regression** is preferred over OLS because the L2 penalty stabilizes regression coefficients against multicollinear lag and rolling features without penalizing training runtime (fit time: 2.22s).
3. **Lasso and Elastic Net** produced slightly higher MAE (~7.38 and ~7.48 respectively). The L1 penalty forces some informative lag weights to zero, causing a modest reduction in predictive accuracy while taking longer to converge.

---

## 5. Test Evaluation of Selected Model

Evaluating **Ridge Regression** on the untouched final test split (October 1, 2017 through December 31, 2017) using the fixed-model holdout protocol (evaluating the exact model trained on the training split through 2017-06-30 without retraining on validation data):

| Metric | Weekly Seasonal Naive | Ridge Regression (Selected) | Operational Improvement |
| :--- | :--- | :--- | :--- |
| **Test MAE** | 9.0763 units | **6.7529 units** | **25.6% reduction** |
| **Test RMSE** | 12.0507 units | **8.7568 units** | **27.3% reduction** |
| **Test $R^2$** | 0.8210 | **0.9055** | **+0.0845 explained variance** |
| **Negative Preds** | 0 | **35 (0.076%)** | Minimum prediction: -4.30 |

---

## 6. Inspection of Residuals and Negative Demand

### Negative Predictions
* Physical demand cannot be negative. If an unconstrained linear model forecasts negative sales, operational inventory replenishment systems could fail or order negative stock.
* **Validation Split (Q3 Summer Peak):** 0 negative predictions were observed across all models (minimum predicted demand for Ridge was 4.30 units).
* **Test Split (Q4 Winter Drop):** Exactly 35 out of 46,000 predictions (0.076%) fell slightly below zero (minimum predicted value: -4.30 units). This occurs on low-volume SKU/store combinations during winter trough periods due to linear unconstrained subtraction.
* **Operational Handling:** In production, a defensive floor / non-negative clipping rule (`np.clip(preds, 0, None)`) should be applied before feeding predictions into replenishment order calculators. However, raw evaluation metrics are reported here without synthetic post-processing to ensure evaluation honesty and transparency.

### Residuals Analysis
* Visual inspection of `reports/residuals_analysis.png` reveals an error distribution centered near zero (mean residual: -0.17 units). While largely symmetric, the residuals are leptokurtic with an excess kurtosis of +0.97 (heavier tails than a standard Gaussian distribution).
* Residuals plotted against fitted values exhibit **heteroscedasticity** rather than constant variance: the residual standard deviation increases across fitted-value quartiles from 6.10 units (lowest quartile, fitted 4.30–41.59) to 7.57, 9.53, and 12.37 units (highest quartile, fitted 91.05–184.73), representing an approximate 4.1-fold increase in variance.
* This volume-dependent error dispersion is standard and expected in retail demand modeling, where higher volume store-item pairs naturally experience wider absolute swings in demand. This pattern does not invalidate the model or alter the reported predictive evaluation metrics (Validation MAE 7.1288, Test MAE 6.7529).

---

## 7. Artifact Deliverables

The following output files have been generated:
* `reports/model_comparison.csv`: Tabular evaluation results across all models.
* `reports/model_comparison.png`: Dual-panel comparative bar chart (Validation MAE & RMSE).
* `reports/actual_vs_predicted.png`: Time-series aggregate trajectory and sample scatter plot.
* `reports/residuals_analysis.png`: Residual error distribution and fitted value diagnostics.
* `reports/milestone_2_report.md`: This comprehensive evaluation documentation.
