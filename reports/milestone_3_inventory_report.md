# StockWise — Milestone 3: Inventory Intelligence Report

**Execution Timestamp:** 2026-10-19 21:35:49  
**Pipeline Runtime:** 10.01 seconds  
**Analyzed Series:** 500 store-item combinations across 10 stores and 50 SKUs  

---

## 1. Executive Summary & Context

Milestone 3 adds a dedicated **Inventory Intelligence Engine** (`src/stockwise/inventory.py`) on top of the demand forecasting engine built in Milestone 2. 

The underlying public dataset contains 913,000 historical sales transactions. It does **NOT** contain historical stock balances, purchase orders, supplier lead times, delivery variances, holding costs, or stockout records. Therefore, this module provides **mathematically pure, parameterized scenario calculations**, enabling retailers to evaluate replenishment policies without fabricating unobserved historical inventory states.

### Key Highlights
* **Demand Segmentation:** Ranks all 500 store-item pairs into empirical velocity/volatility quadrants, identifying **238 critical SKUs** requiring immediate inventory review.
* **Dual Safety-Stock Framework:** Implemented both standard **Historical Sales Variance** (default) and **Forecast Error Variance (RMSE)**.
* **Working Capital Impact:** For high-volume items, transitioning from raw sales standard deviation to model residual RMSE reduces calculated safety stock buffers by **15% to 39%** under theoretical standard normal error assumptions. Nominal service-level targets have not been empirically validated through discrete-event inventory simulation; heteroscedastic, heavy-tailed residuals documented in Milestone 2 may cause realized service levels to differ from nominal targets.
* **Decision Explainability:** Every recommendation outputs a structured audit trail detailing inputs, formulas, assumptions, data limitations, and actionable guidance.

---

## 2. Demand Velocity & Volatility Risk Segmentation

### Categorization Methodology
Store-item combinations are ranked by empirical percentiles:
1. **Demand Velocity Percentile:** Rank of mean daily sales ($p_{vel}$).
2. **Demand Volatility Percentile:** Rank of daily sales standard deviation ($p_{vol}$).
3. **Demand Risk Score:** $0.5 \cdot p_{vel} + 0.5 \cdot p_{vol}$ (prioritization heuristic, strictly **not** a calibrated stockout probability).

Partitions are determined strictly by the empirical median (top half $> 50.0$ percentile, bottom half $\le 50.0$ percentile):

| Quadrant | Description | Store-Item Count | Review Priority | Operational Recommendation |
| :--- | :--- | :---: | :---: | :--- |
| **Quadrant I** | High Velocity - Volatile | 238 | **CRITICAL** | Close daily monitoring, dynamic safety stock, collaborative supplier replenishment. |
| **Quadrant II** | High Velocity - Stable | 12 | **ELEVATED** | High volume revenue drivers; automated lean replenishment with low safety buffers. |
| **Quadrant III** | Low Velocity - Stable | 238 | **STANDARD** | Predictable baseline items; periodic review with minimal inventory risk. |
| **Quadrant IV** | Lumpy / Erratic Demand | 12 | **LOW** | Low volume with intermittent spikes; consider higher safety buffers or order-on-demand. |

The full distribution is visualized in `reports/demand_risk_matrix.png`.

---

## 3. Replenishment Policy Scenarios (Lead Time = 7 Days, Service Level = 95%)

Below is an illustrative cross-section of evaluated scenarios across the 4 quadrants under standard operational assumptions (7-day lead time, 95% cycle service level, minimum order quantity of 10 units):

| SKU Entity | Demand Quadrant | Mean Daily Sales | Lead Time Demand ($d_L$) | Safety Stock (Historical) | ROP (Historical) | Safety Stock (Forecast RMSE) | SS Reduction | Suggested Order (MOQ 10) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Store 2, Item 13 | High Velocity - Volatile | 137.4 | 961.6 | 102.8 | 1064.5 | 62.7 | 39.0% | 730 |
| Store 2, Item 28 | High Velocity - Volatile | 142.7 | 999.0 | 99.8 | 1098.8 | 63.5 | 36.4% | 750 |
| Store 7, Item 38 | High Velocity - Stable | 69.9 | 489.1 | 49.6 | 538.6 | 38.4 | 22.6% | 370 |
| Store 7, Item 36 | High Velocity - Stable | 68.3 | 478.1 | 49.6 | 527.7 | 38.5 | 22.4% | 360 |
| Store 9, Item 26 | Low Velocity - Stable | 62.3 | 435.9 | 49.8 | 485.7 | 36.3 | 27.2% | 330 |
| Store 1, Item 48 | Low Velocity - Stable | 59.6 | 417.2 | 50.1 | 467.3 | 37.5 | 25.2% | 320 |
| Store 7, Item 12 | Lumpy / Erratic Demand | 62.3 | 435.9 | 55.6 | 491.6 | 34.1 | 38.7% | 340 |
| Store 6, Item 35 | Lumpy / Erratic Demand | 63.0 | 441.2 | 55.2 | 496.3 | 35.7 | 35.3% | 340 |

*Note: In the table above, initial on-hand inventory is hypothetically set to 50% of expected lead-time demand to illustrate reorder triggering.*

---

## 4. Key Analytical Insights

### 1. Forecast-Driven Safety Stock vs. Historical Heuristics
* Standard textbook replenishment sizes safety stock using the standard deviation of raw historical demand:
  $$SS_{hist} = Z \cdot \sigma_D \cdot \sqrt{L}$$
* When an accurate machine learning forecaster is available, safety stock only needs to buffer against the **unpredictable forecast error**:
  $$SS_{model} = Z \cdot \text{RMSE}_{s,i} \cdot \sqrt{L}$$
* Because the Milestone 2 Ridge model explains over 90% of variance ($R^2 > 0.90$), $\text{RMSE}_{s,i}$ is significantly smaller than $\sigma_{D,s,i}$. Across the evaluated scenarios, model-driven safety stock reduced calculated buffer quantities by **15% to 39%** under theoretical standard normal error assumptions.
* **Important Statistical Caveat on Nominal Service Levels:** This buffer reduction assumes Gaussian, independent residual errors. Nominal service-level targets (e.g., 90%, 95%, 98%) have not been empirically validated through discrete-event or historical inventory simulation. Because the demand residuals exhibit positive kurtosis (+0.97, heavy tails) and heteroscedasticity, realized cycle service levels may differ from nominal theoretical targets in practice. No specific realized service level has been measured or claimed.

### 2. Methodological Preservation of Untouched Test Set
* In strict adherence to scientific evaluation integrity, out-of-sample forecast residuals were computed **exclusively on the Q3 2017 validation split** (92 daily observations per store-item pair).
* The Q4 2017 test period was **never** used for policy tuning or parameter selection, ensuring its validity as an untouched holdout benchmark.

### 3. Forecasting Horizon Limitations
* The Ridge model is an operational daily refreshed one-day-ahead predictor ($t+1$ given actuals through $t$).
* For multi-day lead times ($L = 3, 7, 14$), expected lead-time demand is approximated as:
  $$\hat{d}_L = L \cdot \max(0.0, \, \hat{y}_{t+1})$$
* **Explicit Assumption:** This calculation assumes the daily demand rate remains stationary at $\hat{y}_{t+1}$ across the lead-time window. True multi-horizon vector forecasts will be explored in future milestones via direct or autoregressive multi-step rollouts.

---

## 5. Artifact Deliverables

The following artifacts have been generated in `reports/`:
* `reports/inventory_replenishment_scenarios.csv`: Complete tabular dataset of scenario calculations across evaluated lead times and service levels.
* `reports/demand_risk_matrix.png`: High-resolution scatter visualization of demand velocity vs. volatility segmentation.
* `reports/milestone_3_inventory_report.md`: This comprehensive documentation report.
