# Enterprise Credit Risk & Expected Loss Engine (Basel II / IFRS 9)

A modular, production-ready credit risk modeling platform calculating **Probability of Default (PD)**, **Loss Given Default (LGD)**, **Exposure at Default (EAD)**, and **Expected Loss (EL)** using the LendingClub dataset.

---

## Repository Structure

```
Credit Risk & Loan Default Prediction/
├── data/
│   ├── raw/
│   │   └── lending_club_loans.csv
│   └── processed/
│       ├── cleaned_loans.parquet
│       └── cleaned_loans.csv
├── sql/
│   ├── 01_schema_setup.sql          # Table DDL & schema setup
│   ├── 02_data_cleaning.sql          # SQL hygiene, parsing, and target formulation
│   └── 03_feature_aggregation.sql    # Grade & delinquency cohort aggregation views
├── src/
│   ├── __init__.py
│   ├── data_loader.py                # DuckDB ETL and benchmark data ingestion
│   ├── feature_engineering.py        # Financial feature engineering & encoding (Step 2)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_pd.py               # PD Champion / Challenger models (Step 3)
│   │   ├── basel_metrics.py          # LGD, EAD, and Expected Loss (Step 4)
│   │   └── explainability.py         # SHAP beeswarm & customer scorecard (Step 5)
│   └── dashboard/
│       └── app.py                    # Streamlit underwriter interface (Step 6)
├── notebooks/
│   └── exploratory_data_analysis.ipynb
├── requirements.txt
└── README.md
```

---

## Step 1 & 2 Execution Summary (SQL Extraction & Data Hygiene)

- **Engine**: DuckDB in-memory / file-based SQL engine.
- **Target Formulation**:
  - Filtered OUT active loans: `Current`, `In Grace Period`, `Late (16-30 days)`, `Late (31-120 days)`.
  - Binary Target:
    - Default ($y=1$): `Charged Off`, `Default`, `Does not meet the credit policy. Status:Charged Off`
    - Fully Paid ($y=0$): `Fully Paid`, `Does not meet the credit policy. Status:Fully Paid`
- **String Parsing Rules**:
  - `term`: Stripped `' months'` and cast to integer (`36`, `60`).
  - `emp_length`: Standardized to integer scale (`0` to `10`).
  - `int_rate` & `revol_util`: Cleaned of `%` signs and cast to float.
- **Verification Metrics**:
  - Ingested: 15,000 raw loans.
  - Retained Cleaned Cohort: 12,293 loans (2,707 ongoing loans filtered out).
  - Default Flag Ratio: 79.54% Fully Paid ($y=0$), 20.46% Default ($y=1$).
  - Monotonic Risk: Grade A Default Rate = 5.75% vs. Grade G Default Rate = 52.79%.

---

## Step 2 Execution Summary (Feature Engineering & EDA)

- **Module**: [`src/feature_engineering.py`](file:///e:/Credit%20Risk%20&%20Loan%20Default%20Prediction/src/feature_engineering.py)
- **Hierarchical Imputation**: Imputed financial medians grouped hierarchically by `grade` and `sub_grade` with global median fallback.
- **Winsorization (99th Percentile Capping)**: Robustly capped extreme right-tail skewness on `annual_inc`, `dti`, and `revol_bal`.
- **Domain Financial Ratios**:
  - $\text{installment\_to\_income} = \frac{\text{installment} \times 12}{\text{annual\_inc}}$
  - $\text{credit\_history\_length} = \text{issue\_d} - \text{earliest\_cr\_line}$ (in months)
  - $\text{unutilized\_credit\_limit} = \frac{\text{revol\_bal}}{\text{revol\_util} / 100} - \text{revol\_bal}$
  - $\text{revolving\_debt\_to\_income} = \frac{\text{revol\_bal}}{\text{annual\_inc}}$
- **Encodings**:
  - Ordinal mapping for `grade` (1–7) and `sub_grade` (1–35).
  - One-Hot Encoding for nominal variables (`home_ownership`, `verification_status`, `purpose`).
- **Scorecard Metrics**: Weight of Evidence (WoE) and Information Value (IV) rankings generated.
- **Notebook**: [`notebooks/exploratory_data_analysis.ipynb`](file:///e:/Credit%20Risk%20&%20Loan%20Default%20Prediction/notebooks/exploratory_data_analysis.ipynb) provides visual analytics.
- **Artifacts Saved**:
  - `data/processed/features_transformed.parquet` (41,059 records × 34 features from 50,000 raw loans)
  - `artifacts/models/credit_feature_engineer.joblib` (fitted transformer for inference)

---

## Step 3 Execution Summary (Probability of Default Modeling & Calibration)

- **Dataset Scale**: Scaled raw benchmark generation to **50,000 records**, resulting in **41,059** cleaned completed loans.
- **Stratified Split (80/20)**:
  - Training Set: **32,847 loans** (Default rate: 20.71%)
  - Test Set: **8,212 loans** (Default rate: 20.70%)
- **Models Benchmarked**:
  1. *Baseline Scorecard*: Logistic Regression with L2 Regularization
  2. *Challenger 1*: Random Forest Classifier (balanced weights)
  3. *Challenger 2*: XGBoost with `scale_pos_weight = 3.83`
  4. *Champion*: Calibrated XGBoost via `CalibratedClassifierCV(method='sigmoid')`
- **Key Performance Results (Test Set)**:
  | Model | ROC-AUC | PR-AUC | Gini ($2 \times AUC - 1$) | Brier Score | Mean Predicted PD | Empirical Default Rate |
  |---|---|---|---|---|---|---|
  | **Random Forest** | 0.7009 | 0.3603 | 0.4018 | 0.2141 | 0.4522 | 0.2070 |
  | **Logistic Regression L2** | 0.6990 | 0.3598 | 0.3980 | 0.1514 | 0.2068 | 0.2070 |
  | **XGBoost (Calibrated Champion)** | 0.6985 | 0.3583 | 0.3971 | **0.1510** | **0.2071** | **0.2070** |
  | **XGBoost (Raw)** | 0.6945 | 0.3525 | 0.3889 | 0.2132 | 0.4425 | 0.2070 |
- **Calibration Significance**: Raw XGBoost and Random Forest overpredicted probability levels (~44–45%) due to class balancing weights; **Sigmoid Calibration aligned the mean predicted PD to 20.71%**, matching the ground-truth default rate (20.70%) and optimizing the Brier score to 0.1510.
- **Saved Model & Report Artifacts**:
  - `artifacts/models/pd_champion_calibrated.joblib`
  - `artifacts/models/pd_logistic_regression_l2.joblib`
  - `artifacts/models/pd_random_forest.joblib`
  - `artifacts/models/pd_xgboost.joblib`
  - `artifacts/reports/model_performance_metrics.json`
  - `artifacts/reports/model_comparison_table.csv`
  - `artifacts/reports/roc_pr_calibration_curves.png`

---

## Step 4 Execution Summary (Basel II / IFRS 9 Expected Loss Engine)

- **Module**: [`src/models/basel_metrics.py`](file:///e:/Credit%20Risk%20&%20Loan%20Default%20Prediction/src/models/basel_metrics.py)
- **Mathematical Formulations**:
  - **Expected Loss (EL)**: $\text{EL} = \text{PD} \times \text{LGD} \times \text{EAD}$
  - **Exposure at Default (EAD)**:
    - Origination Scenario: $\text{EAD} = \text{loan\_amnt} \times \text{CCF}$ ($\text{CCF}=1.0$)
    - Historical Scenario: $\text{EAD} = \max(\text{loan\_amnt} - \text{total\_rec\_prncp}, 0)$
  - **Loss Given Default (LGD)**:
    - Historical Net Recovery Rate: $\text{Recovery Rate} = \frac{\text{recoveries} - \text{collection\_recovery\_fee}}{\text{EAD}}$
    - Empirical $\text{LGD} = \text{clip}(1 - \text{Recovery Rate}, 0.0, 1.0)$
    - Multivariate Ridge LGD model trained on historical defaults and bounded in $[0.05, 0.95]$.
- **Portfolio Capital Provisions Breakdown**:
  - Total Portfolio Loans: **41,059**
  - Total Exposure at Default (EAD): **$481,279,900.00**
  - Total Expected Loss Provision (EL): **$85,014,238.34**
  - Unexpected Loss (UL / Capital at Risk): **$162,863,578.63**
  - Portfolio Weighted Average PD: **20.52%**
  - Portfolio Weighted Average LGD: **86.08%**
  - Overall Expected Loss Rate: **17.66%**

### Provisions by Institutional Risk Tier

| Risk Tier | Loan Count | Total Exposure (EAD) | Expected Loss Provision (EL) | Avg PD | Avg LGD | EL Rate (%) |
|---|---|---|---|---|---|---|
| **Low (Grade A-B)** | 7,990 | $93,115,400.00 | $5,971,251.88 | 7.38% | 85.99% | 6.34% |
| **Medium (Grade C-D)** | 19,391 | $234,491,100.00 | $33,668,978.72 | 16.71% | 86.02% | 14.38% |
| **High (Grade E-G)** | 13,678 | $153,673,400.00 | $45,374,007.74 | 34.33% | 86.06% | 29.54% |

- **Saved Artifacts**:
  - `artifacts/models/lgd_ridge_model.joblib`
  - `artifacts/models/lgd_grade_table.json`
  - `artifacts/reports/portfolio_basel_summary.json`
  - `artifacts/reports/basel_provisions_by_tier.csv`
  - `artifacts/reports/expected_loss_distribution.png`
  - `data/processed/portfolio_basel_scored.parquet`



