# Credit Risk Modeling & Expected Loss Engine

An end-to-end credit risk analytics platform implementing Probability of Default (PD), Loss Given Default (LGD), Exposure at Default (EAD), and Basel II / IFRS 9 Expected Loss (EL) calculations using LendingClub loan portfolio data.

---

## Architecture Overview

```
Credit Risk & Loan Default Prediction/
├── data/
│   ├── raw/
│   │   └── lending_club_loans.csv
│   └── processed/
│       ├── cleaned_loans.parquet
│       ├── features_transformed.parquet
│       ├── portfolio_basel_scored.parquet
│       └── test_features.parquet
├── sql/
│   ├── 01_schema_setup.sql          # Table DDL and indexes
│   ├── 02_data_cleaning.sql          # Filtering, type casting, target definition
│   └── 03_feature_aggregation.sql    # Grade and cohort aggregations
├── src/
│   ├── __init__.py
│   ├── data_loader.py                # DuckDB ingestion and ETL
│   ├── feature_engineering.py        # Imputation, capping, and financial ratios
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_pd.py               # PD benchmark and probability calibration
│   │   ├── basel_metrics.py          # LGD, EAD, and Expected Loss calculations
│   │   └── explainability.py         # SHAP feature attribution and scorecards
│   └── dashboard/
│       └── app.py                    # Streamlit underwriting application
├── notebooks/
│   └── exploratory_data_analysis.ipynb
├── artifacts/
│   ├── models/                       # Serialized model weights
│   └── reports/                      # Evaluation tables and diagnostic plots
├── requirements.txt
├── run_pipeline.py                   # Pipeline runner
└── README.md
```

---

## Methodology

### 1. Data Cleaning & Target Definition
- Active and ongoing loans (`Current`, `In Grace Period`, `Late`) are removed to avoid survivorship bias and target leakage.
- Target formulation:
  - Default ($y=1$): `Charged Off`, `Default`, `Does not meet the credit policy. Status:Charged Off`
  - Fully Paid ($y=0$): `Fully Paid`, `Does not meet the credit policy. Status:Fully Paid`
- Text fields are normalized: term cast to integer (`36`, `60`), employment length standardized to integers (`0` to `10`), and percentage symbols stripped from rates and utilization.

### 2. Feature Engineering
- **Hierarchical Imputation**: Financial medians computed grouped by credit grade and sub-grade with global median fallback.
- **Winsorization**: 99th percentile capping applied to extreme right-tail distributions (`annual_inc`, `dti`, `revol_bal`).
- **Financial Domain Ratios**:
  - Installment to Income: $(\text{installment} \times 12) / \text{annual\_inc}$
  - Credit History Length: $\text{issue\_d} - \text{earliest\_cr\_line}$ (in months)
  - Unutilized Credit Limit: $(\text{revol\_bal} / (\text{revol\_util} / 100)) - \text{revol\_bal}$
  - Revolving Debt to Income: $\text{revol\_bal} / \text{annual\_inc}$
- **Encoding**: Ordinal mapping for risk tiers (Grade: 1–7, Sub-Grade: 1–35) and one-hot encoding for nominal categories.

### 3. Probability of Default (PD) & Calibration
- Stratified 80/20 train-test partition.
- Models benchmarked:
  - Baseline: Logistic Regression (L2 regularization)
  - Challenger 1: Random Forest Classifier
  - Challenger 2: XGBoost with class imbalance weighting (`scale_pos_weight`)
- **Sigmoid Calibration**: Post-processing via `CalibratedClassifierCV` aligns raw model scores with empirical default rates, lowering the Brier score from 0.2132 to 0.1510.

### 4. Basel II Expected Loss Engine
- Formula: $\text{Expected Loss (EL)} = \text{PD} \times \text{LGD} \times \text{EAD}$
- **Exposure at Default (EAD)**: $\text{loan\_amnt} \times \text{CCF}$ for origination ($\text{CCF}=1.0$) or $\max(\text{loan\_amnt} - \text{total\_rec\_prncp}, 0)$ for historical loans.
- **Loss Given Default (LGD)**: Derived from net recovery rates and modeled via Ridge regression bounded in $[0.05, 0.95]$.
- **Risk Tiers**:
  - Low Risk: Grade A–B (PD < 10%)
  - Medium Risk: Grade C–D (10% $\le$ PD < 25%)
  - High Risk: Grade E–G (PD $\ge$ 25%)

### 5. SHAP Feature Attribution
- TreeExplainer calculates global feature importance and beeswarm distributions across the portfolio.
- Local waterfall attributions decompose individual credit decisions into log-odds risk-increasing and risk-mitigating factors.

---

## Model Evaluation Results

Evaluated on out-of-sample test partition (8,212 loans, empirical default rate: 20.70%):

| Model | ROC-AUC | PR-AUC | Gini | Brier Score | Mean Predicted PD |
|---|---|---|---|---|---|
| **Random Forest** | 0.7009 | 0.3603 | 0.4018 | 0.2141 | 0.4522 |
| **Logistic Regression L2** | 0.6990 | 0.3598 | 0.3980 | 0.1514 | 0.2068 |
| **XGBoost (Calibrated Champion)** | 0.6985 | 0.3583 | 0.3971 | **0.1510** | **0.2071** |
| **XGBoost (Raw)** | 0.6945 | 0.3525 | 0.3889 | 0.2132 | 0.4425 |

---

## Portfolio Capital Provisions

Summary across 41,059 completed loans ($481.28M total exposure):

| Risk Tier | Loan Count | Total Exposure (EAD) | Expected Loss (EL) | Avg PD | Avg LGD | EL Rate (%) |
|---|---|---|---|---|---|---|
| **Low (Grade A-B)** | 7,990 | $93,115,400.00 | $5,971,251.88 | 7.38% | 85.99% | 6.34% |
| **Medium (Grade C-D)** | 19,391 | $234,491,100.00 | $33,668,978.72 | 16.71% | 86.02% | 14.38% |
| **High (Grade E-G)** | 13,678 | $153,673,400.00 | $45,374,007.74 | 34.33% | 86.06% | 29.54% |
| **Total** | **41,059** | **$481,279,900.00** | **$85,014,238.34** | **20.52%** | **86.08%** | **17.66%** |

---

## Quickstart

### 1. Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Pipeline
Executes ETL, model training, evaluation, Basel metrics, and scorecard exports:
```bash
python run_pipeline.py
```

### 3. Launch Dashboard
Starts the underwriting interface:
```bash
streamlit run src/dashboard/app.py
```
