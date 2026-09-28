# Credit Risk Modeling & Expected Loss Engine

An end-to-end credit risk analytics platform for **loan default prediction, Probability of Default (PD), Loss Given Default (LGD), Exposure at Default (EAD), Expected Loss (EL), portfolio risk segmentation, model explainability, and interactive loan underwriting**.

The project uses LendingClub loan portfolio data and combines machine learning, financial risk modeling, feature engineering, SQL analytics, and an interactive Streamlit dashboard.

---

## Dashboard

### Single Loan Underwriting

The underwriting interface allows a user to enter borrower, loan, and credit-bureau characteristics and evaluate the resulting credit-risk profile.

![Portfolio Capital Provisions](Images/Screenshot%202026-09-28%20173957.png)

### Portfolio Capital Provisions

The portfolio view summarizes exposure, PD, LGD, expected loss, capital requirements, and portfolio risk distribution.

![Portfolio Capital Provisions](Images/Screenshot%202026-09-28%20174007.png)

---

## What This Project Does

The system follows a credit-risk workflow similar to an end-to-end underwriting and portfolio-risk pipeline:

```text
LendingClub Loan Data
        ↓
Data Cleaning & Target Definition
        ↓
Feature Engineering
        ↓
PD Model Training
        ↓
Probability Calibration
        ↓
PD + LGD + EAD
        ↓
Expected Loss Calculation
        ↓
Risk Segmentation
        ↓
SHAP Explainability
        ↓
Portfolio Risk Analysis
        ↓
Streamlit Underwriting Dashboard
```

The application supports two major use cases:

### 1. Single Loan Underwriting

For an individual borrower, the system estimates:

- Probability of Default (PD)
- Loss Given Default (LGD)
- Exposure at Default (EAD)
- Expected Loss (EL)
- Risk Grade / Risk Tier

### 2. Portfolio Risk Analysis

For the loan portfolio, the system aggregates:

- Total exposure
- Expected loss
- Average PD
- Average LGD
- Risk-tier distribution
- Capital / provisioning metrics
- Credit-grade risk concentration

---

# Credit Risk Framework

## Probability of Default (PD)

PD represents the estimated probability that a borrower will default on a loan.

The project benchmarks multiple classification models and calibrates the final probability estimates before using them in Expected Loss calculations.

## Loss Given Default (LGD)

LGD represents the percentage of exposure expected to be lost after considering recoveries.

The project estimates LGD using historical recovery information and a bounded Ridge Regression approach.

Predictions are constrained to a practical range:

```text
5% ≤ LGD ≤ 95%
```

## Exposure at Default (EAD)

EAD represents the exposure expected to be outstanding when default occurs.

For origination scenarios, the project uses:

```text
EAD = Loan Amount × CCF
```

with a CCF assumption of `1.0`.

For historical loans, EAD is estimated from the remaining principal exposure.

## Expected Loss (EL)

The core credit-risk calculation is:

```text
Expected Loss = PD × LGD × EAD
```

This converts model predictions into a monetary estimate of expected credit loss.

---

# Dataset

The project uses **LendingClub loan portfolio data** containing borrower characteristics, loan characteristics, credit-history variables, repayment information, recovery information, and loan-status outcomes.

The dataset includes variables from several categories.

### Loan Characteristics

- Loan amount
- Loan term
- Interest rate
- Installment
- Loan grade
- Loan sub-grade
- Loan purpose

### Borrower Characteristics

- Annual income
- Employment length
- Home ownership
- Income verification
- Debt-to-income ratio

### Credit Bureau Variables

- Revolving balance
- Revolving utilization
- Number of open accounts
- Total credit lines
- Credit history length
- Recent inquiries
- Delinquencies

### Loan Performance

Loan status is converted into a binary default target.

Default examples include:

```text
Charged Off
Default
Does not meet the credit policy: Charged Off
```

Non-default examples include:

```text
Fully Paid
Does not meet the credit policy: Fully Paid
```

Ongoing loan statuses are excluded from the completed-loan default modeling target to avoid treating incomplete outcomes as observed defaults/non-defaults.

---

# Data Processing

The preprocessing pipeline includes:

### Data Cleaning

- Remove incomplete/ongoing loan outcomes
- Standardize loan terms
- Convert percentage fields to numeric values
- Normalize employment length
- Handle missing values
- Remove unusable variables

### Hierarchical Imputation

Financial variables are imputed using borrower risk segments where appropriate:

```text
Credit Grade + Sub-Grade Median
            ↓
Global Median Fallback
```

### Winsorization

Selected extreme financial values are capped at the 99th percentile to reduce the influence of extreme outliers.

Variables include:

- Annual income
- DTI
- Revolving balance

---

# Feature Engineering

Additional financial and credit-risk features are created from the raw variables.

### Installment-to-Income

```text
(Installment × 12) / Annual Income
```

Measures annualized payment burden relative to income.

### Credit History Length

```text
Issue Date - Earliest Credit Line
```

Converted into months.

### Unutilized Credit Limit

```text
(Revolving Balance / Revolving Utilization) - Revolving Balance
```

Provides an estimate of remaining revolving credit capacity.

### Revolving Debt-to-Income

```text
Revolving Balance / Annual Income
```

Measures revolving debt relative to borrower income.

---

# Feature Encoding

### Ordinal Variables

Credit grades and sub-grades are mapped to ordered numerical values so the models can capture their inherent risk ordering.

```text
Grade: A → G
Sub-Grade: A1 → G5
```

### Nominal Variables

Categorical variables without an intrinsic ordering are handled using one-hot encoding.

---

# Machine Learning Models

Three primary models are benchmarked for Probability of Default.

## Logistic Regression

Used as an interpretable baseline model.

Advantages:

- Highly interpretable
- Strong baseline for credit-risk modeling
- Produces probability estimates
- Easy to audit

## Random Forest

A tree-based ensemble model used to capture nonlinear relationships and feature interactions.

## XGBoost

Gradient-boosted decision trees are used as a nonlinear challenger model.

Class imbalance is addressed using a positive-class weighting strategy such as:

```text
scale_pos_weight
```

The final XGBoost probability output is calibrated before being used as the primary PD estimate.

---

# Probability Calibration

Raw machine-learning probabilities are not automatically reliable as credit-risk probabilities.

The project therefore applies:

```text
CalibratedClassifierCV
```

using sigmoid calibration.

Calibration improved the Brier score from approximately:

```text
Before calibration: 0.2132
After calibration:  0.1510
```

This is particularly important because PD directly affects the Expected Loss calculation.

---

# Model Evaluation

The models are evaluated on an out-of-sample test set.

### Evaluation Metrics

- ROC-AUC
- PR-AUC
- Gini coefficient
- Brier score
- Mean predicted PD

### Model Results

| Model | ROC-AUC | PR-AUC | Gini | Brier Score | Mean Predicted PD |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.7009 | 0.3603 | 0.4018 | 0.2141 | 45.22% |
| Logistic Regression | 0.6990 | 0.3598 | 0.3980 | 0.1514 | 20.68% |
| XGBoost — Calibrated | 0.6985 | 0.3583 | 0.3971 | 0.1510 | 20.71% |
| XGBoost — Raw | 0.6945 | 0.3525 | 0.3889 | 0.2132 | 44.25% |

The calibrated XGBoost model is used as the primary PD engine because the project considers both discrimination and probability calibration.

---

# Risk Segmentation

Borrowers are grouped into three risk tiers using credit grade and predicted PD.

| Risk Tier | Credit Grade | PD Range |
|---|---|---:|
| Low | A–B | < 10% |
| Medium | C–D | 10% – 25% |
| High | E–G | ≥ 25% |

These tiers translate model output into an interpretable business-level risk classification.

---

# Portfolio Expected Loss

The portfolio engine aggregates loan-level PD, LGD, and EAD estimates.

Current portfolio analysis contains approximately:

```text
Completed Loans: 41,059
Total Exposure:  $481.28M
```

### Portfolio Risk Summary

| Risk Tier | Loans | Exposure | Expected Loss | Avg PD | Avg LGD | EL Rate |
|---|---:|---:|---:|---:|---:|---:|
| Low | 7,990 | $93.12M | $5.97M | 7.38% | 85.99% | 6.34% |
| Medium | 19,391 | $234.49M | $33.67M | 16.71% | 86.02% | 14.38% |
| High | 13,678 | $153.67M | $45.37M | 34.33% | 86.06% | 29.54% |
| **Total** | **41,059** | **$481.28M** | **$85.01M** | **20.52%** | **86.08%** | **17.66%** |

---

# Explainability

Credit-risk models need to provide more than predictions. Analysts also need to understand which borrower characteristics influence a prediction.

The project uses **SHAP (SHapley Additive exPlanations)** for model explainability.

### Global Explainability

`TreeExplainer` is used to analyze portfolio-level feature importance and identify the variables that contribute most strongly to model predictions.

### Local Explainability

For individual borrowers, SHAP waterfall explanations decompose the prediction into risk-increasing and risk-reducing contributions.

Conceptually:

```text
Base Prediction
      +
Feature Contributions
      ↓
Final PD Prediction
```

---

# Interactive Streamlit Dashboard

The project includes an interactive Streamlit application designed around a credit-analyst workflow.

## Single Loan Underwriting

Users can enter:

### Loan Terms

- Loan amount
- Term
- Interest rate
- Credit grade
- Sub-grade

### Borrower Financial Profile

- Annual income
- DTI
- Employment length
- Home ownership
- Income verification

### Credit Bureau History

- Revolving balance
- Revolving utilization
- Delinquencies
- Recent inquiries
- Open accounts
- Total credit lines

The dashboard then calculates the corresponding risk metrics.

## Portfolio Capital Provisions

The portfolio section provides aggregated analysis of:

- Portfolio exposure
- Expected loss
- PD
- LGD
- EAD
- Risk tiers
- Credit-grade distribution
- Capital/provisioning metrics

## Model Governance

The model-governance section presents information related to:

- Model methodology
- Model version
- Dataset
- Evaluation metrics
- Probability calibration
- Risk segmentation
- Modeling assumptions
- Limitations

---

# SQL Layer

SQL is used as part of the data-processing and portfolio analytics pipeline.

```text
sql/
├── 01_schema_setup.sql
├── 02_data_cleaning.sql
└── 03_feature_aggregation.sql
```

### Schema Setup

Creates the required tables and indexes.

### Data Cleaning

Handles filtering, type conversion, target definition, and data preparation.

### Feature Aggregation

Creates portfolio-level aggregations such as credit-grade statistics, risk cohorts, portfolio exposure, and default statistics.

---

# Project Structure

```text
Credit-Risk-Loan-Default-Prediction/
│
├── data/
│   ├── raw/
│   │   └── lending_club_loans.csv
│   │
│   └── processed/
│       ├── cleaned_loans.parquet
│       ├── features_transformed.parquet
│       ├── portfolio_basel_scored.parquet
│       └── test_features.parquet
│
├── Images/
│   ├── Screenshot 2026-09-28 173957.png
│   └── Screenshot 2026-09-28 174007.png
│
├── sql/
│   ├── 01_schema_setup.sql
│   ├── 02_data_cleaning.sql
│   └── 03_feature_aggregation.sql
│
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   │
│   ├── models/
│   │   ├── train_pd.py
│   │   ├── basel_metrics.py
│   │   └── explainability.py
│   │
│   └── dashboard/
│       └── app.py
│
├── notebooks/
│   └── exploratory_data_analysis.ipynb
│
├── artifacts/
│   ├── models/
│   └── reports/
│
├── requirements.txt
├── run_pipeline.py
└── README.md
```

---

# Tech Stack

| Category | Technology |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Data Storage | Parquet |
| SQL / Analytics | SQL, DuckDB |
| Visualization | Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Classification | Logistic Regression, Random Forest, XGBoost |
| Risk Modeling | PD / LGD / EAD |
| Calibration | CalibratedClassifierCV |

---

# Installation

Clone the repository:

```bash
git clone https://github.com/Zyrus-47/Credit-Risk-Loan-Default-Prediction.git
cd Credit-Risk-Loan-Default-Prediction
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Pipeline

Run the complete ETL, feature engineering, model training, evaluation, risk scoring, and export pipeline:

```bash
python run_pipeline.py
```

The pipeline generates processed datasets, trained model artifacts, evaluation outputs, and portfolio-level risk scores.

---

# Running the Dashboard

Launch the Streamlit application:

```bash
streamlit run src/dashboard/app.py
```

The application will open in your browser.

---

# Key Outputs

### Loan-Level

- Probability of Default
- PD
- LGD
- EAD
- Expected Loss
- Risk Tier
- SHAP explanation

### Portfolio-Level

- Total exposure
- Expected loss
- Average PD
- Average LGD
- Risk distribution
- Risk-tier exposure
- Provisioning metrics

### Model-Level

- ROC-AUC
- PR-AUC
- Gini coefficient
- Brier score
- Calibration performance
- Feature importance

---

# Why This Project Matters

A basic loan-default project typically stops at:

```text
Dataset → ML Model → Accuracy
```

This project extends that workflow into a more realistic credit-risk analytics system:

```text
Raw Loan Data
      ↓
Data Engineering
      ↓
Feature Engineering
      ↓
PD Modeling
      ↓
Probability Calibration
      ↓
LGD Modeling
      ↓
EAD Estimation
      ↓
Expected Loss
      ↓
Risk Segmentation
      ↓
Explainability
      ↓
Portfolio Analytics
      ↓
Underwriting Dashboard
```

The result connects machine-learning predictions with credit-risk concepts and business-level financial analytics.

---

# Limitations

This project is intended as a portfolio and educational implementation of credit-risk modeling rather than a production banking model.

Important limitations include:

- LendingClub data represents a historical lending population and underwriting process.
- Model performance may change on different populations or time periods.
- The train/test split is stratified rather than a fully out-of-time validation framework.
- LGD is modeled using historical recovery information and does not represent a regulatory LGD model.
- EAD assumptions are simplified for origination scenarios.
- The implementation does not constitute regulatory capital compliance.
- Production deployment would require stronger model validation, monitoring, stress testing, and governance.
- A production IFRS 9 implementation would additionally require appropriate staging, lifetime ECL methodology, forward-looking macroeconomic scenarios, and related governance.

---
