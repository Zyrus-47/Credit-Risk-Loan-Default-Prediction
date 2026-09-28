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
├── tests/
│   └── test_data_loader.py           # Test suite for ETL and SQL sanitization
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
