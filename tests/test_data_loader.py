"""
Unit and integration tests for data loading, SQL hygiene, and target variable formulation.
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from src.data_loader import DataLoader, PROJECT_ROOT, RAW_DATA_DIR, PROCESSED_DATA_DIR


@pytest.fixture(scope="module")
def data_loader_instance():
    """Initializes DataLoader and executes ETL."""
    loader = DataLoader(db_path=":memory:")
    cleaned_df = loader.load_and_transform()
    stats = loader.get_summary_statistics()
    return {"loader": loader, "df": cleaned_df, "stats": stats}


def test_processed_files_exist():
    """Verify raw and processed data artifacts are persisted."""
    assert (RAW_DATA_DIR / "lending_club_loans.csv").exists(), "Raw CSV should exist"
    assert (PROCESSED_DATA_DIR / "cleaned_loans.parquet").exists(), "Processed Parquet should exist"
    assert (PROCESSED_DATA_DIR / "cleaned_loans.csv").exists(), "Processed CSV should exist"


def test_target_definition_and_leakage_filter(data_loader_instance):
    """
    Ensure active/ongoing loans are filtered out to prevent survivorship bias and target leakage.
    Target must be strictly 0 or 1.
    """
    df = data_loader_instance["df"]
    
    # Check invalid statuses are not present
    prohibited_statuses = {"Current", "In Grace Period", "Late (16-30 days)", "Late (31-120 days)"}
    assert not set(df["loan_status"].unique()).intersection(prohibited_statuses), (
        "Active/ongoing loans must be excluded from modeling dataset"
    )

    # Check target default_flag values
    unique_targets = set(df["default_flag"].dropna().unique())
    assert unique_targets.issubset({0, 1}), f"Default flag must only be 0 or 1, found {unique_targets}"
    assert df["default_flag"].isnull().sum() == 0, "Default flag must not contain nulls"


def test_string_parsing_rules(data_loader_instance):
    """
    Validate parsing rules:
    - term stripped of ' months' and cast to integer
    - emp_length converted to integer [0, 10]
    - int_rate and revol_util cleaned of '%' and cast to float
    """
    df = data_loader_instance["df"]

    # Term parsing
    assert pd.api.types.is_integer_dtype(df["term"]), "Term should be integer"
    assert set(df["term"].unique()).issubset({36, 60}), "Terms should be 36 or 60"

    # Interest rate parsing
    assert pd.api.types.is_float_dtype(df["int_rate"]), "int_rate should be float"
    assert df["int_rate"].min() >= 0.0 and df["int_rate"].max() <= 100.0, "int_rate should be between 0 and 100"

    # Revolving utilization parsing
    assert pd.api.types.is_float_dtype(df["revol_util"]), "revol_util should be float"
    assert df["revol_util"].min() >= 0.0, "revol_util should be non-negative"

    # Employment length
    assert pd.api.types.is_integer_dtype(df["emp_length"]), "emp_length should be integer"
    assert df["emp_length"].min() >= 0 and df["emp_length"].max() <= 10, "emp_length should be between 0 and 10"


def test_primary_core_attributes_present(data_loader_instance):
    """Verify all primary core attributes required by specification are present in cleaned dataset."""
    df = data_loader_instance["df"]
    required_cols = [
        "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
        "emp_length", "home_ownership", "annual_inc", "verification_status",
        "dti", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
        "revol_bal", "revol_util", "total_acc", "recoveries",
        "collection_recovery_fee", "total_rec_prncp", "default_flag"
    ]
    for col in required_cols:
        assert col in df.columns, f"Required column '{col}' missing from cleaned dataset"


def test_sql_aggregation_views(data_loader_instance):
    """Verify SQL aggregation tables exist and produce consistent metrics."""
    conn = data_loader_instance["loader"].conn
    
    grade_df = conn.execute("SELECT * FROM grade_risk_summary").fetchdf()
    assert not grade_df.empty, "grade_risk_summary table should have data"
    assert "default_rate_pct" in grade_df.columns
    
    # Grade A should have a lower default rate than Grade G
    a_rate = grade_df[grade_df["grade"] == "A"]["default_rate_pct"].values[0]
    g_rate = grade_df[grade_df["grade"] == "G"]["default_rate_pct"].values[0]
    assert a_rate < g_rate, f"Grade A default rate ({a_rate}%) must be lower than Grade G ({g_rate}%)"
