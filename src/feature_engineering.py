"""
Feature Engineering and Preprocessing Pipeline for Credit Risk & Expected Loss Engine.
Implements financial domain ratios, median imputation grouped by grade/sub-grade,
99th percentile winsorization, ordinal/nominal encoding, and Weight of Evidence (WoE).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"


def parse_date_to_months(series: pd.Series) -> pd.Series:
    """Safely converts date strings like 'Jan-2016' or '2016-01-01' to pandas datetime."""
    return pd.to_datetime(series, format="mixed", errors="coerce")


def calculate_woe_iv(
    df: pd.DataFrame,
    feature: str,
    target: str = "default_flag",
    bins: int = 5,
) -> Tuple[pd.DataFrame, float]:
    """
    Computes Weight of Evidence (WoE) and Information Value (IV) for institutional scorecards.
    
    Formula:
        Distr(0) = Non-Defaults in bin / Total Non-Defaults
        Distr(1) = Defaults in bin / Total Defaults
        WoE = ln( Distr(0) / Distr(1) )
        IV = sum( (Distr(0) - Distr(1)) * WoE )
    """
    temp = df[[feature, target]].copy()
    
    if pd.api.types.is_numeric_dtype(temp[feature]) and temp[feature].nunique() > bins:
        temp["bin"] = pd.qcut(temp[feature], q=bins, duplicates="drop").astype(str)
    else:
        temp["bin"] = temp[feature].astype(str)

    grouped = temp.groupby("bin", as_index=False).agg(
        total=(target, "count"),
        defaults=(target, "sum"),
    )
    grouped["non_defaults"] = grouped["total"] - grouped["defaults"]

    total_defaults = grouped["defaults"].sum()
    total_non_defaults = grouped["non_defaults"].sum()

    # Prevent division by zero with Laplace-style smoothing
    grouped["dist_non_default"] = (grouped["non_defaults"] + 0.5) / (total_non_defaults + 1.0)
    grouped["dist_default"] = (grouped["defaults"] + 0.5) / (total_defaults + 1.0)

    grouped["woe"] = np.log(grouped["dist_non_default"] / grouped["dist_default"])
    grouped["iv_component"] = (grouped["dist_non_default"] - grouped["dist_default"]) * grouped["woe"]
    iv = float(grouped["iv_component"].sum())

    return grouped[["bin", "total", "defaults", "non_defaults", "woe", "iv_component"]], iv


class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Production Scikit-Learn compatible Feature Engineering Transformer.
    Fits all imputation medians, percentile caps, and category encodings strictly
    on training data to prevent lookahead bias / data leakage.
    """

    # Standard Ordinal Mappings for Basel Credit Tiers
    GRADE_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
    SUB_GRADE_MAP = {
        f"{g}{i}": (idx * 5) + i
        for idx, g in enumerate(["A", "B", "C", "D", "E", "F", "G"])
        for i in range(1, 6)
    }

    NUMERIC_CAP_COLS = ["annual_inc", "dti", "revol_bal"]
    NOMINAL_COLS = ["home_ownership", "verification_status", "purpose"]

    def __init__(self, cap_percentile: float = 0.99):
        self.cap_percentile = cap_percentile
        self.global_medians_: Dict[str, float] = {}
        self.group_medians_: Dict[Tuple[str, str], Dict[str, float]] = {}
        self.percentile_caps_: Dict[str, float] = {}
        self.nominal_categories_: Dict[str, List[str]] = {}
        self.woe_tables_: Dict[str, pd.DataFrame] = {}
        self.iv_summary_: Dict[str, float] = {}
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CreditFeatureEngineer":
        """Learn statistical medians, percentile caps, and category levels from training data."""
        df = X.copy()
        if y is not None:
            df["default_flag"] = y.values

        # 1. Learn Global and Hierarchical Imputation Medians (grouped by grade & sub_grade)
        impute_cols = [
            "annual_inc", "dti", "revol_util", "revol_bal",
            "open_acc", "total_acc", "delinq_2yrs", "inq_last_6mths", "pub_rec", "emp_length"
        ]
        for col in impute_cols:
            if col in df.columns:
                self.global_medians_[col] = float(df[col].median(skipna=True))

        # Hierarchical median mapping
        if "grade" in df.columns and "sub_grade" in df.columns:
            grouped = df.groupby(["grade", "sub_grade"])[impute_cols].median()
            self.group_medians_ = grouped.to_dict(orient="index")

        # 2. Learn 99th Percentile Caps (Winsorization)
        for col in self.NUMERIC_CAP_COLS:
            if col in df.columns:
                self.percentile_caps_[col] = float(df[col].quantile(self.cap_percentile))

        # 3. Store nominal category levels for One-Hot Encoding
        for col in self.NOMINAL_COLS:
            if col in df.columns:
                self.nominal_categories_[col] = sorted(df[col].dropna().unique().tolist())

        # 4. Compute WoE and IV if target is supplied
        if "default_flag" in df.columns:
            eval_cols = ["grade", "sub_grade", "dti", "annual_inc", "int_rate", "delinq_2yrs", "inq_last_6mths"]
            for col in eval_cols:
                if col in df.columns:
                    woe_df, iv_val = calculate_woe_iv(df, feature=col, target="default_flag")
                    self.woe_tables_[col] = woe_df
                    self.iv_summary_[col] = iv_val

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies missing value imputation, outlier capping, domain ratio engineering, and encoding."""
        df = X.copy()

        # 1. Missing Value Imputation grouped by grade/sub_grade with fallback to global median
        impute_cols = [
            "annual_inc", "dti", "revol_util", "revol_bal",
            "open_acc", "total_acc", "delinq_2yrs", "inq_last_6mths", "pub_rec", "emp_length"
        ]
        for col in impute_cols:
            if col in df.columns:
                if "grade" in df.columns and "sub_grade" in df.columns:
                    # Apply group-level median
                    def fill_group_median(row):
                        if pd.isna(row[col]):
                            key = (row.get("grade"), row.get("sub_grade"))
                            if key in self.group_medians_ and col in self.group_medians_[key] and not pd.isna(self.group_medians_[key][col]):
                                return self.group_medians_[key][col]
                            return self.global_medians_.get(col, 0.0)
                        return row[col]
                    
                    df[col] = df.apply(fill_group_median, axis=1)
                else:
                    df[col] = df[col].fillna(self.global_medians_.get(col, 0.0))

        # 2. Outlier Capping (Winsorization at 99th percentile)
        for col in self.NUMERIC_CAP_COLS:
            if col in df.columns and col in self.percentile_caps_:
                df[col] = np.clip(df[col], a_min=None, a_max=self.percentile_caps_[col])

        # 3. Domain-Specific Financial Features
        # installment_to_income = (installment * 12) / annual_inc
        annual_inc_safe = np.maximum(df["annual_inc"].values, 1.0)
        df["installment_to_income"] = (df["installment"] * 12.0) / annual_inc_safe

        # credit_history_length = issue_d - earliest_cr_line (in months)
        if "issue_d" in df.columns and "earliest_cr_line" in df.columns:
            issue_dt = parse_date_to_months(df["issue_d"])
            earliest_dt = parse_date_to_months(df["earliest_cr_line"])
            months_diff = (issue_dt.dt.year - earliest_dt.dt.year) * 12 + (issue_dt.dt.month - earliest_dt.dt.month)
            # Default to median credit history (~144 months / 12 years) if missing
            df["credit_history_length"] = months_diff.fillna(144.0).clip(lower=0.0)
        else:
            df["credit_history_length"] = 144.0

        # unutilized_credit_limit = (revol_bal / (revol_util / 100)) - revol_bal
        # Safe division when revol_util is very low or 0
        util_pct = np.clip(df["revol_util"].values / 100.0, 0.01, 1.5)
        total_revol_limit = df["revol_bal"].values / util_pct
        df["unutilized_credit_limit"] = np.maximum(total_revol_limit - df["revol_bal"].values, 0.0)

        # revolving_debt_to_income = revol_bal / annual_inc
        df["revolving_debt_to_income"] = df["revol_bal"] / annual_inc_safe

        # 4. Ordinal Risk Encoding for Grade & Sub-Grade
        if "grade" in df.columns:
            df["grade_num"] = df["grade"].map(self.GRADE_MAP).fillna(4)
        if "sub_grade" in df.columns:
            df["sub_grade_num"] = df["sub_grade"].map(self.SUB_GRADE_MAP).fillna(18)

        # 5. One-Hot Encoding for Nominal Variables
        for col in self.NOMINAL_COLS:
            if col in df.columns and col in self.nominal_categories_:
                for cat in self.nominal_categories_[col]:
                    clean_cat = str(cat).lower().replace(" ", "_").replace("-", "_")
                    df[f"{col}_{clean_cat}"] = (df[col] == cat).astype(int)

        # Drop non-feature raw metadata strings from modeling matrix
        drop_cols = [
            "id", "grade", "sub_grade", "home_ownership", "verification_status",
            "purpose", "issue_d", "earliest_cr_line", "loan_status", "default_flag",
            "recoveries", "collection_recovery_fee", "total_rec_prncp"
        ]
        feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        self.feature_names_ = feature_df.columns.tolist()

        return feature_df


def run_feature_engineering_pipeline(
    input_path: Optional[Path] = None,
    save_artifacts: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, CreditFeatureEngineer]:
    """
    Executes end-to-end Step 2 Feature Engineering:
    - Reads cleaned dataset
    - Fits CreditFeatureEngineer
    - Exports transformed dataset and fitted preprocessor
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    source_file = input_path or (PROCESSED_DATA_DIR / "cleaned_loans.parquet")
    if not source_file.exists():
        # Fallback to CSV if parquet does not exist
        source_file = PROCESSED_DATA_DIR / "cleaned_loans.csv"

    logger.info(f"Loading cleaned data from: {source_file}")
    df = pd.read_parquet(source_file) if source_file.suffix == ".parquet" else pd.read_csv(source_file)

    y = df["default_flag"].copy()
    X = df.drop(columns=["default_flag"])

    # Fit and transform
    engineer = CreditFeatureEngineer(cap_percentile=0.99)
    engineer.fit(X, y)
    X_transformed = engineer.transform(X)

    logger.info(f"Feature transformation complete. Feature matrix shape: {X_transformed.shape}")
    logger.info(f"Domain features created: installment_to_income, credit_history_length, unutilized_credit_limit, revolving_debt_to_income")
    logger.info(f"Information Value (IV) Highlights:")
    for feat, iv in sorted(engineer.iv_summary_.items(), key=lambda x: x[1], reverse=True):
        tier = "Suspicious/Extreme" if iv > 0.5 else ("Strong" if iv > 0.3 else ("Medium" if iv > 0.1 else "Weak"))
        logger.info(f"  {feat:15s} | IV: {iv:.4f} ({tier} predictor)")

    if save_artifacts:
        preprocessor_path = MODELS_DIR / "credit_feature_engineer.joblib"
        joblib.dump(engineer, preprocessor_path)
        logger.info(f"Saved fitted preprocessor to: {preprocessor_path}")

        # Combine features with target for modeling and save
        full_transformed = X_transformed.copy()
        full_transformed["default_flag"] = y.values
        out_parquet = PROCESSED_DATA_DIR / "features_transformed.parquet"
        full_transformed.to_parquet(out_parquet, index=False)
        logger.info(f"Saved transformed dataset to: {out_parquet}")

    return X_transformed, y, engineer


if __name__ == "__main__":
    X_trans, y_target, eng = run_feature_engineering_pipeline()
    print("\n--- STEP 2: FEATURE ENGINEERING VERIFICATION ---")
    print(f"Transformed Features Count: {X_trans.shape[1]}")
    print(f"Sample Engineered Features:")
    sample_cols = [
        "installment_to_income", "credit_history_length",
        "unutilized_credit_limit", "revolving_debt_to_income",
        "grade_num", "sub_grade_num"
    ]
    print(X_trans[sample_cols].head())
    print("\nInformation Value (IV) Ranking:")
    for feat, iv in sorted(eng.iv_summary_.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat:20s}: IV = {iv:.4f}")
