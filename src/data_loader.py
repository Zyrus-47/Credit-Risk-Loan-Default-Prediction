"""
Data loader and SQL ingestion pipeline for LendingClub credit risk dataset.
Interfaces with DuckDB to run automated data hygiene, filtering, and target labeling.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import duckdb
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SQL_DIR = PROJECT_ROOT / "sql"


class DataLoader:
    """Manages raw loan data ingestion, synthetic benchmarking, and DuckDB SQL ETL."""

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize DataLoader.
        
        Args:
            db_path: Path to persistent DuckDB file. If None, uses in-memory DuckDB.
        """
        self.db_path = db_path or ":memory:"
        self.conn = duckdb.connect(self.db_path)
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    def generate_synthetic_lendingclub_data(
        self,
        n_samples: int = 15000,
        random_state: int = 42,
        output_file: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Generates realistic LendingClub benchmark dataset when local raw data is absent.
        Reflects real empirical credit grade distributions, interest rates, and default dynamics.
        """
        logger.info(f"Generating {n_samples} benchmark LendingClub records with seed {random_state}...")
        np.random.seed(random_state)

        # Grades and sub-grades
        grades = ["A", "B", "C", "D", "E", "F", "G"]
        grade_probs = [0.18, 0.28, 0.26, 0.15, 0.08, 0.03, 0.02]
        chosen_grades = np.random.choice(grades, size=n_samples, p=grade_probs)
        chosen_sub_grades = [f"{g}{np.random.randint(1, 6)}" for g in chosen_grades]

        # Loan amounts and terms
        loan_amnts = np.round(np.random.exponential(scale=10000, size=n_samples) + 2000, -2)
        loan_amnts = np.clip(loan_amnts, 1000, 40000)
        terms = np.random.choice(["36 months", "60 months"], size=n_samples, p=[0.72, 0.28])

        # Interest rates linked to grades
        base_rates = {"A": 7.5, "B": 11.0, "C": 14.5, "D": 18.5, "E": 23.0, "F": 27.5, "G": 30.0}
        int_rates_num = np.array([
            np.clip(base_rates[g] + np.random.normal(0, 1.5), 5.0, 32.0)
            for g in chosen_grades
        ])
        int_rates_str = [f"{r:.2f}%" for r in int_rates_num]

        # Monthly installments: Standard amortization formula installment = P * r / (1 - (1+r)^-n)
        term_months = np.array([36 if t == "36 months" else 60 for t in terms])
        r_monthly = (int_rates_num / 100.0) / 12.0
        installments = np.round(
            loan_amnts * (r_monthly * (1 + r_monthly)**term_months) / ((1 + r_monthly)**term_months - 1),
            2,
        )

        # Borrower income & DTI
        annual_inc = np.round(np.random.lognormal(mean=11.1, sigma=0.55, size=n_samples), -2)
        annual_inc = np.clip(annual_inc, 12000, 400000)
        dtis = np.round(np.random.normal(loc=18.5, scale=7.5, size=n_samples), 2)
        dtis = np.clip(dtis, 2.0, 55.0)

        # Employment length
        emp_categories = [
            "< 1 year", "1 year", "2 years", "3 years", "4 years",
            "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years", None
        ]
        emp_lengths = np.random.choice(emp_categories, size=n_samples, p=[
            0.08, 0.06, 0.09, 0.08, 0.06, 0.07, 0.05, 0.05, 0.04, 0.04, 0.35, 0.03
        ])

        # Home ownership & Verification
        home_ownership = np.random.choice(["MORTGAGE", "RENT", "OWN", "OTHER"], size=n_samples, p=[0.49, 0.40, 0.10, 0.01])
        verification_status = np.random.choice(["Source Verified", "Verified", "Not Verified"], size=n_samples, p=[0.38, 0.33, 0.29])
        purposes = np.random.choice(
            ["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "small_business", "medical"],
            size=n_samples,
            p=[0.58, 0.22, 0.07, 0.05, 0.04, 0.04],
        )

        # Delinquency and credit history
        delinq_2yrs = np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.82, 0.11, 0.04, 0.02, 0.01])
        inq_last_6mths = np.random.choice([0, 1, 2, 3, 4, 5], size=n_samples, p=[0.55, 0.25, 0.12, 0.05, 0.02, 0.01])
        open_acc = np.random.poisson(lam=11, size=n_samples)
        open_acc = np.clip(open_acc, 2, 45)
        total_acc = open_acc + np.random.poisson(lam=12, size=n_samples)
        pub_rec = np.random.choice([0, 1, 2], size=n_samples, p=[0.86, 0.12, 0.02])

        revol_bal = np.round(np.random.exponential(scale=14000, size=n_samples), 0)
        revol_util_num = np.clip(np.random.normal(loc=52.0, scale=24.0, size=n_samples), 0.0, 99.9)
        revol_util_str = [f"{u:.1f}%" for u in revol_util_num]

        # Dates
        issue_d = np.random.choice(["Jan-2016", "Mar-2016", "Jun-2016", "Sep-2016", "Dec-2016", "Feb-2017"], size=n_samples)
        earliest_cr_line = np.random.choice(["Jan-2001", "Jun-1998", "Mar-2005", "Sep-1994", "Nov-2008"], size=n_samples)

        # Statuses: Active loans (Current, In Grace Period, Late) vs Closed (Fully Paid, Charged Off, Default)
        status_pool = [
            "Fully Paid",
            "Charged Off",
            "Current",
            "Default",
            "In Grace Period",
            "Late (16-30 days)",
            "Does not meet the credit policy. Status:Fully Paid",
            "Does not meet the credit policy. Status:Charged Off",
        ]
        # Probability depends on grade
        grade_default_bias = {"A": 0.06, "B": 0.13, "C": 0.22, "D": 0.31, "E": 0.40, "F": 0.48, "G": 0.55}
        loan_statuses = []
        recoveries = []
        collection_recovery_fees = []
        total_rec_prncp = []

        for i in range(n_samples):
            g = chosen_grades[i]
            p_def = grade_default_bias[g]
            
            # 18% of portfolio is ongoing/active (which our SQL filter will sanitize)
            is_ongoing = np.random.rand() < 0.18
            if is_ongoing:
                status = np.random.choice(["Current", "In Grace Period", "Late (16-30 days)"], p=[0.88, 0.08, 0.04])
                rec = 0.0
                fee = 0.0
                rec_prncp = round(loan_amnts[i] * np.random.uniform(0.1, 0.7), 2)
            else:
                is_default = np.random.rand() < p_def
                if is_default:
                    status = np.random.choice(
                        ["Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"],
                        p=[0.85, 0.10, 0.05],
                    )
                    # For defaults: partial principal repaid, rest unrecovered with partial recovery
                    repaid_pct = np.random.uniform(0.05, 0.55)
                    rec_prncp = round(loan_amnts[i] * repaid_pct, 2)
                    unpaid = loan_amnts[i] - rec_prncp
                    # Recovery rate on unpaid balance typically 5% - 25%
                    recovery_rate = np.random.uniform(0.05, 0.28)
                    rec = round(unpaid * recovery_rate, 2)
                    fee = round(rec * 0.15, 2)  # 15% collection fee
                else:
                    status = np.random.choice(
                        ["Fully Paid", "Does not meet the credit policy. Status:Fully Paid"],
                        p=[0.93, 0.07],
                    )
                    rec_prncp = loan_amnts[i]
                    rec = 0.0
                    fee = 0.0

            loan_statuses.append(status)
            recoveries.append(rec)
            collection_recovery_fees.append(fee)
            total_rec_prncp.append(rec_prncp)

        df = pd.DataFrame({
            "id": [f"LC_{1000000 + i}" for i in range(n_samples)],
            "loan_amnt": loan_amnts,
            "funded_amnt": loan_amnts,
            "term": terms,
            "int_rate": int_rates_str,
            "installment": installments,
            "grade": chosen_grades,
            "sub_grade": chosen_sub_grades,
            "emp_title": ["Professional"] * n_samples,
            "emp_length": emp_lengths,
            "home_ownership": home_ownership,
            "annual_inc": annual_inc,
            "verification_status": verification_status,
            "issue_d": issue_d,
            "loan_status": loan_statuses,
            "purpose": purposes,
            "title": ["Loan Request"] * n_samples,
            "dti": dtis,
            "delinq_2yrs": delinq_2yrs,
            "earliest_cr_line": earliest_cr_line,
            "inq_last_6mths": inq_last_6mths,
            "open_acc": open_acc,
            "pub_rec": pub_rec,
            "revol_bal": revol_bal,
            "revol_util": revol_util_str,
            "total_acc": total_acc,
            "total_rec_prncp": total_rec_prncp,
            "recoveries": recoveries,
            "collection_recovery_fee": collection_recovery_fees,
        })

        target_path = output_file or (RAW_DATA_DIR / "lending_club_loans.csv")
        df.to_csv(target_path, index=False)
        logger.info(f"Saved benchmark dataset to {target_path} (Shape: {df.shape})")
        return df

    def execute_sql_file(self, sql_path: Path) -> None:
        """Executes SQL statements from a file inside DuckDB."""
        if not sql_path.exists():
            raise FileNotFoundError(f"SQL file not found: {sql_path}")
        
        logger.info(f"Executing SQL file: {sql_path.name}")
        sql_content = sql_path.read_text(encoding="utf-8")
        
        # Split statements on semicolon while preserving blocks
        statements = [stmt.strip() for stmt in sql_content.split(";") if stmt.strip()]
        for stmt in statements:
            self.conn.execute(stmt)

    def load_and_transform(self, raw_csv_path: Optional[str] = None) -> pd.DataFrame:
        """
        Executes full Step 1 and Step 2 SQL data hygiene pipeline:
        1. Checks for raw CSV or triggers benchmark generation
        2. Ingests raw data into DuckDB table `raw_lending_club_loans`
        3. Runs `01_schema_setup.sql`
        4. Runs `02_data_cleaning.sql`
        5. Runs `03_feature_aggregation.sql`
        6. Validates target distribution and outputs processed datasets
        """
        # Determine CSV source
        if raw_csv_path and os.path.exists(raw_csv_path):
            csv_path = Path(raw_csv_path)
        else:
            default_csv = RAW_DATA_DIR / "lending_club_loans.csv"
            if not default_csv.exists():
                logger.info(f"No existing raw dataset found at {default_csv}. Generating synthetic benchmark dataset.")
                self.generate_synthetic_lendingclub_data(output_file=default_csv)
            csv_path = default_csv

        logger.info(f"Ingesting raw CSV: {csv_path}")

        # 1. Run Schema Setup
        schema_file = SQL_DIR / "01_schema_setup.sql"
        self.execute_sql_file(schema_file)

        # 2. Ingest CSV into raw_lending_club_loans table
        # We replace the table with read_csv_auto to handle type flexibility seamlessly
        self.conn.execute(f"""
            CREATE OR REPLACE TABLE raw_lending_club_loans AS
            SELECT * FROM read_csv_auto('{csv_path.as_posix()}', all_varchar=True);
        """)
        raw_count = self.conn.execute("SELECT COUNT(*) FROM raw_lending_club_loans").fetchone()[0]
        logger.info(f"Raw records ingested into DuckDB: {raw_count:,}")

        # 3. Execute Data Cleaning & Target Extraction
        cleaning_file = SQL_DIR / "02_data_cleaning.sql"
        self.conn.execute("DROP TABLE IF EXISTS cleaned_loans;")
        self.execute_sql_file(cleaning_file)

        # 4. Execute Feature Aggregations
        agg_file = SQL_DIR / "03_feature_aggregation.sql"
        self.conn.execute("DROP TABLE IF EXISTS grade_risk_summary;")
        self.conn.execute("DROP TABLE IF EXISTS purpose_risk_summary;")
        self.conn.execute("DROP TABLE IF EXISTS delinquency_cohort_summary;")
        self.execute_sql_file(agg_file)

        # 5. Extract cleaned data into pandas DataFrame
        cleaned_df = self.conn.execute("SELECT * FROM cleaned_loans").fetchdf()
        logger.info(f"Cleaned and sanitized dataset extracted: {len(cleaned_df):,} records")

        # 6. Save Processed Artifacts
        parquet_out = PROCESSED_DATA_DIR / "cleaned_loans.parquet"
        csv_out = PROCESSED_DATA_DIR / "cleaned_loans.csv"
        cleaned_df.to_parquet(parquet_out, index=False)
        cleaned_df.to_csv(csv_out, index=False)
        logger.info(f"Saved processed datasets to {parquet_out} and {csv_out}")

        return cleaned_df

    def get_summary_statistics(self) -> dict:
        """Extracts validation metrics from DuckDB to verify Step 1 completion."""
        total_cleaned = self.conn.execute("SELECT COUNT(*) FROM cleaned_loans").fetchone()[0]
        default_dist = self.conn.execute("""
            SELECT 
                default_flag,
                COUNT(*) AS count,
                ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM cleaned_loans), 2) AS pct
            FROM cleaned_loans
            GROUP BY default_flag
            ORDER BY default_flag;
        """).fetchdf()

        grade_summary = self.conn.execute("SELECT * FROM grade_risk_summary").fetchdf()
        
        # Check parsing rules:
        term_types = self.conn.execute("SELECT DISTINCT term FROM cleaned_loans ORDER BY term").fetchall()
        int_rate_min_max = self.conn.execute("SELECT MIN(int_rate), MAX(int_rate), AVG(int_rate) FROM cleaned_loans").fetchone()
        revol_util_min_max = self.conn.execute("SELECT MIN(revol_util), MAX(revol_util), AVG(revol_util) FROM cleaned_loans").fetchone()
        emp_length_values = self.conn.execute("SELECT DISTINCT emp_length FROM cleaned_loans ORDER BY emp_length").fetchall()

        return {
            "total_cleaned_loans": total_cleaned,
            "default_distribution": default_dist.to_dict(orient="records"),
            "grade_risk_summary": grade_summary.to_dict(orient="records"),
            "distinct_terms": [t[0] for t in term_types],
            "int_rate_stats": {"min": int_rate_min_max[0], "max": int_rate_min_max[1], "mean": round(int_rate_min_max[2], 2)},
            "revol_util_stats": {"min": revol_util_min_max[0], "max": revol_util_min_max[1], "mean": round(revol_util_min_max[2], 2)},
            "emp_length_values": [e[0] for e in emp_length_values],
        }


if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load_and_transform()
    stats = loader.get_summary_statistics()
    print("\n--- STEP 1 & 2 VERIFICATION SUMMARY ---")
    print(f"Total Cleaned Records: {stats['total_cleaned_loans']:,}")
    print("\nDefault Distribution (0 = Fully Paid, 1 = Default):")
    for row in stats["default_distribution"]:
        print(f"  Target {row['default_flag']}: {row['count']:,} loans ({row['pct']}%)")
    print(f"\nDistinct Parsed Terms: {stats['distinct_terms']}")
    print(f"Interest Rate Stats: {stats['int_rate_stats']}")
    print(f"Revolving Utilization Stats: {stats['revol_util_stats']}")
    print(f"Emp Length Integer Values: {stats['emp_length_values']}")
    print("\nGrade-Level Default Risk Benchmark:")
    for g in stats["grade_risk_summary"]:
        print(f"  Grade {g['grade']}: {g['total_loans']:,} loans | Default Rate: {g['default_rate_pct']}% | Avg Rate: {g['avg_int_rate']}%")
