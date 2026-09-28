"""
Basel II / IFRS 9 Expected Loss Engine.
Calculates Probability of Default (PD), Exposure at Default (EAD),
Loss Given Default (LGD), and Expected Loss (EL = PD * LGD * EAD).
Includes institutional risk tier segmentation and regulatory capital provisions.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"


class BaselExpectedLossEngine:
    """
    Implements Basel II / IFRS 9 Expected Capital Provisions:
    EL = PD * LGD * EAD
    """

    DEFAULT_CCF = 1.0  # Credit Conversion Factor for fixed term installment loans

    def __init__(self, ccf: float = 1.0):
        self.ccf = ccf
        self.lgd_model: Optional[Ridge] = None
        self.grade_median_lgd_: Dict[str, float] = {}
        self.global_median_lgd_: float = 0.80
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def calculate_ead(
        loan_amnt: Union[pd.Series, np.ndarray, float],
        total_rec_prncp: Optional[Union[pd.Series, np.ndarray, float]] = None,
        ccf: float = 1.0,
    ) -> Union[pd.Series, np.ndarray, float]:
        """
        Calculates Exposure at Default (EAD).
        - For historical/defaulted: EAD = max(loan_amnt - total_rec_prncp, 0)
        - For active origination: EAD = loan_amnt * CCF
        """
        if total_rec_prncp is not None:
            # Historical unpaid exposure
            if isinstance(loan_amnt, pd.Series):
                ead = (loan_amnt - total_rec_prncp).clip(lower=0.0)
            elif isinstance(loan_amnt, np.ndarray):
                ead = np.maximum(loan_amnt - total_rec_prncp, 0.0)
            else:
                ead = max(float(loan_amnt) - float(total_rec_prncp), 0.0)
        else:
            # Origination exposure with Credit Conversion Factor
            ead = loan_amnt * ccf

        return ead

    @staticmethod
    def calculate_historical_lgd(
        ead: pd.Series,
        recoveries: pd.Series,
        collection_recovery_fee: pd.Series,
    ) -> pd.Series:
        """
        Calculates empirical historical LGD for defaulted loans:
        Recovery Rate = (recoveries - collection_recovery_fee) / EAD
        LGD = 1 - Recovery Rate, bounded between [0.0, 1.0].
        """
        safe_ead = np.maximum(ead.values, 1.0)
        net_recoveries = np.maximum(recoveries.values - collection_recovery_fee.values, 0.0)
        recovery_rate = np.clip(net_recoveries / safe_ead, 0.0, 1.0)
        lgd = np.clip(1.0 - recovery_rate, 0.0, 1.0)
        return pd.Series(lgd, index=ead.index)

    def fit_lgd_model(self, cleaned_loans_df: pd.DataFrame) -> "BaselExpectedLossEngine":
        """
        Trains Ridge LGD regression model and calculates grade-level empirical median LGDs
        from historical defaulted loans.
        """
        logger.info("Fitting secondary LGD model on historical defaulted loans...")
        defaults_df = cleaned_loans_df[cleaned_loans_df["default_flag"] == 1].copy()

        if defaults_df.empty:
            raise ValueError("No defaulted loans found in dataset to train LGD model.")

        # 1. Compute historical EAD and LGD
        defaults_df["ead"] = self.calculate_ead(
            defaults_df["loan_amnt"], defaults_df["total_rec_prncp"]
        )
        defaults_df["historical_lgd"] = self.calculate_historical_lgd(
            defaults_df["ead"],
            defaults_df["recoveries"],
            defaults_df["collection_recovery_fee"],
        )

        # 2. Grade-level empirical median LGDs
        self.grade_median_lgd_ = (
            defaults_df.groupby("grade")["historical_lgd"].median().to_dict()
        )
        self.global_median_lgd_ = float(defaults_df["historical_lgd"].median())
        logger.info(f"Grade Empirical Median LGDs: {self.grade_median_lgd_}")

        # 3. Train Ridge Regression for dynamic borrower LGD estimation
        feature_cols = ["loan_amnt", "installment", "int_rate", "dti", "annual_inc"]
        available_cols = [c for c in feature_cols if c in defaults_df.columns]

        X_lgd = defaults_df[available_cols].fillna(defaults_df[available_cols].median())
        y_lgd = defaults_df["historical_lgd"]

        self.lgd_model = Ridge(alpha=10.0, random_state=42)
        self.lgd_model.fit(X_lgd, y_lgd)
        logger.info("Fitted Ridge LGD Model successfully.")

        # Persist LGD artifacts
        joblib.dump(self.lgd_model, MODELS_DIR / "lgd_ridge_model.joblib")
        with open(MODELS_DIR / "lgd_grade_table.json", "w", encoding="utf-8") as f:
            json.dump(self.grade_median_lgd_, f, indent=4)

        return self

    def predict_lgd(
        self,
        df: pd.DataFrame,
        method: str = "ridge",
    ) -> np.ndarray:
        """
        Predicts Loss Given Default (LGD) bounded in [0.05, 0.95].
        Method options: 'ridge' (multivariate model) or 'grade_median' (tabular benchmark).
        """
        if method == "grade_median" or self.lgd_model is None:
            if "grade" in df.columns:
                lgd_preds = df["grade"].map(self.grade_median_lgd_).fillna(self.global_median_lgd_).values
            else:
                lgd_preds = np.full(len(df), self.global_median_lgd_)
        else:
            feature_cols = ["loan_amnt", "installment", "int_rate", "dti", "annual_inc"]
            available = [c for c in feature_cols if c in df.columns]
            X = df[available].copy()
            for col in available:
                X[col] = X[col].fillna(X[col].median())
            raw_preds = self.lgd_model.predict(X)
            lgd_preds = np.clip(raw_preds, 0.05, 0.95)

        return np.round(lgd_preds, 4)

    @staticmethod
    def assign_risk_rating(
        grade: Optional[pd.Series] = None,
        pd_series: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> pd.Series:
        """
        Maps loans into institutional Risk Ratings:
        - Low: Grade A–B (or PD < 0.10)
        - Medium: Grade C–D (or 0.10 <= PD < 0.25)
        - High: Grade E–G (or PD >= 0.25)
        """
        if pd_series is not None:
            pd_vals = np.asarray(pd_series)
            tiers = np.where(
                pd_vals < 0.10,
                "Low (Grade A-B)",
                np.where(pd_vals < 0.25, "Medium (Grade C-D)", "High (Grade E-G)"),
            )
            return pd.Series(tiers)

        if grade is not None:
            return grade.map(
                lambda g: "Low (Grade A-B)"
                if g in ["A", "B"]
                else ("Medium (Grade C-D)" if g in ["C", "D"] else "High (Grade E-G)")
            )

        raise ValueError("Either grade or pd_series must be provided.")

    def compute_portfolio_expected_loss(
        self,
        portfolio_df: pd.DataFrame,
        pd_array: np.ndarray,
        is_origination: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Computes loan-level and portfolio-level Basel Expected Loss (EL = PD * LGD * EAD).
        """
        df = portfolio_df.copy()
        n = len(df)
        assert len(pd_array) == n, "pd_array length must match portfolio_df"

        df["pd"] = pd_array

        # 1. EAD
        if is_origination or "total_rec_prncp" not in df.columns:
            df["ead"] = self.calculate_ead(df["loan_amnt"], ccf=self.ccf)
        else:
            df["ead"] = self.calculate_ead(df["loan_amnt"], df["total_rec_prncp"], ccf=self.ccf)

        # 2. LGD
        df["lgd"] = self.predict_lgd(df, method="ridge")

        # 3. Expected Loss (EL = PD * LGD * EAD)
        df["expected_loss"] = np.round(df["pd"] * df["lgd"] * df["ead"], 2)
        df["el_rate_pct"] = np.round((df["expected_loss"] / np.maximum(df["ead"], 1.0)) * 100.0, 2)

        # 4. Institutional Risk Tiers
        grade_col = df["grade"] if "grade" in df.columns else None
        df["risk_tier"] = self.assign_risk_rating(grade=grade_col, pd_series=df["pd"])

        # 5. Unexpected Loss (UL) approximation: UL = EAD * sqrt(PD * var_lgd + LGD^2 * PD * (1 - PD))
        var_lgd = 0.04  # standard conservative Basel empirical LGD variance
        df["unexpected_loss"] = np.round(
            df["ead"] * np.sqrt(df["pd"] * var_lgd + (df["lgd"]**2) * df["pd"] * (1.0 - df["pd"])),
            2,
        )

        # Portfolio Aggregations
        total_funded = float(df["loan_amnt"].sum())
        total_ead = float(df["ead"].sum())
        total_el = float(df["expected_loss"].sum())
        total_ul = float(df["unexpected_loss"].sum())
        weighted_pd = float(np.average(df["pd"], weights=df["ead"]))
        weighted_lgd = float(np.average(df["lgd"], weights=df["ead"]))
        overall_el_rate = float((total_el / total_ead) * 100.0)

        # Segment breakdown
        tier_summary = df.groupby("risk_tier").agg(
            loan_count=("expected_loss", "count"),
            total_funded=("loan_amnt", "sum"),
            total_ead=("ead", "sum"),
            total_expected_loss=("expected_loss", "sum"),
            avg_pd=("pd", "mean"),
            avg_lgd=("lgd", "mean"),
            el_rate_pct=("el_rate_pct", "mean"),
        ).reset_index()

        portfolio_metrics = {
            "total_portfolio_loans": n,
            "total_funded_volume_usd": round(total_funded, 2),
            "total_exposure_at_default_usd": round(total_ead, 2),
            "total_expected_loss_provision_usd": round(total_el, 2),
            "total_unexpected_loss_usd": round(total_ul, 2),
            "portfolio_weighted_pd_pct": round(weighted_pd * 100.0, 2),
            "portfolio_weighted_lgd_pct": round(weighted_lgd * 100.0, 2),
            "portfolio_el_rate_pct": round(overall_el_rate, 2),
            "tier_breakdown": tier_summary.to_dict(orient="records"),
        }

        return df, portfolio_metrics

    def plot_loss_distributions(
        self, scored_df: pd.DataFrame, output_path: Optional[Path] = None
    ) -> Path:
        """Visualizes expected loss dollar distributions across risk ratings."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 1. Total Provisions by Risk Rating
        tier_agg = scored_df.groupby("risk_tier")["expected_loss"].sum().reset_index()
        colors = ["#2ecc71", "#f39c12", "#e74c3c"]
        axes[0].bar(tier_agg["risk_tier"], tier_agg["expected_loss"], color=colors[:len(tier_agg)], edgecolor="black")
        axes[0].set_title("Total Expected Loss Capital Provisions by Risk Tier", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("Expected Capital Provision ($)")
        axes[0].yaxis.set_major_formatter("${x:,.0f}")
        for idx, row in tier_agg.iterrows():
            axes[0].text(idx, row["expected_loss"] + (row["expected_loss"] * 0.02), f"${row['expected_loss']:,.0f}", ha="center", fontweight="bold")

        # 2. EL vs Loan Amount Scatter colored by PD
        sc = axes[1].scatter(
            scored_df["loan_amnt"],
            scored_df["expected_loss"],
            c=scored_df["pd"],
            cmap="coolwarm",
            alpha=0.6,
            edgecolors="none",
        )
        cbar = plt.colorbar(sc, ax=axes[1])
        cbar.set_label("Probability of Default (PD)")
        axes[1].set_title("Expected Loss ($) vs. Loan Amount", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Loan Amount ($)")
        axes[1].set_ylabel("Expected Loss ($)")
        axes[1].xaxis.set_major_formatter("${x:,.0f}")
        axes[1].yaxis.set_major_formatter("${x:,.0f}")

        plt.tight_layout()
        save_file = output_path or (REPORTS_DIR / "expected_loss_distribution.png")
        plt.savefig(save_file, dpi=300)
        plt.close()
        logger.info(f"Loss distribution plot saved to: {save_file}")
        return save_file

    def save_reports(self, portfolio_metrics: Dict[str, Any], scored_df: pd.DataFrame) -> None:
        """Persists Basel regulatory summaries and loss provisioning tables."""
        summary_json = REPORTS_DIR / "portfolio_basel_summary.json"
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(portfolio_metrics, f, indent=4)

        tier_df = pd.DataFrame(portfolio_metrics["tier_breakdown"])
        tier_df.to_csv(REPORTS_DIR / "basel_provisions_by_tier.csv", index=False)

        scored_df.to_parquet(DATA_DIR / "portfolio_basel_scored.parquet", index=False)
        logger.info(f"Saved Basel capital reports and scored portfolio to {REPORTS_DIR}")


def run_basel_expected_loss_pipeline() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Executes Step 4 Basel Expected Loss calculation on the cleaned loan portfolio."""
    # 1. Load cleaned data for LGD training and portfolio scoring
    cleaned_parquet = DATA_DIR / "cleaned_loans.parquet"
    if not cleaned_parquet.exists():
        cleaned_parquet = DATA_DIR / "cleaned_loans.csv"
    logger.info(f"Loading cleaned data for Basel engine from: {cleaned_parquet}")
    cleaned_df = pd.read_parquet(cleaned_parquet) if cleaned_parquet.suffix == ".parquet" else pd.read_csv(cleaned_parquet)

    # 2. Load trained Champion PD model and preprocessor
    champion_path = MODELS_DIR / "pd_champion_calibrated.joblib"
    preprocessor_path = MODELS_DIR / "credit_feature_engineer.joblib"
    if not champion_path.exists() or not preprocessor_path.exists():
        raise FileNotFoundError("Champion PD model or preprocessor missing. Please run Step 3 first.")

    champion_pd_model = joblib.load(champion_path)
    preprocessor = joblib.load(preprocessor_path)

    # 3. Transform features and predict PD
    X_features = preprocessor.transform(cleaned_df.drop(columns=["default_flag"]))
    calibrated_pd = champion_pd_model.predict_proba(X_features)[:, 1]

    # 4. Initialize and fit Basel Engine
    engine = BaselExpectedLossEngine(ccf=1.0)
    engine.fit_lgd_model(cleaned_df)

    # 5. Compute Expected Loss & Provisions
    scored_df, metrics = engine.compute_portfolio_expected_loss(
        portfolio_df=cleaned_df,
        pd_array=calibrated_pd,
        is_origination=True,
    )
    engine.plot_loss_distributions(scored_df)
    engine.save_reports(metrics, scored_df)

    return scored_df, metrics


if __name__ == "__main__":
    scored_portfolio, portfolio_summary = run_basel_expected_loss_pipeline()
    print("\n" + "="*85)
    print(" " * 20 + "BASEL II / IFRS 9 EXPECTED LOSS PORTFOLIO SUMMARY")
    print("="*85)
    print(f"Total Portfolio Loans:            {portfolio_summary['total_portfolio_loans']:,}")
    print(f"Total Funded Volume:              ${portfolio_summary['total_funded_volume_usd']:,.2f}")
    print(f"Total Exposure at Default (EAD):   ${portfolio_summary['total_exposure_at_default_usd']:,.2f}")
    print(f"Total Expected Loss Provision (EL):${portfolio_summary['total_expected_loss_provision_usd']:,.2f}")
    print(f"Total Unexpected Loss (UL):       ${portfolio_summary['total_unexpected_loss_usd']:,.2f}")
    print(f"Portfolio Weighted Avg PD:        {portfolio_summary['portfolio_weighted_pd_pct']:.2f}%")
    print(f"Portfolio Weighted Avg LGD:       {portfolio_summary['portfolio_weighted_lgd_pct']:.2f}%")
    print(f"Portfolio Expected Loss Rate:     {portfolio_summary['portfolio_el_rate_pct']:.2f}%\n")
    print("Capital Provisions by Institutional Risk Tier:")
    tier_df = pd.DataFrame(portfolio_summary["tier_breakdown"])
    print(tier_df.to_string(index=False))
    print("="*85)
