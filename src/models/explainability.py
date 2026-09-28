"""SHAP-based feature attribution and loan scorecards."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.models.basel_metrics import BaselExpectedLossEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"


class CreditRiskExplainer:
    """Generates portfolio and borrower-level SHAP explanations."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        calibrated_model_path: Optional[Path] = None,
        preprocessor_path: Optional[Path] = None,
        lgd_model_path: Optional[Path] = None,
    ):
        self.model_path = model_path or (MODELS_DIR / "pd_xgboost.joblib")
        self.calibrated_model_path = calibrated_model_path or (MODELS_DIR / "pd_champion_calibrated.joblib")
        self.preprocessor_path = preprocessor_path or (MODELS_DIR / "credit_feature_engineer.joblib")
        self.lgd_model_path = lgd_model_path or (MODELS_DIR / "lgd_ridge_model.joblib")

        self.xgb_model = joblib.load(self.model_path)
        self.calibrated_model = (
            joblib.load(self.calibrated_model_path)
            if self.calibrated_model_path.exists()
            else None
        )
        self.preprocessor = (
            joblib.load(self.preprocessor_path)
            if self.preprocessor_path.exists()
            else None
        )
        self.lgd_model = (
            joblib.load(self.lgd_model_path)
            if self.lgd_model_path.exists()
            else None
        )

        # Initialize SHAP TreeExplainer on XGBoost
        logger.info("Initializing SHAP TreeExplainer on XGBoost model...")
        self.explainer = shap.TreeExplainer(self.xgb_model)

    def generate_global_explainability(
        self,
        X_sample: pd.DataFrame,
        sample_size: int = 1500,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Generates macro risk driver insights across the portfolio:
        1. SHAP Beeswarm Summary Plot
        2. Global Feature Importance Bar Chart
        3. Quantified macro driver ranking JSON
        """
        save_dir = output_dir or REPORTS_DIR
        save_dir.mkdir(parents=True, exist_ok=True)

        if len(X_sample) > sample_size:
            X_eval = X_sample.sample(n=sample_size, random_state=42)
        else:
            X_eval = X_sample.copy()

        logger.info(f"Computing SHAP values for {len(X_eval)} portfolio samples...")
        shap_values = self.explainer(X_eval)

        # 1. Beeswarm Summary Plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_eval, show=False, max_display=15)
        plt.title("Portfolio SHAP Summary: Macro Credit Risk Drivers", fontsize=14, fontweight="bold", pad=15)
        beeswarm_path = save_dir / "shap_summary_beeswarm.png"
        plt.tight_layout()
        plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved SHAP Beeswarm to: {beeswarm_path}")

        # 2. Global Importance Bar Chart
        plt.figure(figsize=(10, 6))
        shap.plots.bar(shap_values, max_display=15, show=False)
        plt.title("Top 15 Credit Risk Feature Importances (Mean |SHAP|)", fontsize=13, fontweight="bold", pad=12)
        bar_path = save_dir / "shap_feature_importance_bar.png"
        plt.tight_layout()
        plt.savefig(bar_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved SHAP Feature Importance Bar to: {bar_path}")

        # 3. Macro Risk Drivers Table
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        feature_importance = pd.DataFrame({
            "feature": X_eval.columns,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values(by="mean_abs_shap", ascending=False)

        top_drivers = feature_importance.head(10).to_dict(orient="records")
        drivers_json = save_dir / "global_risk_drivers.json"
        with open(drivers_json, "w", encoding="utf-8") as f:
            json.dump(top_drivers, f, indent=4)

        return {
            "beeswarm_plot": str(beeswarm_path),
            "bar_plot": str(bar_path),
            "top_drivers": top_drivers,
        }

    def explain_customer_risk(
        self,
        loan_id: str,
        feature_vector: pd.DataFrame,
        loan_amnt: Optional[float] = None,
        lgd: Optional[float] = None,
        ead: Optional[float] = None,
        output_chart_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Generates comprehensive local customer scorecard for loan officers:
        - Calibrated PD & Basel Risk Rating Tier
        - LGD and EAD dollar calculations
        - Total Expected Loss in $ (EL = PD * LGD * EAD)
        - SHAP Waterfall decomposition of risk factors
        - Top risk-increasing vs. risk-reducing attributes
        """
        # Ensure single row DataFrame
        if isinstance(feature_vector, pd.Series):
            df_row = feature_vector.to_frame().T
        else:
            df_row = feature_vector.head(1).copy()

        # If raw borrower vector provided, transform through pipeline
        if self.preprocessor is not None and "term" in df_row.columns and "sub_grade" in df_row.columns:
            raw_loan_amnt = float(df_row["loan_amnt"].iloc[0]) if "loan_amnt" in df_row.columns else 10000.0
            model_input = self.preprocessor.transform(df_row)
        else:
            raw_loan_amnt = float(df_row["loan_amnt"].iloc[0]) if "loan_amnt" in df_row.columns else 10000.0
            model_input = df_row.copy()

        # Ensure correct column alignment and pure numeric float dtypes for XGBoost
        expected_cols = self.xgb_model.feature_names_in_
        missing_cols = [c for c in expected_cols if c not in model_input.columns]
        for c in missing_cols:
            model_input[c] = 0.0
        model_input = model_input[expected_cols]

        for col in model_input.columns:
            if model_input[col].dtype == object or str(model_input[col].dtype) == "category":
                model_input[col] = pd.to_numeric(model_input[col], errors="coerce").fillna(0.0)
            model_input[col] = model_input[col].astype(float)

        # 1. Probability of Default (Calibrated)
        if self.calibrated_model is not None:
            pd_value = float(self.calibrated_model.predict_proba(model_input)[0, 1])
        else:
            pd_value = float(self.xgb_model.predict_proba(model_input)[0, 1])

        # 2. Risk Rating Tier
        risk_rating = BaselExpectedLossEngine.assign_risk_rating(pd_series=[pd_value]).iloc[0]

        # 3. EAD and LGD
        effective_loan_amnt = loan_amnt if loan_amnt is not None else raw_loan_amnt
        effective_ead = ead if ead is not None else float(effective_loan_amnt)

        if lgd is not None:
            effective_lgd = float(lgd)
        elif self.lgd_model is not None:
            lgd_features = ["loan_amnt", "installment", "int_rate", "dti", "annual_inc"]
            available = [c for c in lgd_features if c in model_input.columns]
            lgd_in = model_input[available].copy()
            effective_lgd = float(np.clip(self.lgd_model.predict(lgd_in)[0], 0.05, 0.95))
        else:
            effective_lgd = 0.86  # Benchmark LendingClub average

        # 4. Expected Loss ($)
        expected_loss_usd = round(pd_value * effective_lgd * effective_ead, 2)
        el_rate_pct = round((expected_loss_usd / max(effective_ead, 1.0)) * 100.0, 2)

        # 5. SHAP Local Explanation & Decomposition
        shap_explanation = self.explainer(model_input)[0]
        base_value = float(shap_explanation.base_values)
        feature_names = model_input.columns.tolist()
        values = shap_explanation.values

        factor_impacts = []
        for feat, val, x_val in zip(feature_names, values, model_input.iloc[0]):
            impact_direction = "Risk Increasing (Default +)" if val > 0 else "Risk Mitigating (Default -)"
            factor_impacts.append({
                "feature": feat,
                "borrower_value": round(float(x_val), 2),
                "shap_impact": round(float(val), 4),
                "direction": impact_direction,
            })

        # Sort factors by absolute impact
        factor_impacts.sort(key=lambda x: abs(x["shap_impact"]), reverse=True)
        risk_increasing_factors = [f for f in factor_impacts if f["shap_impact"] > 0][:5]
        risk_mitigating_factors = [f for f in factor_impacts if f["shap_impact"] < 0][:5]

        save_chart = output_chart_path or (REPORTS_DIR / f"customer_scorecard_{loan_id}.png")
        plt.figure(figsize=(8, 4.2))
        shap.plots.waterfall(shap_explanation, max_display=8, show=False)
        plt.title(f"Scorecard Attribution: Loan {loan_id} (PD: {pd_value:.1%})", fontsize=11, fontweight="bold", pad=10)
        plt.tight_layout()
        plt.savefig(save_chart, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved local customer scorecard waterfall to: {save_chart}")

        scorecard = {
            "loan_id": str(loan_id),
            "predicted_pd_pct": round(pd_value * 100.0, 2),
            "risk_tier": risk_rating,
            "loss_given_default_pct": round(effective_lgd * 100.0, 2),
            "exposure_at_default_usd": round(effective_ead, 2),
            "expected_loss_usd": expected_loss_usd,
            "expected_loss_rate_pct": el_rate_pct,
            "base_log_odds": round(base_value, 4),
            "top_risk_increasing_factors": risk_increasing_factors,
            "top_risk_mitigating_factors": risk_mitigating_factors,
            "waterfall_chart_path": str(save_chart),
        }

        return scorecard


def run_explainability_pipeline() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Runs global SHAP evaluation and generates sample low-risk and high-risk scorecards."""
    explainer = CreditRiskExplainer()

    # Load transformed features
    features_path = DATA_DIR / "features_transformed.parquet"
    if not features_path.exists():
        features_path = DATA_DIR / "features_transformed.csv"
    df = pd.read_parquet(features_path) if features_path.suffix == ".parquet" else pd.read_csv(features_path)
    X = df.drop(columns=["default_flag"])

    # 1. Global Explainability
    global_results = explainer.generate_global_explainability(X, sample_size=1500)

    # 2. Local Scorecard: Sample Low Risk Loan (Grade A/B candidate)
    low_risk_idx = X[X["grade_num"] == 1].index[0] if (X["grade_num"] == 1).any() else 0
    low_risk_vector = X.iloc[[low_risk_idx]]
    low_risk_scorecard = explainer.explain_customer_risk(
        loan_id=f"BORROWER_{low_risk_idx}_LOW_RISK",
        feature_vector=low_risk_vector,
        loan_amnt=float(low_risk_vector["loan_amnt"].iloc[0]),
    )

    # 3. Local Scorecard: Sample High Risk Loan (Grade E/F/G candidate)
    high_risk_candidates = X[X["grade_num"] >= 5]
    high_risk_idx = high_risk_candidates.index[0] if not high_risk_candidates.empty else 1
    high_risk_vector = X.iloc[[high_risk_idx]]
    high_risk_scorecard = explainer.explain_customer_risk(
        loan_id=f"BORROWER_{high_risk_idx}_HIGH_RISK",
        feature_vector=high_risk_vector,
        loan_amnt=float(high_risk_vector["loan_amnt"].iloc[0]),
    )

    return global_results, low_risk_scorecard, high_risk_scorecard


if __name__ == "__main__":
    global_res, low_sc, high_sc = run_explainability_pipeline()
    print("\nMacro Risk Drivers:")
    for idx, d in enumerate(global_res["top_drivers"], 1):
        print(f"  {idx:2d}. {d['feature']:25s} | Impact: {d['mean_abs_shap']:.4f}")

    print("\nLow Risk Borrower Profile:")
    print(f"Loan ID:             {low_sc['loan_id']}")
    print(f"Predicted PD:        {low_sc['predicted_pd_pct']:.2f}%")
    print(f"Risk Tier:           {low_sc['risk_tier']}")
    print(f"Expected Loss (EL):  ${low_sc['expected_loss_usd']:,.2f} ({low_sc['expected_loss_rate_pct']:.2f}%)")

    print("\nHigh Risk Borrower Profile:")
    print(f"Loan ID:             {high_sc['loan_id']}")
    print(f"Predicted PD:        {high_sc['predicted_pd_pct']:.2f}%")
    print(f"Risk Tier:           {high_sc['risk_tier']}")
    print(f"Expected Loss (EL):  ${high_sc['expected_loss_usd']:,.2f} ({high_sc['expected_loss_rate_pct']:.2f}%)")
