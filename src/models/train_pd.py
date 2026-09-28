"""
Probability of Default (PD) Modeling Suite (Basel II / IFRS 9 Framework).
Trains and evaluates Baseline (Logistic Regression), Challenger 1 (Random Forest),
and Challenger 2 (XGBoost with scale_pos_weight), followed by Probability Calibration.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"


def compute_gini(roc_auc: float) -> float:
    """Computes the institutional Gini coefficient: Gini = 2 * ROC_AUC - 1."""
    return 2.0 * roc_auc - 1.0


class PDModelSuite:
    """
    Manages end-to-end Probability of Default (PD) model benchmarking,
    calibration, metrics evaluation, and model artifact persistence.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.calibrated_champion: Any = None
        self.metrics: Dict[str, Dict[str, float]] = {}
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    def load_data(
        self, features_path: Optional[Path] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Loads transformed dataset and produces a stratified 80/20 train-test split."""
        path = features_path or (DATA_DIR / "features_transformed.parquet")
        if not path.exists():
            path = DATA_DIR / "features_transformed.csv"
        
        logger.info(f"Loading modeling dataset from {path}")
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        
        X = df.drop(columns=["default_flag"])
        y = df["default_flag"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=self.random_state,
            stratify=y,
        )
        logger.info(
            f"Dataset split complete: Train={X_train.shape[0]:,} loans (Default rate={y_train.mean():.2%}), "
            f"Test={X_test.shape[0]:,} loans (Default rate={y_test.mean():.2%})"
        )
        return X_train, X_test, y_train, y_test

    def train_models(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Dict[str, Any]:
        """Trains Baseline Logistic Regression, Random Forest, and tuned XGBoost."""
        logger.info("Initializing and training PD models...")

        # 1. Baseline Scorecard: Logistic Regression with L2 regularization
        logger.info("Training Baseline Scorecard (Logistic Regression L2)...")
        lr_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=self.random_state)),
        ])
        lr_pipeline.fit(X_train, y_train)
        self.models["Logistic_Regression_L2"] = lr_pipeline

        # 2. Challenger 1: Random Forest Classifier
        logger.info("Training Challenger 1 (Random Forest Classifier)...")
        rf = RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        self.models["Random_Forest"] = rf

        # 3. Challenger 2 (Champion Candidate): XGBoost with class imbalance weight
        logger.info("Training Challenger 2 (XGBoost Classifier with scale_pos_weight)...")
        scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
        logger.info(f"Computed scale_pos_weight: {scale_pos_weight:.2f}")

        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=self.random_state,
            n_jobs=-1,
        )
        xgb.fit(X_train, y_train)
        self.models["XGBoost"] = xgb

        # 4. Probability Calibration on Champion (XGBoost)
        logger.info("Calibrating XGBoost default probabilities (CalibratedClassifierCV - Sigmoid)...")
        calibrated_xgb = CalibratedClassifierCV(
            estimator=xgb,
            method="sigmoid",
            cv=3,
        )
        calibrated_xgb.fit(X_train, y_train)
        self.models["XGBoost_Calibrated"] = calibrated_xgb
        self.calibrated_champion = calibrated_xgb

        return self.models

    def evaluate_models(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> pd.DataFrame:
        """Evaluates all candidate models on test set and computes Basel risk metrics."""
        logger.info("Evaluating models on out-of-time/out-of-sample test set...")
        results = []

        for name, model in self.models.items():
            # Get probability of default (PD)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred_binary = (y_pred_proba >= 0.5).astype(int)

            roc_auc = float(roc_auc_score(y_test, y_pred_proba))
            pr_auc = float(average_precision_score(y_test, y_pred_proba))
            gini = float(compute_gini(roc_auc))
            brier = float(brier_score_loss(y_test, y_pred_proba))

            model_metrics = {
                "ROC_AUC": round(roc_auc, 4),
                "PR_AUC": round(pr_auc, 4),
                "Gini_Coefficient": round(gini, 4),
                "Brier_Score": round(brier, 4),
                "Mean_Predicted_PD": round(float(y_pred_proba.mean()), 4),
                "Empirical_Default_Rate": round(float(y_test.mean()), 4),
            }
            self.metrics[name] = model_metrics
            results.append({"Model": name, **model_metrics})

        results_df = pd.DataFrame(results).sort_values(by="ROC_AUC", ascending=False)
        return results_df

    def plot_diagnostic_curves(
        self, X_test: pd.DataFrame, y_test: pd.Series, output_path: Optional[Path] = None
    ) -> Path:
        """Generates ROC curves, PR curves, and Probability Calibration Curves."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. ROC Curves
        for name, model in self.models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_val = self.metrics[name]["ROC_AUC"]
            axes[0].plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc_val:.3f})")
        axes[0].plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess")
        axes[0].set_title("ROC Curves", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].legend(loc="lower right")

        # 2. Precision-Recall Curves
        for name, model in self.models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = self.metrics[name]["PR_AUC"]
            axes[1].plot(recall, precision, lw=2, label=f"{name} (PR={pr_auc:.3f})")
        axes[1].axhline(y=y_test.mean(), color="k", linestyle="--", lw=1.5, label=f"Baseline ({y_test.mean():.2f})")
        axes[1].set_title("Precision-Recall Curves", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].legend(loc="upper right")

        # 3. Probability Calibration Curves
        for name, model in self.models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
            axes[2].plot(prob_pred, prob_true, marker="o", lw=2, label=f"{name}")
        axes[2].plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfectly Calibrated")
        axes[2].set_title("Probability Calibration Curves", fontsize=12, fontweight="bold")
        axes[2].set_xlabel("Mean Predicted Probability (PD)")
        axes[2].set_ylabel("Fraction of Empirical Defaults")
        axes[2].legend(loc="upper left")

        plt.tight_layout()
        save_file = output_path or (REPORTS_DIR / "roc_pr_calibration_curves.png")
        plt.savefig(save_file, dpi=300)
        plt.close()
        logger.info(f"Diagnostic curves saved to: {save_file}")
        return save_file

    def save_artifacts(self) -> None:
        """Persists trained models and metrics tables to disk."""
        for name, model in self.models.items():
            file_name = f"pd_{name.lower()}.joblib"
            joblib.dump(model, MODELS_DIR / file_name)
        
        # Save champion specifically
        joblib.dump(self.calibrated_champion, MODELS_DIR / "pd_champion_calibrated.joblib")
        logger.info(f"Saved all model artifacts to {MODELS_DIR}")

        # Save metrics JSON & CSV
        metrics_json_path = REPORTS_DIR / "model_performance_metrics.json"
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=4)
        
        comparison_df = pd.DataFrame(self.metrics).T.reset_index().rename(columns={"index": "Model"})
        comparison_df.to_csv(REPORTS_DIR / "model_comparison_table.csv", index=False)
        logger.info(f"Saved evaluation metrics to {REPORTS_DIR}")


def run_pd_training_pipeline() -> Tuple[PDModelSuite, pd.DataFrame]:
    """Orchestrates end-to-end PD modeling suite execution."""
    suite = PDModelSuite(random_state=42)
    X_train, X_test, y_train, y_test = suite.load_data()
    suite.train_models(X_train, y_train)
    results_df = suite.evaluate_models(X_test, y_test)
    suite.plot_diagnostic_curves(X_test, y_test)
    suite.save_artifacts()

    # Save test partition for subsequent Basel & SHAP evaluations
    test_data = X_test.copy()
    test_data["default_flag"] = y_test.values
    test_data.to_parquet(DATA_DIR / "test_features.parquet", index=False)

    return suite, results_df


if __name__ == "__main__":
    suite, comparison_df = run_pd_training_pipeline()
    print("\n" + "="*80)
    print(" " * 25 + "PD MODEL BENCHMARK RESULTS")
    print("="*80)
    print(comparison_df.to_string(index=False))
    print("="*80)
