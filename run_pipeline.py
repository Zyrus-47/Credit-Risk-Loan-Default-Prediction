"""Credit risk modeling and expected loss pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.data_loader import DataLoader
from src.feature_engineering import run_feature_engineering_pipeline
from src.models.train_pd import run_pd_training_pipeline
from src.models.basel_metrics import run_basel_expected_loss_pipeline
from src.models.explainability import run_explainability_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Pipeline")


def main() -> None:
    parser = argparse.ArgumentParser(description="Credit Risk & Expected Loss Pipeline")
    parser.add_argument("--samples", type=int, default=50000, help="Sample size for benchmark data generation")
    parser.add_argument("--force-data-gen", action="store_true", help="Force regenerate raw benchmark dataset")
    args = parser.parse_args()

    start_time = time.time()
    print("\n--- Running Credit Risk Pipeline ---")

    # Data loading and SQL hygiene
    loader = DataLoader()
    raw_csv = PROJECT_ROOT / "data" / "raw" / "lending_club_loans.csv"
    should_gen = args.force_data_gen or not raw_csv.exists()
    cleaned_df = loader.load_and_transform(
        force_regenerate=should_gen,
        n_samples=args.samples,
    )
    data_stats = loader.get_summary_statistics()

    print(f"\nCleaned Records: {data_stats['total_cleaned_loans']:,}")
    for row in data_stats["default_distribution"]:
        label = "Default (1)" if row["default_flag"] == 1 else "Fully Paid (0)"
        print(f"  {label:15s}: {row['count']:,} ({row['pct']}%)")

    # Feature engineering
    X_features, y_target, preprocessor = run_feature_engineering_pipeline(save_artifacts=True)
    print(f"\nFeatures Count: {X_features.shape[1]}")

    # Probability of default modeling
    pd_suite, comparison_df = run_pd_training_pipeline()
    print("\nModel Benchmark Comparison:")
    print(comparison_df.to_string(index=False))

    # Basel expected loss calculations
    scored_portfolio, basel_summary = run_basel_expected_loss_pipeline()
    print(f"\nPortfolio Exposure (EAD):       ${basel_summary['total_exposure_at_default_usd']:,.2f}")
    print(f"Expected Loss Provision (EL):   ${basel_summary['total_expected_loss_provision_usd']:,.2f}")
    print(f"Portfolio Expected Loss Rate:   {basel_summary['portfolio_el_rate_pct']:.2f}%")

    print("\nProvisions by Risk Tier:")
    tier_df = pd.DataFrame(basel_summary["tier_breakdown"])
    print(tier_df.to_string(index=False))

    # Explainability
    global_res, low_sc, high_sc = run_explainability_pipeline()

    print("\nTop Macro Risk Drivers:")
    for idx, d in enumerate(global_res["top_drivers"][:5], 1):
        print(f"  {idx}. {d['feature']:22s} | Impact: {d['mean_abs_shap']:.4f}")

    print(f"\nSample Scorecard (Low Risk):  PD = {low_sc['predicted_pd_pct']:.2f}% | EL = ${low_sc['expected_loss_usd']:,.2f}")
    print(f"Sample Scorecard (High Risk): PD = {high_sc['predicted_pd_pct']:.2f}% | EL = ${high_sc['expected_loss_usd']:,.2f}")

    elapsed = time.time() - start_time
    print(f"\nPipeline finished in {elapsed:.2f}s.\n")


if __name__ == "__main__":
    main()
