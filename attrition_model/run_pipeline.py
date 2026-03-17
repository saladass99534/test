"""
Run Pipeline — End-to-End Orchestration
========================================
Single script to run the complete attrition prediction pipeline:
  1. Load and validate raw data
  2. Build leakage-safe master file
  3. Engineer features
  4. Train models (XGBoost + Baseline)
  5. Generate SHAP explanations
  6. Run survival analysis
  7. Run fairness audit
  8. Save all outputs

Usage:
  python run_pipeline.py                   # Full pipeline with tuning
  python run_pipeline.py --no-tune         # Skip hyperparameter tuning (fast)
  python run_pipeline.py --year 2025       # Specify prediction year
  python run_pipeline.py --threshold 0.4   # Specify risk threshold
"""

import argparse
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PREDICTION_HORIZON_MONTHS, TARGET_COLUMN


def main():
    parser = argparse.ArgumentParser(description='Attrition Prediction Pipeline')
    parser.add_argument('--no-tune', action='store_true',
                        help='Skip Optuna hyperparameter tuning')
    parser.add_argument('--year', type=int, default=None,
                        help='Prediction year (default: latest)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Risk threshold for classification')
    parser.add_argument('--skip-data', action='store_true',
                        help='Skip data loading (reuse existing master file)')
    args = parser.parse_args()

    start_time = time.time()

    print("\n" + "🚀 " * 20)
    print("   ATTRITION PREDICTION PIPELINE")
    print(f"   Prediction Horizon: {PREDICTION_HORIZON_MONTHS} months")
    print("🚀 " * 20)

    # ===================================
    # STEP 1: DATA LOADING
    # ===================================
    from data_loader import build_master, load_master

    if args.skip_data:
        print("\n[1/6] Loading existing master file...")
        df = load_master()
    else:
        print("\n[1/6] Building master file from raw data...")
        df = build_master()

    if df.empty:
        print("CRITICAL: No data loaded. Exiting.")
        sys.exit(1)

    # ===================================
    # STEP 2: FEATURE ENGINEERING
    # ===================================
    print("\n[2/6] Engineering features...")
    from feature_engine import build_features, prepare_model_data

    df = build_features(df)

    # ===================================
    # STEP 3: PREPARE MODEL DATA
    # ===================================
    print("\n[3/6] Preparing model matrices...")
    X_train, y_train, X_test, y_test, test_df, feat_names = prepare_model_data(
        df, prediction_year=args.year
    )

    if len(X_train) == 0 or len(X_test) == 0:
        print("CRITICAL: Empty train or test set. Check data and year selection.")
        sys.exit(1)

    # ===================================
    # STEP 4: TRAIN & EVALUATE MODELS
    # ===================================
    print("\n[4/6] Training models...")
    from model_pipeline import run_pipeline as run_model_pipeline

    results = run_model_pipeline(
        X_train, y_train, X_test, y_test, test_df,
        feat_names, full_df=df,
        tune=(not args.no_tune),
        threshold=args.threshold
    )

    # ===================================
    # STEP 5: SAVE PREDICTIONS TO FEATURED DATA
    # ===================================
    print("\n[5/6] Saving enriched predictions...")
    y_pred = results['y_pred']
    y_prob = results['y_prob']

    test_df_final = test_df.copy()
    test_df_final['Predicted_Risk'] = y_prob
    test_df_final['Predicted_Leaver'] = y_pred

    if results.get('top3_df') is not None:
        test_df_final = test_df_final.reset_index(drop=True)
        top3 = results['top3_df'].reset_index(drop=True)
        test_df_final = pd.concat([test_df_final, top3], axis=1)

    from config import PATH_MODEL_OUTPUT
    import pandas as pd
    pred_path = os.path.join(PATH_MODEL_OUTPUT, 'full_predictions.parquet')
    test_df_final.to_parquet(pred_path, index=False)
    print(f"   ✅ Full predictions saved: {pred_path}")

    # ===================================
    # STEP 6: SUMMARY
    # ===================================
    elapsed = time.time() - start_time

    print("\n" + "="*60)
    print("   🎉 PIPELINE COMPLETE")
    print("="*60)
    print(f"   Time:        {elapsed:.1f} seconds")
    print(f"   Train rows:  {len(X_train):,}")
    print(f"   Test rows:   {len(X_test):,}")
    print(f"   Target rate: {y_test.mean():.2%}")
    print(f"   Threshold:   {args.threshold}")

    test_metrics = results.get('test_metrics', {})
    print(f"\n   Test Set Results:")
    print(f"   AUC:       {test_metrics.get('AUC', 0):.4f}")
    print(f"   Recall:    {test_metrics.get('Recall', 0):.4f}")
    print(f"   Precision: {test_metrics.get('Precision', 0):.4f}")
    print(f"   F1:        {test_metrics.get('F1', 0):.4f}")

    high_risk_count = (y_pred == 1).sum()
    print(f"\n   High risk employees:  {high_risk_count}")
    print(f"   Low risk employees:   {(y_pred == 0).sum()}")

    if results.get('fairness_df') is not None and not results['fairness_df'].empty:
        biased = results['fairness_df'][results['fairness_df'].get('Rate_vs_Overall', pd.Series()) > 1.20]
        if not biased.empty:
            print(f"\n   ⚠️  Bias warnings: {len(biased)} groups flagged")
        else:
            print(f"\n   ✅ No fairness bias detected")

    print(f"\n   Next: Run the dashboard:")
    print(f"      streamlit run dashboard.py")
    print("="*60)

    return results


if __name__ == '__main__':
    import pandas as pd
    main()
