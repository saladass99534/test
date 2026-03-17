"""
Model Pipeline — XGBoost, Survival Analysis, SHAP, CV, Fairness
================================================================
Replaces the model training from Cell 3 of the original notebook.
Key improvements:
  - XGBoost as primary model (replaces Random Forest)
  - SMOTE for class imbalance
  - Stratified K-Fold cross-validation
  - Optuna hyperparameter tuning
  - SHAP explainability (global + per-employee)
  - Cox PH survival analysis
  - Fairness / bias audit
  - Logistic Regression baseline
"""

import pandas as pd
import numpy as np
import os
import warnings
import joblib
import json

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, recall_score, precision_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
import shap

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print("WARNING: imbalanced-learn not installed. SMOTE disabled.")

try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False
    print("WARNING: lifelines not installed. Survival analysis disabled.")

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("WARNING: optuna not installed. Hyperparameter tuning disabled.")

from config import (
    TARGET_COLUMN, PATH_MODEL_OUTPUT, XGBOOST_DEFAULT_PARAMS,
    CV_FOLDS, OPTUNA_N_TRIALS, PROTECTED_COLUMNS,
    SURVIVAL_DURATION_COL, SURVIVAL_EVENT_COL,
    BASE_FEATURES,
)

warnings.filterwarnings('ignore')


# =============================================================
# 1. CROSS-VALIDATION EVALUATION
# =============================================================

def evaluate_cv(model, X, y, n_splits=CV_FOLDS):
    """Run stratified K-fold CV and return metrics."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    auc_scores = []
    recall_scores = []
    precision_scores = []
    f1_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Apply SMOTE to training fold only
        if HAS_IMBLEARN and y_tr.sum() > 5:
            try:
                smote = SMOTE(random_state=42, k_neighbors=min(5, int(y_tr.sum()) - 1))
                X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
            except Exception:
                X_tr_res, y_tr_res = X_tr, y_tr
        else:
            X_tr_res, y_tr_res = X_tr, y_tr

        model.fit(X_tr_res, y_tr_res)
        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)

        try:
            auc_scores.append(roc_auc_score(y_val, y_prob))
        except ValueError:
            auc_scores.append(0)
        recall_scores.append(recall_score(y_val, y_pred, zero_division=0))
        precision_scores.append(precision_score(y_val, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_val, y_pred, zero_division=0))

    metrics = {
        'AUC': f"{np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}",
        'Recall': f"{np.mean(recall_scores):.3f} ± {np.std(recall_scores):.3f}",
        'Precision': f"{np.mean(precision_scores):.3f} ± {np.std(precision_scores):.3f}",
        'F1': f"{np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}",
        'AUC_mean': np.mean(auc_scores),
        'Recall_mean': np.mean(recall_scores),
        'Precision_mean': np.mean(precision_scores),
        'F1_mean': np.mean(f1_scores),
    }
    return metrics


# =============================================================
# 2. HYPERPARAMETER TUNING (OPTUNA)
# =============================================================

def tune_xgboost(X_train, y_train, n_trials=OPTUNA_N_TRIALS):
    """Use Optuna to find optimal XGBoost hyperparameters."""
    if not HAS_OPTUNA:
        print("   Optuna not available. Using default params.")
        return XGBOOST_DEFAULT_PARAMS

    print(f"\n   🔍 Running Optuna hyperparameter search ({n_trials} trials)...")

    # Calculate class weight
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos = n_neg / max(n_pos, 1)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0),
            'scale_pos_weight': scale_pos,
            'eval_metric': 'auc',
            'random_state': 42,
            'use_label_encoder': False,
        }
        model = XGBClassifier(**params)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        auc_scores = []

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            if HAS_IMBLEARN and y_tr.sum() > 5:
                try:
                    smote = SMOTE(random_state=42,
                                  k_neighbors=min(5, int(y_tr.sum()) - 1))
                    X_tr, y_tr = smote.fit_resample(X_tr, y_tr)
                except Exception:
                    pass

            model.fit(X_tr, y_tr, verbose=False)
            y_prob = model.predict_proba(X_val)[:, 1]
            try:
                auc_scores.append(roc_auc_score(y_val, y_prob))
            except ValueError:
                auc_scores.append(0)

        return np.mean(auc_scores)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_params['scale_pos_weight'] = scale_pos
    best_params['eval_metric'] = 'auc'
    best_params['random_state'] = 42
    best_params['use_label_encoder'] = False

    print(f"   ✅ Best AUC: {study.best_value:.4f}")
    print(f"   Best params: {json.dumps(best_params, indent=2, default=str)}")

    return best_params


# =============================================================
# 3. TRAIN FINAL MODEL
# =============================================================

def train_xgboost(X_train, y_train, params=None, tune=True):
    """Train the primary XGBoost model."""
    print("\n" + "="*60)
    print("   TRAINING XGBOOST MODEL")
    print("="*60)

    # Tune if requested
    if tune and HAS_OPTUNA:
        params = tune_xgboost(X_train, y_train)
    elif params is None:
        params = XGBOOST_DEFAULT_PARAMS.copy()
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        params['scale_pos_weight'] = n_neg / max(n_pos, 1)

    # Apply SMOTE to full training set
    if HAS_IMBLEARN and y_train.sum() > 5:
        try:
            smote = SMOTE(random_state=42,
                          k_neighbors=min(5, int(y_train.sum()) - 1))
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            print(f"   SMOTE: {len(X_train):,} → {len(X_train_res):,} rows")
        except Exception as e:
            print(f"   SMOTE failed ({e}), using original data")
            X_train_res, y_train_res = X_train, y_train
    else:
        X_train_res, y_train_res = X_train, y_train

    model = XGBClassifier(**params)
    model.fit(X_train_res, y_train_res, verbose=False)

    # Cross-validation metrics
    print("\n   Cross-Validation Results:")
    cv_metrics = evaluate_cv(
        XGBClassifier(**params), X_train, y_train
    )
    for metric, val in cv_metrics.items():
        if not metric.endswith('_mean'):
            print(f"      {metric}: {val}")

    # Save model
    os.makedirs(PATH_MODEL_OUTPUT, exist_ok=True)
    model_path = os.path.join(PATH_MODEL_OUTPUT, 'xgboost_model.joblib')
    joblib.dump(model, model_path)
    print(f"\n   ✅ Model saved: {model_path}")

    # Save params
    params_path = os.path.join(PATH_MODEL_OUTPUT, 'model_params.json')
    with open(params_path, 'w') as f:
        json.dump({k: str(v) if not isinstance(v, (int, float, bool)) else v
                    for k, v in params.items()}, f, indent=2)

    return model, cv_metrics


# =============================================================
# 4. BASELINE MODEL (Logistic Regression)
# =============================================================

def train_baseline(X_train, y_train):
    """Train a Logistic Regression baseline for comparison."""
    print("\n   Training Logistic Regression baseline...")

    lr = LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=42
    )
    lr_metrics = evaluate_cv(lr, X_train, y_train)
    print(f"   Baseline AUC: {lr_metrics['AUC']}")

    lr.fit(X_train, y_train)
    lr_path = os.path.join(PATH_MODEL_OUTPUT, 'logistic_baseline.joblib')
    joblib.dump(lr, lr_path)

    return lr, lr_metrics


# =============================================================
# 5. SHAP EXPLAINABILITY
# =============================================================

def compute_shap(model, X_test, feature_names=None):
    """Compute SHAP values for explainability."""
    print("\n" + "="*60)
    print("   COMPUTING SHAP VALUES")
    print("="*60)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Save SHAP values
    shap_df = pd.DataFrame(
        shap_values,
        columns=X_test.columns if feature_names is None else feature_names,
        index=X_test.index
    )
    shap_path = os.path.join(PATH_MODEL_OUTPUT, 'shap_values.parquet')
    shap_df.to_parquet(shap_path)
    print(f"   ✅ SHAP values saved: {shap_path}")

    # Top 3 drivers per employee
    top3_drivers = []
    for idx in range(len(shap_df)):
        row = shap_df.iloc[idx]
        top_features = row.abs().nlargest(3).index.tolist()
        top_values = [row[f] for f in top_features]
        top3_drivers.append({
            'SHAP_Top1_Feature': top_features[0] if len(top_features) > 0 else '',
            'SHAP_Top1_Value': top_values[0] if len(top_values) > 0 else 0,
            'SHAP_Top2_Feature': top_features[1] if len(top_features) > 1 else '',
            'SHAP_Top2_Value': top_values[1] if len(top_values) > 1 else 0,
            'SHAP_Top3_Feature': top_features[2] if len(top_features) > 2 else '',
            'SHAP_Top3_Value': top_values[2] if len(top_values) > 2 else 0,
        })
    top3_df = pd.DataFrame(top3_drivers, index=X_test.index)

    # Global feature importance
    mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
    importance_df = pd.DataFrame({
        'Feature': mean_abs_shap.index,
        'Mean_Abs_SHAP': mean_abs_shap.values
    })
    imp_path = os.path.join(PATH_MODEL_OUTPUT, 'shap_feature_importance.csv')
    importance_df.to_csv(imp_path, index=False)
    print(f"   ✅ Feature importance saved: {imp_path}")
    print(f"\n   Top 10 features by SHAP:")
    for _, row in importance_df.head(10).iterrows():
        print(f"      {row['Feature']}: {row['Mean_Abs_SHAP']:.4f}")

    return shap_values, shap_df, top3_df, importance_df


# =============================================================
# 6. SURVIVAL ANALYSIS
# =============================================================

def run_survival_analysis(df):
    """Run Cox PH survival analysis and Kaplan-Meier estimation."""
    if not HAS_LIFELINES:
        print("   Survival analysis skipped (lifelines not installed).")
        return None, None

    print("\n" + "="*60)
    print("   SURVIVAL ANALYSIS")
    print("="*60)

    # Prepare survival data
    surv_df = df.copy()

    # Duration: Tenure in months at snapshot
    if SURVIVAL_DURATION_COL not in surv_df.columns:
        print(f"   ⚠️ Column '{SURVIVAL_DURATION_COL}' not found. Skipping survival analysis.")
        return None, None

    surv_df['duration'] = surv_df[SURVIVAL_DURATION_COL].clip(lower=0.1)
    surv_df['event'] = surv_df[TARGET_COLUMN]

    # De-duplicate per employee (latest snapshot)
    surv_unique = surv_df.sort_values('Snapshot_Date', ascending=False).drop_duplicates(
        subset='Employee_ID', keep='first'
    )

    # Select numeric features for Cox PH
    cox_features = [f for f in BASE_FEATURES if f in surv_unique.columns]
    cox_data = surv_unique[['duration', 'event'] + cox_features].copy()
    cox_data = cox_data.fillna(0).replace([np.inf, -np.inf], 0)

    # Remove zero-variance columns
    cox_data = cox_data.loc[:, cox_data.std() > 0]
    if 'duration' not in cox_data.columns or 'event' not in cox_data.columns:
        print("   ⚠️ Insufficient data for survival analysis.")
        return None, None

    # Fit Cox PH
    cph = CoxPHFitter(penalizer=0.1)
    try:
        cph.fit(cox_data, duration_col='duration', event_col='event')
        print("   ✅ Cox PH model fitted.")
        cph_summary = cph.summary
        cph_path = os.path.join(PATH_MODEL_OUTPUT, 'cox_ph_summary.csv')
        cph_summary.to_csv(cph_path)
        print(f"   Summary saved: {cph_path}")
    except Exception as e:
        print(f"   ⚠️ Cox PH fitting failed: {e}")
        cph = None

    # Kaplan-Meier by Management Level
    kmf = KaplanMeierFitter()
    km_results = {}
    if 'Management_Level_Agg' in surv_unique.columns:
        for level in surv_unique['Management_Level_Agg'].unique():
            mask = surv_unique['Management_Level_Agg'] == level
            level_data = surv_unique[mask]
            if len(level_data) > 10:
                kmf.fit(
                    durations=level_data[SURVIVAL_DURATION_COL].clip(lower=0.1),
                    event_observed=level_data[TARGET_COLUMN],
                    label=level
                )
                km_results[level] = kmf.survival_function_.copy()

    if km_results:
        km_combined = pd.concat(km_results, axis=1)
        km_path = os.path.join(PATH_MODEL_OUTPUT, 'kaplan_meier_by_level.csv')
        km_combined.to_csv(km_path)
        print(f"   ✅ Kaplan-Meier curves saved: {km_path}")

    return cph, km_results


# =============================================================
# 7. FAIRNESS AUDIT
# =============================================================

def audit_fairness(df, predictions, probabilities):
    """
    Audit model predictions for bias across protected groups.
    Returns a fairness report DataFrame.
    """
    print("\n" + "="*60)
    print("   FAIRNESS AUDIT")
    print("="*60)

    results = []
    audit_cols = [c for c in PROTECTED_COLUMNS if c in df.columns]

    if not audit_cols:
        print("   No protected columns found in data. Skipping fairness audit.")
        return pd.DataFrame()

    for col in audit_cols:
        if df[col].nunique() > 20 or df[col].nunique() < 2:
            continue  # Skip high-cardinality or constant columns

        groups = df[col].fillna('Unknown').astype(str)
        unique_groups = groups.unique()

        for group in unique_groups:
            mask = groups == group
            n_total = mask.sum()
            if n_total < 10:
                continue

            n_flagged = predictions[mask].sum()
            flag_rate = n_flagged / n_total
            avg_risk = probabilities[mask].mean()

            # If we have actuals
            if TARGET_COLUMN in df.columns:
                actuals = df[TARGET_COLUMN][mask]
                actual_rate = actuals.mean()
                tp = ((predictions[mask] == 1) & (actuals == 1)).sum()
                fp = ((predictions[mask] == 1) & (actuals == 0)).sum()
                fn = ((predictions[mask] == 0) & (actuals == 1)).sum()
                fpr = fp / max(fp + (n_total - n_flagged - fn), 1)
            else:
                actual_rate = np.nan
                fpr = np.nan

            results.append({
                'Protected_Attribute': col,
                'Group': group,
                'N': n_total,
                'Flag_Rate': round(flag_rate, 4),
                'Avg_Risk_Score': round(avg_risk, 4),
                'Actual_Target_Rate': round(actual_rate, 4) if not np.isnan(actual_rate) else None,
                'False_Positive_Rate': round(fpr, 4) if not np.isnan(fpr) else None,
            })

    fairness_df = pd.DataFrame(results)

    if not fairness_df.empty:
        # Check for bias: flag if any group has >20% higher flag rate than overall
        overall_flag_rate = predictions.mean()
        fairness_df['Rate_vs_Overall'] = (
            fairness_df['Flag_Rate'] / max(overall_flag_rate, 0.001)
        ).round(3)
        bias_flags = fairness_df[fairness_df['Rate_vs_Overall'] > 1.20]
        if not bias_flags.empty:
            print("   ⚠️  POTENTIAL BIAS DETECTED:")
            for _, row in bias_flags.iterrows():
                print(f"      {row['Protected_Attribute']}={row['Group']}: "
                      f"Flag rate {row['Flag_Rate']:.1%} vs overall {overall_flag_rate:.1%} "
                      f"({row['Rate_vs_Overall']:.1f}x)")
        else:
            print("   ✅ No significant bias detected across protected groups.")

        fair_path = os.path.join(PATH_MODEL_OUTPUT, 'fairness_audit.csv')
        fairness_df.to_csv(fair_path, index=False)
        print(f"   Report saved: {fair_path}")

    return fairness_df


# =============================================================
# 8. FULL PIPELINE: PREDICT
# =============================================================

def predict(model, X_test, threshold=0.5):
    """Generate predictions and probabilities."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > threshold).astype(int)
    return y_pred, y_prob


def evaluate_test(y_test, y_pred, y_prob):
    """Evaluate model on test set."""
    print("\n" + "="*60)
    print("   TEST SET EVALUATION")
    print("="*60)

    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = 0

    recall = recall_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"   AUC:       {auc:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   F1:        {f1:.4f}")
    print(f"\n   Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"   {cm}")
    print(f"\n   Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return {'AUC': auc, 'Recall': recall, 'Precision': precision, 'F1': f1, 'CM': cm}


# =============================================================
# 9. RUN FULL MODEL PIPELINE
# =============================================================

def run_pipeline(X_train, y_train, X_test, y_test, test_df,
                 feature_names, full_df=None, tune=True, threshold=0.5):
    """
    End-to-end model pipeline:
      1. Train XGBoost (with optional tuning)
      2. Train Logistic Regression baseline
      3. Evaluate on test set
      4. Compute SHAP values
      5. Run survival analysis
      6. Run fairness audit
      7. Save all outputs

    Returns dict with model, predictions, SHAP, survival, fairness results.
    """
    results = {}

    # --- 1. Train Models ---
    xgb_model, xgb_cv_metrics = train_xgboost(X_train, y_train, tune=tune)
    lr_model, lr_cv_metrics = train_baseline(X_train, y_train)

    results['xgb_model'] = xgb_model
    results['lr_model'] = lr_model
    results['xgb_cv_metrics'] = xgb_cv_metrics
    results['lr_cv_metrics'] = lr_cv_metrics

    # --- 2. Predict ---
    y_pred, y_prob = predict(xgb_model, X_test, threshold)
    test_metrics = evaluate_test(y_test, y_pred, y_prob)
    results['test_metrics'] = test_metrics
    results['y_pred'] = y_pred
    results['y_prob'] = y_prob

    # --- 3. SHAP ---
    try:
        shap_values, shap_df, top3_df, importance_df = compute_shap(
            xgb_model, X_test, feature_names
        )
        results['shap_values'] = shap_values
        results['shap_df'] = shap_df
        results['top3_df'] = top3_df
        results['importance_df'] = importance_df
    except Exception as e:
        print(f"   ⚠️ SHAP computation failed: {e}")
        results['shap_values'] = None

    # --- 4. Survival Analysis ---
    if full_df is not None:
        cph, km_results = run_survival_analysis(full_df)
        results['cph'] = cph
        results['km_results'] = km_results
    else:
        results['cph'] = None
        results['km_results'] = None

    # --- 5. Fairness Audit ---
    if full_df is not None:
        # Align predictions with full test DataFrame
        test_with_preds = test_df.copy()
        test_with_preds['Predicted_Risk'] = y_prob
        test_with_preds['Predicted_Leaver'] = y_pred

        fairness_df = audit_fairness(
            test_with_preds,
            pd.Series(y_pred, index=test_df.index),
            pd.Series(y_prob, index=test_df.index)
        )
        results['fairness_df'] = fairness_df
    else:
        results['fairness_df'] = pd.DataFrame()

    # --- 6. Save Predictions ---
    pred_df = test_df[['Employee_ID', 'Snapshot_Date', 'Year']].copy()
    pred_df['Predicted_Risk'] = y_prob
    pred_df['Predicted_Leaver'] = y_pred
    if results.get('top3_df') is not None:
        pred_df = pd.concat([pred_df.reset_index(drop=True),
                             results['top3_df'].reset_index(drop=True)], axis=1)

    pred_path = os.path.join(PATH_MODEL_OUTPUT, 'predictions.parquet')
    pred_df.to_parquet(pred_path, index=False)
    print(f"\n   ✅ Predictions saved: {pred_path}")

    # --- 7. Model comparison summary ---
    print("\n" + "="*60)
    print("   MODEL COMPARISON SUMMARY")
    print("="*60)
    print(f"   {'Metric':<20} {'XGBoost':<25} {'Logistic Regression':<25}")
    print(f"   {'-'*70}")
    for metric in ['AUC', 'Recall', 'Precision', 'F1']:
        xgb_val = xgb_cv_metrics.get(metric, 'N/A')
        lr_val = lr_cv_metrics.get(metric, 'N/A')
        print(f"   {metric:<20} {xgb_val:<25} {lr_val:<25}")

    return results


# =============================================================
# LOAD SAVED MODEL
# =============================================================

def load_model():
    """Load a previously trained model."""
    model_path = os.path.join(PATH_MODEL_OUTPUT, 'xgboost_model.joblib')
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"Loaded model: {model_path}")
        return model
    else:
        print(f"No saved model found at {model_path}")
        return None


# =============================================================
# CLI ENTRY POINT
# =============================================================

if __name__ == '__main__':
    from data_loader import load_master
    from feature_engine import build_features, prepare_model_data

    df = load_master()
    df = build_features(df)
    X_train, y_train, X_test, y_test, test_df, feat_names = prepare_model_data(df)

    results = run_pipeline(
        X_train, y_train, X_test, y_test, test_df,
        feat_names, full_df=df, tune=True
    )
    print("\n🎉 Model pipeline complete!")
