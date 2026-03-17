"""
Feature Engine — All Feature Engineering Logic
================================================
Replaces Cell 2 (ModelDataBuilder) of the original notebook.
Builds ~30 features across 6 categories:
  1. Tenure & Career
  2. Performance
  3. Peer dynamics
  4. Manager intelligence
  5. Org-level risk
  6. Employment context & temporal
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import joblib
import os

from config import (
    MANAGEMENT_LEVEL_MAP, LEVEL_BENCHMARKS, PERFORMANCE_MAP,
    BASE_FEATURES, CATEGORICAL_FEATURES, ID_COLUMNS, TARGET_COLUMN,
    PATH_MODEL_OUTPUT, PATH_OUTPUT,
)


# =============================================================
# MAIN: BUILD ALL FEATURES
# =============================================================

def build_features(df):
    """
    Takes the master active headcount DataFrame and engineers all features.
    Returns (df_featured, feature_names, scaler).
    """
    print("\n" + "="*60)
    print("   FEATURE ENGINEERING")
    print("="*60)

    df = df.copy()
    df = df.sort_values(by=['Employee_ID', 'Snapshot_Date'])
    df['Snapshot_Date'] = pd.to_datetime(df['Snapshot_Date'])
    df['Year'] = df['Snapshot_Date'].dt.year
    df['Month'] = df['Snapshot_Date'].dt.month

    # Ensure Management_Level_Agg exists
    if 'Management_Level_Agg' not in df.columns:
        if 'Management Level' in df.columns:
            df['Management_Level_Agg'] = (
                df['Management Level'].astype(str).str.strip()
                .map(MANAGEMENT_LEVEL_MAP).fillna('Unmapped')
            )
        else:
            df['Management_Level_Agg'] = 'Unmapped'

    # ----------------------------------------------------------
    # 1. TENURE & CAREER FEATURES
    # ----------------------------------------------------------
    print("   [1/6] Tenure & Career features...")

    # A. Total Tenure
    if 'Length of Service' in df.columns:
        df['Total_Tenure_Years'] = pd.to_numeric(
            df['Length of Service'], errors='coerce'
        )
        df['Total_Tenure_Years'] = df['Total_Tenure_Years'].fillna(
            df['Total_Tenure_Years'].median()
        )
    else:
        df['Total_Tenure_Years'] = 0

    # B. Time in Title (direct from data if available)
    if 'Time In Title (Current)' in df.columns:
        df['Time_In_Title_Current'] = pd.to_numeric(
            df['Time In Title (Current)'], errors='coerce'
        ).fillna(0)
    else:
        df['Time_In_Title_Current'] = 0

    # C. Cycles in Level (standardized by benchmark)
    df['First_Date_In_Level'] = df.groupby(
        ['Employee_ID', 'Management_Level_Agg']
    )['Snapshot_Date'].transform('min')
    df['Tenure_Days_In_Level'] = (
        df['Snapshot_Date'] - df['First_Date_In_Level']
    ).dt.days
    df['Years_In_Level'] = df['Tenure_Days_In_Level'] / 365.25
    df['Expected_Years'] = (
        df['Management_Level_Agg'].map(LEVEL_BENCHMARKS).fillna(3.0)
    )
    df['Cycles_In_Level'] = df['Years_In_Level'] / df['Expected_Years']
    df['Is_Stuck_In_Level'] = (df['Cycles_In_Level'] >= 1.5).astype(int)

    # D. Days since last promotion
    if 'Is_Promoted' in df.columns:
        promo_rows = df[df['Is_Promoted'] == 1][['Employee_ID', 'Snapshot_Date']].copy()
        promo_rows = promo_rows.rename(columns={'Snapshot_Date': 'Last_Promo_Date'})
        promo_rows = promo_rows.sort_values('Last_Promo_Date').drop_duplicates(
            subset='Employee_ID', keep='last'
        )
        df = pd.merge(df, promo_rows, on='Employee_ID', how='left')
        df['Days_Since_Last_Promotion'] = (
            (df['Snapshot_Date'] - df['Last_Promo_Date']).dt.days
        ).fillna(-1)  # -1 means never promoted
        df.drop(columns=['Last_Promo_Date'], inplace=True, errors='ignore')
    else:
        df['Days_Since_Last_Promotion'] = -1

    # ----------------------------------------------------------
    # 2. PERFORMANCE FEATURES
    # ----------------------------------------------------------
    print("   [2/6] Performance features...")

    # Ensure Performance_Score exists
    if 'Performance_Score' not in df.columns:
        df['Performance_Score'] = np.nan

    df['Is_High_Performer'] = df['Performance_Score'].isin([4, 5]).astype(int)

    # A. Performance Drop Flag (YoY)
    df['Year_Minus_1'] = df['Year'] - 1
    perf_ref = df[['Employee_ID', 'Year', 'Performance_Score']].drop_duplicates(
        subset=['Employee_ID', 'Year']
    )
    perf_ref.columns = ['Employee_ID', 'Year_Minus_1', 'Prev_Year_Performance']
    df = pd.merge(df, perf_ref, on=['Employee_ID', 'Year_Minus_1'], how='left')
    df['Performance_Drop_Flag'] = (
        (df['Prev_Year_Performance'].notna()) &
        (df['Performance_Score'] < df['Prev_Year_Performance'])
    ).astype(int)
    df.drop(columns=['Year_Minus_1', 'Prev_Year_Performance'],
            inplace=True, errors='ignore')

    # B. Rolling 3-year average performance
    yearly_perf = df.drop_duplicates(
        subset=['Employee_ID', 'Year']
    )[['Employee_ID', 'Year', 'Performance_Score']].copy()
    yearly_perf = yearly_perf.sort_values(['Employee_ID', 'Year'])
    yearly_perf['Perf_Score_Rolling_Avg_3Yr'] = (
        yearly_perf.groupby('Employee_ID')['Performance_Score']
        .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    )
    df = pd.merge(
        df,
        yearly_perf[['Employee_ID', 'Year', 'Perf_Score_Rolling_Avg_3Yr']],
        on=['Employee_ID', 'Year'], how='left'
    )

    # C. Performance trajectory slope (linear trend)
    def calc_slope(series):
        """OLS slope of performance over time indices."""
        valid = series.dropna()
        if len(valid) < 2:
            return 0.0
        x = np.arange(len(valid), dtype=float)
        y = valid.values.astype(float)
        # Simple linear regression
        x_mean = x.mean()
        y_mean = y.mean()
        denom = ((x - x_mean) ** 2).sum()
        if denom == 0:
            return 0.0
        return ((x - x_mean) * (y - y_mean)).sum() / denom

    yearly_perf_slope = df.drop_duplicates(
        subset=['Employee_ID', 'Year']
    )[['Employee_ID', 'Year', 'Performance_Score']].copy()
    yearly_perf_slope = yearly_perf_slope.sort_values(['Employee_ID', 'Year'])
    yearly_perf_slope['Perf_Trajectory_Slope'] = (
        yearly_perf_slope.groupby('Employee_ID')['Performance_Score']
        .transform(lambda x: x.expanding(min_periods=2).apply(calc_slope, raw=False))
    )
    df = pd.merge(
        df,
        yearly_perf_slope[['Employee_ID', 'Year', 'Perf_Trajectory_Slope']],
        on=['Employee_ID', 'Year'], how='left'
    )
    df['Perf_Trajectory_Slope'] = df['Perf_Trajectory_Slope'].fillna(0)

    # D. Consecutive low-performance years
    yearly_low = df.drop_duplicates(
        subset=['Employee_ID', 'Year']
    )[['Employee_ID', 'Year', 'Performance_Score']].copy()
    yearly_low = yearly_low.sort_values(['Employee_ID', 'Year'])
    yearly_low['Is_Low'] = (yearly_low['Performance_Score'] <= 2).astype(int)
    yearly_low['Consecutive_Low_Perf_Years'] = (
        yearly_low.groupby('Employee_ID')['Is_Low']
        .transform(lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1))
    )
    df = pd.merge(
        df,
        yearly_low[['Employee_ID', 'Year', 'Consecutive_Low_Perf_Years']],
        on=['Employee_ID', 'Year'], how='left'
    )
    df['Consecutive_Low_Perf_Years'] = df['Consecutive_Low_Perf_Years'].fillna(0)

    # ----------------------------------------------------------
    # 3. PEER DYNAMICS
    # ----------------------------------------------------------
    print("   [3/6] Peer dynamics features...")

    if 'Level 4 Org' not in df.columns:
        df['Level 4 Org'] = 'Unknown'

    if 'Is_Promoted' not in df.columns:
        df['Is_Promoted'] = 0

    promo_counts = df.groupby(
        ['Year', 'Management_Level_Agg', 'Level 4 Org']
    )['Is_Promoted'].transform('sum')
    headcount_counts = df.groupby(
        ['Year', 'Management_Level_Agg', 'Level 4 Org']
    )['Employee_ID'].transform('count')
    df['Peer_Promo_Rate'] = promo_counts / headcount_counts.replace(0, 1)
    df['Peer_Pressure_Flag'] = (
        (df['Is_Promoted'] == 0) & (df['Peer_Promo_Rate'] > 0.10)
    ).astype(int)

    # ----------------------------------------------------------
    # 4. MANAGER INTELLIGENCE
    # ----------------------------------------------------------
    print("   [4/6] Manager intelligence features...")

    if 'Manager ID' in df.columns:
        df['Manager ID'] = df['Manager ID'].astype(str).str.strip().str.replace(
            r'\.0$', '', regex=True
        )
        df['Employee_ID'] = df['Employee_ID'].astype(str).str.strip()

        # A. Manager span of control
        if 'Active Manager Direct EMP+CWK Reports' in df.columns:
            df['Manager_Span_Of_Control'] = pd.to_numeric(
                df['Active Manager Direct EMP+CWK Reports'], errors='coerce'
            ).fillna(0)
        else:
            # Estimate from data: count direct reports per manager per snapshot
            mgr_span = df.groupby(
                ['Manager ID', 'Snapshot_Date']
            )['Employee_ID'].transform('count')
            df['Manager_Span_Of_Control'] = mgr_span

        # B. Manager churn history (lagged by 1 year to avoid leakage)
        # Only count VOLUNTARY leavers under each manager
        if 'HC_LVO_EE' in df.columns:
            # We need to look at ALL rows (including leavers) for historical counts
            # But our df only has active rows. We use yearly churn counts if available.
            # Approximate: count employees who left in prior year per manager
            df['Manager_Past_Churn_Count'] = 0  # default

            # Build from promotion/performance merge pattern:
            # managers whose reports left in Year-1
            # Since we filtered out leaver rows, we estimate from the data structure
            # This will be refined when full pipeline runs on unfiltered data

        # C. Manager performance (lookup manager's own score)
        mgr_perf = df[['Employee_ID', 'Snapshot_Date', 'Performance_Score']].copy()
        mgr_perf.columns = ['Manager ID', 'Snapshot_Date', 'Manager_Performance_Score']
        mgr_perf['Manager ID'] = mgr_perf['Manager ID'].astype(str).str.strip()
        mgr_perf = mgr_perf.drop_duplicates(subset=['Manager ID', 'Snapshot_Date'])
        try:
            df = pd.merge(df, mgr_perf, on=['Manager ID', 'Snapshot_Date'], how='left')
        except Exception:
            df['Manager_Performance_Score'] = np.nan
        df['Manager_Performance_Score'] = df['Manager_Performance_Score'].fillna(3)

        # D. Employee vs Manager performance ratio
        df['Performer_vs_Mgr_Score'] = (
            df['Performance_Score'] / df['Manager_Performance_Score'].replace(0, 1)
        )
        df.loc[df['Performance_Score'].isna(), 'Performer_vs_Mgr_Score'] = np.nan

        # E. Manager stability (6-month rolling window)
        df['Prev_Manager_ID'] = df.groupby('Employee_ID')['Manager ID'].shift(1)
        df['Mgr_Change_Event'] = (
            (df['Prev_Manager_ID'].notna()) &
            (df['Manager ID'] != df['Prev_Manager_ID'])
        ).astype(int)
        RISK_WINDOW = 6
        df['Manager_Changed_Recently'] = (
            df.groupby('Employee_ID')['Mgr_Change_Event']
            .transform(lambda x: x.rolling(window=RISK_WINDOW, min_periods=1).max())
        ).fillna(0).astype(int)
        df.drop(columns=['Prev_Manager_ID', 'Mgr_Change_Event'],
                inplace=True, errors='ignore')
    else:
        print("   WARNING: 'Manager ID' column not found. Using defaults.")
        df['Manager_Past_Churn_Count'] = 0
        df['Manager_Performance_Score'] = 3
        df['Performer_vs_Mgr_Score'] = np.nan
        df['Manager_Changed_Recently'] = 0
        df['Manager_Span_Of_Control'] = 0

    # ----------------------------------------------------------
    # 5. ORG-LEVEL RISK FEATURES
    # ----------------------------------------------------------
    print("   [5/6] Org-level risk features...")

    # A. Job family churn rate (trailing — uses target from prior year)
    if 'Job Family' in df.columns:
        jf_churn = df.groupby(['Year', 'Job Family'])[TARGET_COLUMN].transform('mean')
        # Lag by 1 year to avoid leakage
        jf_lookup = df.drop_duplicates(
            subset=['Year', 'Job Family']
        )[['Year', 'Job Family', TARGET_COLUMN]].copy()
        jf_lookup.columns = ['Prev_Year_JF', 'Job Family', 'Job_Family_Churn_Rate']
        jf_lookup['Prev_Year_JF'] = jf_lookup['Prev_Year_JF'] + 1  # lag
        df['Year_for_JF'] = df['Year']
        df = pd.merge(
            df, jf_lookup,
            left_on=['Year_for_JF', 'Job Family'],
            right_on=['Prev_Year_JF', 'Job Family'],
            how='left'
        )
        df['Job_Family_Churn_Rate'] = df['Job_Family_Churn_Rate'].fillna(0)
        df.drop(columns=['Year_for_JF', 'Prev_Year_JF'],
                inplace=True, errors='ignore')
    else:
        df['Job_Family_Churn_Rate'] = 0

    # B. Org-level churn rate (Level 4 Org, lagged)
    org_churn = df.drop_duplicates(
        subset=['Year', 'Level 4 Org']
    )[['Year', 'Level 4 Org', TARGET_COLUMN]].copy()
    org_churn.columns = ['Prev_Year_Org', 'Level 4 Org', 'Org_Level_Churn_Rate']
    org_churn['Prev_Year_Org'] = org_churn['Prev_Year_Org'] + 1
    df = pd.merge(
        df, org_churn,
        left_on=['Year', 'Level 4 Org'],
        right_on=['Prev_Year_Org', 'Level 4 Org'],
        how='left'
    )
    df['Org_Level_Churn_Rate'] = df['Org_Level_Churn_Rate'].fillna(0)
    df.drop(columns=['Prev_Year_Org'], inplace=True, errors='ignore')

    # ----------------------------------------------------------
    # 6. EMPLOYMENT CONTEXT & TEMPORAL
    # ----------------------------------------------------------
    print("   [6/6] Employment context & temporal features...")

    # A. Hire source flags
    if 'HC_Hire_Lateral_Comm' in df.columns:
        df['Is_Lateral_Hire'] = pd.to_numeric(
            df['HC_Hire_Lateral_Comm'], errors='coerce'
        ).fillna(0).abs().clip(0, 1).astype(int)
    else:
        df['Is_Lateral_Hire'] = 0

    if 'HC_Hire_Campus_Comm' in df.columns:
        df['Is_Campus_Hire'] = pd.to_numeric(
            df['HC_Hire_Campus_Comm'], errors='coerce'
        ).fillna(0).abs().clip(0, 1).astype(int)
    else:
        df['Is_Campus_Hire'] = 0

    # B. International assignment
    if 'International Assignment Flag' in df.columns:
        df['Has_International_Assignment'] = (
            df['International Assignment Flag'].astype(str).str.strip()
            .str.lower().isin(['yes', 'y', '1', 'true'])
        ).astype(int)
    else:
        df['Has_International_Assignment'] = 0

    # C. Acquisition employee
    if 'Acquisition Company' in df.columns:
        df['Is_Acquisition_Employee'] = (
            df['Acquisition Company'].notna() &
            (df['Acquisition Company'].astype(str).str.strip() != '') &
            (df['Acquisition Company'].astype(str).str.strip().str.lower() != 'nan')
        ).astype(int)
    else:
        df['Is_Acquisition_Employee'] = 0

    # D. FTE
    if 'FTE' in df.columns:
        df['FTE_Value'] = pd.to_numeric(df['FTE'], errors='coerce').fillna(1.0)
    else:
        df['FTE_Value'] = 1.0

    # E. Reorg affected (trailing 6 months)
    for reorg_col in ['HC_RGI_EE', 'HC_RGO_EE']:
        if reorg_col not in df.columns:
            df[reorg_col] = 0
    df['_reorg_event'] = ((df['HC_RGI_EE'] == 1) | (df['HC_RGO_EE'] == 1)).astype(int)
    df['Is_Reorg_Affected'] = (
        df.groupby('Employee_ID')['_reorg_event']
        .transform(lambda x: x.rolling(window=6, min_periods=1).max())
    ).fillna(0).astype(int)
    df.drop(columns=['_reorg_event'], inplace=True, errors='ignore')

    # F. Transfer recent (trailing 6 months)
    for xfer_col in ['HC_XIN_EE', 'HC_XOT_EE']:
        if xfer_col not in df.columns:
            df[xfer_col] = 0
    df['_xfer_event'] = ((df['HC_XIN_EE'] == 1) | (df['HC_XOT_EE'] == 1)).astype(int)
    df['Is_Transfer_Recent'] = (
        df.groupby('Employee_ID')['_xfer_event']
        .transform(lambda x: x.rolling(window=6, min_periods=1).max())
    ).fillna(0).astype(int)
    df.drop(columns=['_xfer_event'], inplace=True, errors='ignore')

    # G. Temporal / Seasonal features
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['Is_Post_Bonus_Window'] = df['Month'].isin([2, 3, 4]).astype(int)

    # H. Tenure for survival analysis
    if 'Length of Service' in df.columns:
        df['Tenure_Months_At_Snapshot'] = (
            pd.to_numeric(df['Length of Service'], errors='coerce') * 12
        ).fillna(0)
    else:
        df['Tenure_Months_At_Snapshot'] = 0

    # ----------------------------------------------------------
    # FILL NaN for all feature columns
    # ----------------------------------------------------------
    print("\n   Filling NaN values in feature columns...")
    available_features = [f for f in BASE_FEATURES if f in df.columns]
    missing_features = [f for f in BASE_FEATURES if f not in df.columns]

    if missing_features:
        print(f"   ⚠️  Features not created (missing source data): {missing_features}")
        for mf in missing_features:
            df[mf] = 0

    for feat in BASE_FEATURES:
        if df[feat].dtype in ['float64', 'float32', 'int64', 'int32']:
            df[feat] = df[feat].fillna(0)

    # ----------------------------------------------------------
    # SAVE FEATURED DATA
    # ----------------------------------------------------------
    print(f"\n   Features built: {len(BASE_FEATURES)} base + {len(CATEGORICAL_FEATURES)} categorical")
    print(f"   Total rows: {len(df):,}")

    try:
        feat_path = os.path.join(PATH_OUTPUT, 'Model_Ready_Data.parquet')
        df.to_parquet(feat_path, index=False)
        print(f"   ✅ Saved featured data: {feat_path}")
    except Exception as e:
        print(f"   Parquet error: {e}. Saving CSV...")
        feat_path = os.path.join(PATH_OUTPUT, 'Model_Ready_Data.csv')
        df.to_csv(feat_path, index=False)
        print(f"   ✅ Saved featured data: {feat_path}")

    return df


# =============================================================
# SCALE FEATURES (for model training)
# =============================================================

def scale_features(X_train, X_test=None):
    """
    Apply RobustScaler to numeric features.
    Saves the scaler for dashboard prediction use.
    Returns (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = None
    if X_test is not None:
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

    # Save scaler
    os.makedirs(PATH_MODEL_OUTPUT, exist_ok=True)
    scaler_path = os.path.join(PATH_MODEL_OUTPUT, 'feature_scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"   Scaler saved: {scaler_path}")

    return X_train_scaled, X_test_scaled, scaler


# =============================================================
# PREPARE MODEL MATRICES
# =============================================================

def prepare_model_data(df, prediction_year=None):
    """
    Prepare X_train, y_train, X_test, y_test from the featured DataFrame.
    If prediction_year is None, auto-selects the latest year with positive targets.

    Returns (X_train, y_train, X_test, y_test, test_df, feature_names, train_employee_ids)
    Note: train_employee_ids is for GroupKFold cross-validation.
    """
    if prediction_year is None:
        # Auto-select latest year with positive targets
        # (Current year like 2026 may have 0 positives since the future hasn't happened)
        years_with_pos = (
            df.groupby('Year')[TARGET_COLUMN].sum()
        )
        years_with_targets = years_with_pos[years_with_pos > 0]
        if not years_with_targets.empty:
            prediction_year = int(years_with_targets.index.max())
            print(f"\n   Auto-selected prediction year: {prediction_year} "
                  f"(latest with {int(years_with_targets[prediction_year])} positive targets)")
        else:
            prediction_year = df['Year'].max()
            print(f"\n   WARNING: No year has positive targets. Using latest: {prediction_year}")

    print(f"   Preparing model data for prediction year: {prediction_year}")

    # One-hot encode categorical features
    all_features = BASE_FEATURES + CATEGORICAL_FEATURES
    available = [f for f in all_features if f in df.columns]

    X_all = pd.get_dummies(
        df[available], columns=CATEGORICAL_FEATURES, drop_first=True
    )
    y_all = df[TARGET_COLUMN]

    # Temporal split: train on historical, test on prediction year
    train_mask = df['Year'] < prediction_year
    X_train = X_all[train_mask].fillna(0)
    y_train = y_all[train_mask].fillna(0)

    # Extract employee IDs for GroupKFold (aligned with X_train)
    train_employee_ids = df.loc[train_mask, 'Employee_ID']

    # Test set: de-duplicated per employee (latest snapshot in prediction year)
    test_raw = df[df['Year'] == prediction_year].sort_values(
        ['Employee_ID', 'Snapshot_Date'], ascending=[True, False]
    )
    test_unique = test_raw.drop_duplicates(subset=['Employee_ID'], keep='first')

    X_test = pd.get_dummies(
        test_unique[available], columns=CATEGORICAL_FEATURES, drop_first=True
    )
    y_test = test_unique[TARGET_COLUMN]

    # Align columns
    for c in X_train.columns:
        if c not in X_test.columns:
            X_test[c] = 0
    X_test = X_test[X_train.columns].fillna(0)

    feature_names = X_train.columns.tolist()

    print(f"   Train: {len(X_train):,} rows ({y_train.sum():,} positive)")
    print(f"   Test:  {len(X_test):,} rows ({y_test.sum():,} positive)")
    print(f"   Unique employees in train: {train_employee_ids.nunique():,}")

    return X_train, y_train, X_test, y_test, test_unique, feature_names, train_employee_ids


# =============================================================
# CLI ENTRY POINT
# =============================================================

if __name__ == '__main__':
    from data_loader import load_master
    df = load_master()
    df_feat = build_features(df)
    print(f"\nFeatured data shape: {df_feat.shape}")
