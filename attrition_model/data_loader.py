"""
Data Loader — Schema Validation, Master File Builder, Leakage-Safe Target
==========================================================================
Replaces Cell 1 (MasterFileBuilder) of the original notebook.
Key improvements:
  - Voluntary-only target construction
  - Forward-looking target (no same-year leakage)
  - Leakage column exclusion
  - Schema validation & data quality report
  - Modular plug-in pattern for engagement/compensation data
"""

import pandas as pd
import numpy as np
import glob
import os
import sys
from dateutil.relativedelta import relativedelta

from config import (
    PATH_HEADCOUNT, PATH_PERFORMANCE, PATH_PROMOTIONS, PATH_OUTPUT,
    PATH_MODEL_OUTPUT,
    SKIP_HEADCOUNT, SKIP_PERFORMANCE, SKIP_PROMOTIONS,
    PERFORMANCE_MAP, MANAGEMENT_LEVEL_MAP,
    LEAKAGE_COLUMNS, PROTECTED_COLUMNS,
    PREDICTION_HORIZON_MONTHS,
    EXPECTED_HEADCOUNT_COLUMNS,
    TARGET_COLUMN,
)


# =============================================================
# HELPER FUNCTIONS
# =============================================================

def clean_id(val):
    """Normalize employee/manager IDs: strip whitespace, remove trailing .0"""
    s = str(val).strip()
    if s.endswith('.0'):
        s = s[:-2]
    return s


def load_folder_csvs(folder_path, header_row_index):
    """Load and concatenate all CSVs from a folder."""
    print(f"   -> Scanning: {folder_path}")
    files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not files:
        print(f"   -> WARNING: No CSV files found in {folder_path}")
        return pd.DataFrame()
    df_list = []
    for f in files:
        try:
            df = pd.read_csv(f, header=header_row_index, low_memory=False)
            df_list.append(df)
            print(f"      Loaded: {os.path.basename(f)} ({len(df)} rows)")
        except Exception as e:
            print(f"      ERROR reading {os.path.basename(f)}: {e}")
    if not df_list:
        return pd.DataFrame()
    combined = pd.concat(df_list, ignore_index=True)
    print(f"   -> Total rows loaded: {len(combined)}")
    return combined


# =============================================================
# SCHEMA VALIDATION
# =============================================================

def validate_schema(df, expected_cols, source_name):
    """Check that critical columns exist and report any missing."""
    df.columns = df.columns.str.strip()
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        print(f"\n   ⚠️  SCHEMA WARNING [{source_name}]: Missing columns: {missing}")
        print(f"       Available columns ({len(df.columns)}): {df.columns.tolist()[:20]}...")
    else:
        print(f"   ✅ Schema validated [{source_name}]: All {len(expected_cols)} expected columns present")
    return df


# =============================================================
# DATA QUALITY REPORT
# =============================================================

def print_data_quality_report(df, label="Dataset"):
    """Print a summary of the dataset's quality."""
    print(f"\n{'='*60}")
    print(f"   DATA QUALITY REPORT: {label}")
    print(f"{'='*60}")
    print(f"   Total rows:    {len(df):,}")
    print(f"   Total columns: {len(df.columns)}")

    if 'Snapshot_Date' in df.columns:
        print(f"   Date range:    {df['Snapshot_Date'].min()} to {df['Snapshot_Date'].max()}")
    if 'Employee_ID' in df.columns:
        print(f"   Unique employees: {df['Employee_ID'].nunique():,}")
    if 'Year' in df.columns:
        print(f"   Years covered: {sorted(df['Year'].unique())}")

    # Missing value summary (top 10)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(1)
    missing_report = missing_pct[missing_pct > 0].sort_values(ascending=False).head(10)
    if not missing_report.empty:
        print(f"\n   Top missing columns:")
        for col, pct in missing_report.items():
            print(f"      {col}: {pct}%")

    if TARGET_COLUMN in df.columns:
        target_dist = df[TARGET_COLUMN].value_counts()
        print(f"\n   Target distribution:")
        for val, count in target_dist.items():
            print(f"      {val}: {count:,} ({count/len(df)*100:.1f}%)")

    print(f"{'='*60}\n")


# =============================================================
# PLUG-IN DATA SOURCES (Stubs — fill in when data is available)
# =============================================================

def load_engagement_data():
    """
    Stub: Load employee engagement / satisfaction survey data.
    When available, return a DataFrame with columns:
      ['Employee_ID', 'Survey_Date', 'Satisfaction_Score', 'Engagement_Score']
    """
    print("   ℹ️  Engagement data: Not available (stub). Skipping.")
    return pd.DataFrame(columns=['Employee_ID', 'Survey_Date',
                                 'Satisfaction_Score', 'Engagement_Score'])


def load_compensation_data():
    """
    Stub: Load compensation / salary data.
    When available, return a DataFrame with columns:
      ['Employee_ID', 'Year', 'Base_Salary', 'Bonus_Amount', 'Total_Comp']
    """
    print("   ℹ️  Compensation data: Not available (stub). Skipping.")
    return pd.DataFrame(columns=['Employee_ID', 'Year',
                                 'Base_Salary', 'Bonus_Amount', 'Total_Comp'])


# =============================================================
# MAIN: BUILD MASTER FILE
# =============================================================

def build_master():
    """
    Full ETL pipeline: load raw data -> clean -> merge -> build
    leakage-safe target -> save master file.

    Returns the master DataFrame.
    """
    os.makedirs(PATH_OUTPUT, exist_ok=True)
    os.makedirs(PATH_MODEL_OUTPUT, exist_ok=True)

    # ----------------------------------------------------------
    # A. PROCESS HEADCOUNT
    # ----------------------------------------------------------
    print("\n" + "="*60)
    print("   STEP 1: PROCESSING HEADCOUNT")
    print("="*60)

    df_hc = load_folder_csvs(PATH_HEADCOUNT, SKIP_HEADCOUNT)
    if df_hc.empty:
        sys.exit("CRITICAL: Headcount data is empty. Cannot proceed.")

    # Clean headers
    df_hc.columns = df_hc.columns.str.strip()
    df_hc = validate_schema(df_hc, EXPECTED_HEADCOUNT_COLUMNS, "Headcount")

    # Rename key columns
    df_hc.rename(columns={
        'Data As Of': 'Snapshot_Date',
        'Employee ID/Position ID': 'Employee_ID',
    }, inplace=True)

    # Fix Snapshot_Date
    df_hc['Snapshot_Date'] = pd.to_datetime(df_hc['Snapshot_Date'], errors='coerce')
    mask_bad_dates = df_hc['Snapshot_Date'].isna()
    if mask_bad_dates.any():
        raw_vals = df_hc.loc[mask_bad_dates, 'Snapshot_Date']
        df_hc.loc[mask_bad_dates, 'Snapshot_Date'] = pd.to_datetime(
            pd.to_numeric(raw_vals, errors='coerce'), unit='D', origin='1899-12-30'
        )
    df_hc['Snapshot_Date'] = df_hc['Snapshot_Date'].astype('datetime64[ms]')
    df_hc = df_hc.dropna(subset=['Snapshot_Date'])
    df_hc['Year'] = df_hc['Snapshot_Date'].dt.year
    df_hc['Month'] = df_hc['Snapshot_Date'].dt.month

    # Clean Employee_ID
    df_hc['Employee_ID'] = df_hc['Employee_ID'].apply(clean_id)

    # ----------------------------------------------------------
    # B. NORMALIZE FLAGS (handle -1 / 1 issue)
    # ----------------------------------------------------------
    flag_cols = ['HC_LVO_EE', 'HC_LIN_EE', 'HC_LOT_EE', 'HC_CTT_EE',
                 'HC_BSL_EE', 'HC_RGI_EE', 'HC_RGO_EE', 'HC_RTT_EE',
                 'HC_XTT_EE', 'HC_XOT_EE', 'HC_XIN_EE',
                 'HC_HIR_EE', 'HC_HTT_EE']
    for col in flag_cols:
        if col in df_hc.columns:
            df_hc[col] = pd.to_numeric(df_hc[col], errors='coerce').fillna(0).astype(int).abs()

    print(f"   Voluntary leaver rows: {df_hc['HC_LVO_EE'].sum():,}")
    if 'HC_LIN_EE' in df_hc.columns:
        print(f"   Involuntary leaver rows: {df_hc['HC_LIN_EE'].sum():,}")

    # ----------------------------------------------------------
    # C. PROCESS PERFORMANCE
    # ----------------------------------------------------------
    print("\n" + "="*60)
    print("   STEP 2: PROCESSING PERFORMANCE")
    print("="*60)

    df_perf_raw = load_folder_csvs(PATH_PERFORMANCE, SKIP_PERFORMANCE)
    if not df_perf_raw.empty:
        df_perf_raw.columns = df_perf_raw.columns.str.strip()
        df_perf_raw.rename(columns={'Employee ID': 'Employee_ID'}, inplace=True)

        yer_cols = [c for c in df_perf_raw.columns if 'YER Rating' in c]
        if yer_cols:
            df_perf_long = pd.melt(
                df_perf_raw,
                id_vars=['Employee_ID'],
                value_vars=yer_cols,
                var_name='Rating_Source_Col',
                value_name='Performance_Text'
            )
            df_perf_long['Performance_Year'] = (
                df_perf_long['Rating_Source_Col'].str.extract(r'(\d{4})').astype(float)
            )
            df_perf_long['Effective_Year'] = df_perf_long['Performance_Year'] + 1
            df_perf_long['Performance_Score'] = (
                df_perf_long['Performance_Text'].str.strip().map(PERFORMANCE_MAP)
            )
            df_perf_long['Employee_ID'] = df_perf_long['Employee_ID'].apply(clean_id)

            df_hc = pd.merge(
                df_hc,
                df_perf_long[['Employee_ID', 'Effective_Year',
                              'Performance_Text', 'Performance_Score']],
                left_on=['Employee_ID', 'Year'],
                right_on=['Employee_ID', 'Effective_Year'],
                how='left'
            )
            df_hc.drop(columns=['Effective_Year'], inplace=True, errors='ignore')

            # Forward-fill within employee
            df_hc = df_hc.sort_values(by=['Employee_ID', 'Snapshot_Date'])
            df_hc['Performance_Text'] = df_hc.groupby('Employee_ID')['Performance_Text'].ffill()
            df_hc['Performance_Score'] = df_hc.groupby('Employee_ID')['Performance_Score'].ffill()
            print("   -> Performance merged and forward-filled.")
        else:
            print("   -> WARNING: No YER Rating columns found in performance data.")
    else:
        print("   -> No performance data loaded.")
        df_hc['Performance_Text'] = np.nan
        df_hc['Performance_Score'] = np.nan

    # ----------------------------------------------------------
    # D. PROCESS PROMOTIONS
    # ----------------------------------------------------------
    print("\n" + "="*60)
    print("   STEP 3: PROCESSING PROMOTIONS")
    print("="*60)

    df_promo = load_folder_csvs(PATH_PROMOTIONS, SKIP_PROMOTIONS)
    if not df_promo.empty:
        df_promo.columns = df_promo.columns.str.strip()

        # Smart rename for Employee ID
        if 'Employee' in df_promo.columns:
            df_promo.rename(columns={'Employee': 'Employee_ID'}, inplace=True)
        elif 'Employee ID' in df_promo.columns:
            df_promo.rename(columns={'Employee ID': 'Employee_ID'}, inplace=True)

        df_promo.rename(columns={
            'Process Year': 'Process_Year',
            'Nomination Status - After Memo': 'Nomination_Status',
        }, inplace=True)

        required_promo_cols = ['Nomination_Status', 'Process_Year', 'Employee_ID']
        if all(c in df_promo.columns for c in required_promo_cols):
            df_promo['Is_Promoted'] = (
                df_promo['Nomination_Status'].astype(str)
                .str.contains('Nominated', case=False, na=False)
                .astype(int)
            )
            df_promo['Process_Year'] = pd.to_numeric(df_promo['Process_Year'], errors='coerce')
            df_promo['Effective_Year'] = df_promo['Process_Year'] + 1
            df_promo['Employee_ID'] = df_promo['Employee_ID'].apply(clean_id)
            df_promo = df_promo.dropna(subset=['Effective_Year', 'Employee_ID'])

            # Deduplicate
            df_promo_dedup = (
                df_promo.groupby(['Employee_ID', 'Effective_Year'])['Is_Promoted']
                .max().reset_index()
            )

            df_hc = pd.merge(
                df_hc,
                df_promo_dedup[['Employee_ID', 'Effective_Year', 'Is_Promoted']],
                left_on=['Employee_ID', 'Year'],
                right_on=['Employee_ID', 'Effective_Year'],
                how='left'
            )
            df_hc.drop(columns=['Effective_Year'], inplace=True, errors='ignore')
            df_hc['Is_Promoted'] = df_hc['Is_Promoted'].fillna(0).astype(int)
            print("   -> Promotions merged.")
        else:
            missing = [c for c in required_promo_cols if c not in df_promo.columns]
            print(f"   -> WARNING: Missing promotion columns: {missing}")
            df_hc['Is_Promoted'] = 0
    else:
        print("   -> No promotion data loaded.")
        df_hc['Is_Promoted'] = 0

    # ----------------------------------------------------------
    # E. MERGE PLUG-IN DATA (stubs for now)
    # ----------------------------------------------------------
    print("\n" + "="*60)
    print("   STEP 4: LOADING PLUG-IN DATA SOURCES")
    print("="*60)

    df_engagement = load_engagement_data()
    if not df_engagement.empty:
        df_hc = pd.merge(df_hc, df_engagement, on='Employee_ID', how='left')

    df_comp = load_compensation_data()
    if not df_comp.empty:
        df_hc = pd.merge(df_hc, df_comp,
                         on=['Employee_ID', 'Year'], how='left')

    # ----------------------------------------------------------
    # F. ADD MANAGEMENT LEVEL AGGREGATION
    # ----------------------------------------------------------
    if 'Management Level' in df_hc.columns:
        df_hc['Management_Level_Agg'] = (
            df_hc['Management Level'].astype(str).str.strip()
            .map(MANAGEMENT_LEVEL_MAP).fillna('Unmapped')
        )
    else:
        df_hc['Management_Level_Agg'] = 'Unmapped'

    # ----------------------------------------------------------
    # G. BUILD LEAKAGE-SAFE TARGET
    # ----------------------------------------------------------
    print("\n" + "="*60)
    print("   STEP 5: BUILDING LEAKAGE-SAFE TARGET")
    print(f"   Prediction horizon: {PREDICTION_HORIZON_MONTHS} months")
    print("="*60)

    df_hc = df_hc.sort_values(by=['Employee_ID', 'Snapshot_Date'])

    # 1. Find the FIRST month each employee has HC_LVO_EE == 1
    #    This is their actual voluntary departure date
    vol_leavers = df_hc[df_hc['HC_LVO_EE'] == 1].copy()
    first_leave_date = (
        vol_leavers.groupby('Employee_ID')['Snapshot_Date']
        .min().reset_index()
        .rename(columns={'Snapshot_Date': 'Vol_Departure_Date'})
    )
    print(f"   Unique voluntary leavers found: {len(first_leave_date):,}")

    # 2. Merge departure date onto all rows
    df_hc = pd.merge(df_hc, first_leave_date, on='Employee_ID', how='left')

    # 3. Construct the forward-looking target
    #    For each active row: Will this employee voluntarily leave
    #    within the next PREDICTION_HORIZON_MONTHS months?
    df_hc['Horizon_End'] = df_hc['Snapshot_Date'].apply(
        lambda d: d + relativedelta(months=PREDICTION_HORIZON_MONTHS)
    )
    df_hc[TARGET_COLUMN] = (
        (df_hc['Vol_Departure_Date'].notna()) &
        (df_hc['Vol_Departure_Date'] > df_hc['Snapshot_Date']) &
        (df_hc['Vol_Departure_Date'] <= df_hc['Horizon_End'])
    ).astype(int)

    # 4. Keep only ACTIVE rows (CTT=1, LVO=0) — exclude leaver rows
    #    and exclude contingent workers
    active_mask = (
        (df_hc['HC_CTT_EE'] == 1) &
        (df_hc['HC_LVO_EE'] == 0) &
        (df_hc['Management_Level_Agg'] != 'Contingent Worker')
    )
    df_active = df_hc[active_mask].copy()

    print(f"   Active rows retained: {len(df_active):,}")
    print(f"   Rows with target=1:  {df_active[TARGET_COLUMN].sum():,}")
    print(f"   Rows with target=0:  {(df_active[TARGET_COLUMN] == 0).sum():,}")
    print(f"   Target rate:         {df_active[TARGET_COLUMN].mean():.2%}")

    if df_active[TARGET_COLUMN].sum() == 0:
        print("   ⚠️  WARNING: Zero positive targets. Check data or increase horizon.")

    # 5. Clean up temporary columns
    df_active.drop(columns=['Vol_Departure_Date', 'Horizon_End'],
                   inplace=True, errors='ignore')

    # ----------------------------------------------------------
    # H. DROP LEAKAGE COLUMNS
    # ----------------------------------------------------------
    leakage_to_drop = [c for c in LEAKAGE_COLUMNS if c in df_active.columns]
    if leakage_to_drop:
        df_active.drop(columns=leakage_to_drop, inplace=True, errors='ignore')
        print(f"\n   🛡️  Dropped {len(leakage_to_drop)} leakage columns: {leakage_to_drop}")

    # ----------------------------------------------------------
    # I. SAFE TYPE FIXES
    # ----------------------------------------------------------
    for col in df_active.columns:
        if df_active[col].dtype == 'object':
            df_active[col] = df_active[col].astype(str).replace('nan', None)

    if df_active['Snapshot_Date'].dtype != 'datetime64[ms]':
        df_active['Snapshot_Date'] = df_active['Snapshot_Date'].astype('datetime64[ms]')

    # ----------------------------------------------------------
    # J. SAVE & REPORT
    # ----------------------------------------------------------
    print_data_quality_report(df_active, "Master Active Headcount")

    try:
        master_path = os.path.join(PATH_OUTPUT, 'Master_Active_Headcount.parquet')
        df_active.to_parquet(master_path, index=False)
        print(f"   ✅ Saved: {master_path}")
    except Exception as e:
        print(f"   Parquet error: {e}. Saving as CSV...")
        master_path = os.path.join(PATH_OUTPUT, 'Master_Active_Headcount.csv')
        df_active.to_csv(master_path, index=False)
        print(f"   ✅ Saved: {master_path}")

    return df_active


# =============================================================
# LOAD EXISTING MASTER (for downstream use)
# =============================================================

def load_master():
    """Load the pre-built master file (parquet or csv)."""
    parquet_path = os.path.join(PATH_OUTPUT, 'Master_Active_Headcount.parquet')
    csv_path = os.path.join(PATH_OUTPUT, 'Master_Active_Headcount.csv')

    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        print(f"Loaded master parquet: {len(df):,} rows")
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path, low_memory=False)
        print(f"Loaded master CSV: {len(df):,} rows")
    else:
        print("No master file found. Running build_master()...")
        df = build_master()

    df['Snapshot_Date'] = pd.to_datetime(df['Snapshot_Date'])
    return df


# =============================================================
# CLI ENTRY POINT
# =============================================================

if __name__ == '__main__':
    print("\n" + "🚀 " * 20)
    print("   ATTRITION PIPELINE: DATA LOADER")
    print("🚀 " * 20)
    df = build_master()
    print(f"\nFinal master shape: {df.shape}")
