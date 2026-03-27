import pandas as pd
import glob
import os
import sys
import re
import io
import msoffcrypto

# =========================
# 1. CONFIGURATION
# =========================
PATH_HEADCOUNT   = '../01_Raw_Headcount/'
PATH_PERFORMANCE = '../02_Raw_Performance/'
PATH_PROMOTIONS  = '../03_Raw_Promotion/'
PATH_EOS         = '../04_Raw_EOS/'
PATH_OUTPUT      = '../05_Master_Data/'

SKIP_HEADCOUNT   = 3
SKIP_PERFORMANCE = 1
SKIP_PROMOTIONS  = 2

EOS_PASSWORD = "E@SD@t@"

# EOS is administered in March and September every year.
# This list drives the non-response flag logic — we know exactly
# which month-year combinations SHOULD have a survey score.
EOS_CYCLE_MONTHS = [3, 9]   # March = 3, September = 9

os.makedirs(PATH_OUTPUT, exist_ok=True)

# =========================
# 2. MAPPINGS
# =========================
PERFORMANCE_MAP = {
    'Significantly Outperformed': 5,
    'Exceeding': 5,
    'Outperformed': 4,
    'Strongly Performed': 3,
    'Tracking': 3,
    'Achieved Most Expectations': 2,
    'Needs Improvement': 1,
    'New Hire: Needs Improvement': 1
}

MONTH_MAP = {
    'jan': 1, 'feb': 2, 'mar': 3,
    'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9,
    'oct': 10, 'nov': 11, 'dec': 12
}

# =========================
# 3. HELPERS
# =========================
def clean_id(val):
    s = str(val).strip()
    if s.endswith('.0'):
        s = s[:-2]
    return s

def load_folder_csvs(folder_path, header_row_index):
    files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, header=header_row_index, low_memory=False))
        except Exception as e:
            print(f"Error reading {f}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def read_excel_with_password(path):
    """
    Try normal read first. If encrypted, decrypt using msoffcrypto.
    Returns a pandas ExcelFile object or None on failure.
    """
    try:
        return pd.ExcelFile(path, engine="openpyxl")
    except Exception:
        try:
            decrypted = io.BytesIO()
            with open(path, "rb") as f:
                office = msoffcrypto.OfficeFile(f)
                office.load_key(password=EOS_PASSWORD)
                office.decrypt(decrypted)
            return pd.ExcelFile(decrypted, engine="openpyxl")
        except Exception as e:
            print(f"   -> Skipping EOS file {os.path.basename(path)}: {e}")
            return None

# =========================
# A. HEADCOUNT
# =========================
print("--- 1. Processing Headcount ---")
df_hc = load_folder_csvs(PATH_HEADCOUNT, SKIP_HEADCOUNT)
if df_hc.empty:
    sys.exit("STOP: Headcount Empty")

df_hc.columns = df_hc.columns.str.strip()
df_hc.rename(columns={
    'Data As Of': 'Snapshot_Date',
    'Employee ID/Position ID': 'Employee_ID'
}, inplace=True)

df_hc['Snapshot_Date'] = pd.to_datetime(df_hc['Snapshot_Date'], errors='coerce')
mask = df_hc['Snapshot_Date'].isna()
if mask.any():
    raw = df_hc.loc[mask, 'Snapshot_Date']
    df_hc.loc[mask, 'Snapshot_Date'] = pd.to_datetime(
        pd.to_numeric(raw, errors='coerce'),
        unit='D', origin='1899-12-30'
    )

df_hc['Snapshot_Date'] = df_hc['Snapshot_Date'].astype('datetime64[ms]')
df_hc['Year']       = df_hc['Snapshot_Date'].dt.year
df_hc['Month_Date'] = df_hc['Snapshot_Date'].dt.to_period('M').dt.to_timestamp()
df_hc['Employee_ID'] = df_hc['Employee_ID'].apply(clean_id)
df_hc = df_hc.dropna(subset=['Snapshot_Date'])

# =========================
# B. PERFORMANCE
# =========================
print("--- 2. Processing Performance ---")
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
        df_perf_long['Effective_Year']   = df_perf_long['Performance_Year'] + 1
        df_perf_long['Performance_Score'] = (
            df_perf_long['Performance_Text'].str.strip().map(PERFORMANCE_MAP)
        )
        df_perf_long['Employee_ID'] = df_perf_long['Employee_ID'].apply(clean_id)

        df_hc = pd.merge(
            df_hc,
            df_perf_long[['Employee_ID', 'Effective_Year', 'Performance_Text', 'Performance_Score']],
            left_on=['Employee_ID', 'Year'],
            right_on=['Employee_ID', 'Effective_Year'],
            how='left'
        )
        df_hc.drop(columns=['Effective_Year'], inplace=True)
        df_hc = df_hc.sort_values(['Employee_ID', 'Snapshot_Date'])
        df_hc[['Performance_Text', 'Performance_Score']] = (
            df_hc.groupby('Employee_ID')[['Performance_Text', 'Performance_Score']].ffill()
        )

# =========================
# C. PROMOTIONS
# =========================
print("--- 3. Processing Promotions ---")
df_promo = load_folder_csvs(PATH_PROMOTIONS, SKIP_PROMOTIONS)
if not df_promo.empty:
    df_promo.columns = df_promo.columns.str.strip()
    if 'Employee' in df_promo.columns:
        df_promo.rename(columns={'Employee': 'Employee_ID'}, inplace=True)
    elif 'Employee ID' in df_promo.columns:
        df_promo.rename(columns={'Employee ID': 'Employee_ID'}, inplace=True)

    df_promo.rename(columns={
        'Process Year': 'Process_Year',
        'Nomination Status - After Memo': 'Nomination_Status'
    }, inplace=True)

    df_promo['Is_Promoted'] = (
        df_promo['Nomination_Status']
        .astype(str)
        .str.contains('Nominated', case=False, na=False)
        .astype(int)
    )

    df_promo['Process_Year']   = pd.to_numeric(df_promo['Process_Year'], errors='coerce')
    df_promo['Effective_Year'] = df_promo['Process_Year'] + 1
    df_promo['Employee_ID']    = df_promo['Employee_ID'].apply(clean_id)

    df_promo = df_promo.dropna(subset=['Employee_ID', 'Effective_Year'])
    df_promo = (
        df_promo
        .groupby(['Employee_ID', 'Effective_Year'])['Is_Promoted']
        .max()
        .reset_index()
    )

    df_hc = pd.merge(
        df_hc,
        df_promo,
        left_on=['Employee_ID', 'Year'],
        right_on=['Employee_ID', 'Effective_Year'],
        how='left'
    )
    df_hc['Is_Promoted'] = df_hc['Is_Promoted'].fillna(0).astype(int)
    df_hc.drop(columns=['Effective_Year'], inplace=True)

# =========================
# D. EOS — FULL FEATURE SET
#
# Produces four columns:
#   Intent_Stay_EOS      — LOCF score (including Jan/Feb 2022 fix)
#   EOS_Delta            — change vs prior survey cycle (negative = declining)
#   Months_Since_Survey  — staleness of the score (0 in survey month, up to 5)
#   EOS_Nonresponse      — 1 if employee was active in a survey month but did not respond
#
# Jan/Feb 2022 fix:
#   Core data starts Jan 2022 so there is no Sept 2021 row to carry forward from.
#   Fix: for each employee, find their EARLIEST available EOS score and back-fill
#   it to any earlier months that would otherwise be null. This ensures Jan and Feb
#   2022 receive the March 2022 score (or earliest available) rather than staying null.
#   This is conservative — it does not invent a 2021 score, it just ensures the
#   first known score is present from the start of the employee's window.
# =========================
print("--- 4. Processing EOS ---")

eos_rows = []

for root, _, files in os.walk(PATH_EOS):
    for f in files:
        if not f.endswith('.xlsx') or f.startswith('~$'):
            continue

        folder     = os.path.basename(root).lower()
        month      = next((v for k, v in MONTH_MAP.items() if k in folder), None)
        year_match = re.search(r'20\d{2}', root)
        if not month or not year_match:
            continue

        year     = int(year_match.group())
        eos_date = pd.Timestamp(year=year, month=month, day=1)

        xls = read_excel_with_password(os.path.join(root, f))
        if xls is None:
            continue

        sheet = next(
            (s for s in ['Data', 'Clean', 'Clean Data'] if s in xls.sheet_names),
            None
        )
        if sheet is None:
            continue

        df = pd.read_excel(xls, sheet_name=sheet)
        df.columns = df.columns.str.strip()

        if 'EE_ID' in df.columns:
            df['Employee_ID'] = df['EE_ID'].apply(clean_id)
        elif 'Participant Unique Identifier' in df.columns:
            df['Employee_ID'] = df['Participant Unique Identifier'].apply(clean_id)
        else:
            continue

        if 'Intent_Stay' not in df.columns:
            continue

        tmp = df[['Employee_ID', 'Intent_Stay']].copy()
        tmp['EOS_Date'] = eos_date
        eos_rows.append(tmp)

if eos_rows:
    df_eos = pd.concat(eos_rows, ignore_index=True)
    df_eos['Employee_ID'] = df_eos['Employee_ID'].astype(str)

    # ------------------------------------------------------------------
    # STEP D1: Merge raw EOS scores onto headcount on exact survey months
    # This gives us a score only in March and September rows initially.
    # ------------------------------------------------------------------
    df_hc = pd.merge(
        df_hc,
        df_eos.rename(columns={'Intent_Stay': 'Intent_Stay_Raw', 'EOS_Date': 'EOS_Date'}),
        left_on=['Employee_ID', 'Month_Date'],
        right_on=['Employee_ID', 'EOS_Date'],
        how='left'
    )
    df_hc.drop(columns=['EOS_Date'], inplace=True, errors='ignore')

    # ------------------------------------------------------------------
    # STEP D2: Forward-fill to get LOCF score across all months
    # After ffill, each row carries the most recent survey score forward.
    # This is the primary score column used as a model feature.
    # ------------------------------------------------------------------
    df_hc = df_hc.sort_values(['Employee_ID', 'Month_Date'])
    df_hc['Intent_Stay_EOS'] = (
        df_hc.groupby('Employee_ID')['Intent_Stay_Raw'].transform(lambda x: x.ffill())
    )

    # ------------------------------------------------------------------
    # STEP D3: Jan/Feb 2022 fix — backward-fill the EARLIEST known score
    #
    # Problem: Jan and Feb 2022 sit before the first survey cycle in the
    # data window (March 2022). ffill() finds no prior value so they stay null.
    #
    # Fix: for each employee, take their first ever non-null EOS score and
    # back-fill it to any rows that are still null BEFORE that score arrived.
    # This is equivalent to saying "we don't know their score was different
    # before March 2022, so we use March 2022 as the best available proxy."
    #
    # bfill() only affects rows that are still null after ffill() — i.e.
    # only rows that precede the employee's very first survey response.
    # ------------------------------------------------------------------
    df_hc['Intent_Stay_EOS'] = (
        df_hc.groupby('Employee_ID')['Intent_Stay_EOS'].transform(lambda x: x.bfill())
    )

    # ------------------------------------------------------------------
    # STEP D4: Compute EOS_Delta — change vs the previous survey cycle
    #
    # Logic:
    #   1. Keep only survey-month rows with actual responses.
    #   2. Within each employee, shift the score by 1 to get the prior value.
    #   3. Compute delta = current - prior.
    #   4. Merge delta back onto all headcount rows.
    #   5. Forward-fill delta the same way as the score itself.
    #
    # A negative delta means engagement dropped since last survey — one of
    # the strongest leading indicators of voluntary attrition.
    # ------------------------------------------------------------------
    df_eos_scored = df_eos.copy()
    df_eos_scored = df_eos_scored.sort_values(['Employee_ID', 'EOS_Date'])
    df_eos_scored['Intent_Stay_Prev'] = (
        df_eos_scored.groupby('Employee_ID')['Intent_Stay'].shift(1)
    )
    df_eos_scored['EOS_Delta'] = (
        df_eos_scored['Intent_Stay'] - df_eos_scored['Intent_Stay_Prev']
    )
    # Delta is null for an employee's very first survey — that is correct.
    # There is no prior cycle to compare against.

    df_hc = pd.merge(
        df_hc,
        df_eos_scored[['Employee_ID', 'EOS_Date', 'EOS_Delta']].rename(
            columns={'EOS_Date': 'EOS_Date_Delta'}
        ),
        left_on=['Employee_ID', 'Month_Date'],
        right_on=['Employee_ID', 'EOS_Date_Delta'],
        how='left'
    )
    df_hc.drop(columns=['EOS_Date_Delta'], inplace=True, errors='ignore')

    # Forward-fill delta the same way as the score
    df_hc['EOS_Delta'] = (
        df_hc.groupby('Employee_ID')['EOS_Delta'].transform(lambda x: x.ffill())
    )
    # Note: intentionally do NOT back-fill delta for Jan/Feb 2022.
    # The delta for the first cycle is genuinely unknown — leave as null.
    # The model will treat null delta as missing, which is correct.

    # ------------------------------------------------------------------
    # STEP D5: Compute Months_Since_Survey
    #
    # For each row, find the most recent survey date that is <= this row's
    # Month_Date, then compute the difference in months.
    #
    # This tells the model how stale the score is:
    #   0 = survey month itself (freshest)
    #   1 = one month after survey
    #   ...up to 5 = month just before the next survey
    #
    # After the Jan/Feb 2022 back-fill, those rows get Months_Since_Survey
    # computed from the March 2022 survey date — which means Jan 2022 = -2
    # and Feb 2022 = -1. We floor these at 0 to avoid negative staleness.
    # ------------------------------------------------------------------

    # Build a lookup: for each employee, the sorted list of their survey dates
    all_survey_dates = (
        df_eos[['Employee_ID', 'EOS_Date']]
        .drop_duplicates()
        .sort_values(['Employee_ID', 'EOS_Date'])
    )

    # Merge as-of: for each headcount row, find the latest survey date <= Month_Date
    df_hc = df_hc.sort_values(['Employee_ID', 'Month_Date'])
    all_survey_dates = all_survey_dates.sort_values(['Employee_ID', 'EOS_Date'])

    df_hc = pd.merge_asof(
        df_hc,
        all_survey_dates.rename(columns={'EOS_Date': 'Last_Survey_Date'}),
        left_on='Month_Date',
        right_on='Last_Survey_Date',
        by='Employee_ID',
        direction='backward'   # find latest survey date <= snapshot month
    )

    # Compute month difference and floor at 0 (handles Jan/Feb 2022 edge case)
    df_hc['Months_Since_Survey'] = (
        (df_hc['Month_Date'].dt.year  - df_hc['Last_Survey_Date'].dt.year) * 12 +
        (df_hc['Month_Date'].dt.month - df_hc['Last_Survey_Date'].dt.month)
    ).clip(lower=0)

    # Employees with no survey at all get null — correct, do not impute
    df_hc.loc[df_hc['Last_Survey_Date'].isna(), 'Months_Since_Survey'] = None
    df_hc.drop(columns=['Last_Survey_Date'], inplace=True, errors='ignore')

    # ------------------------------------------------------------------
    # STEP D6: Compute EOS_Nonresponse flag
    #
    # Definition: employee was active in a survey cycle month (March or
    # September) but has no raw Intent_Stay score for that specific cycle.
    #
    # This is a weak behavioral signal — employees who skip a survey they
    # previously completed may be showing early disengagement.
    #
    # Value encoding:
    #   1   = active in survey month, expected to respond, did NOT respond
    #   0   = responded (or first ever survey so no prior to compare)
    #   NaN = month is not a survey cycle month (not applicable)
    #
    # Important: we only flag non-response for employees who have responded
    # to AT LEAST ONE prior survey. A new joiner missing their first survey
    # is not non-response — it may simply be they were excluded by policy.
    # ------------------------------------------------------------------

    # Identify employees who have responded to at least one survey ever
    ever_responded = (
        df_eos.groupby('Employee_ID')['Intent_Stay']
        .count()
        .reset_index()
        .rename(columns={'Intent_Stay': 'N_Surveys_Taken'})
    )
    ever_responded = ever_responded[ever_responded['N_Surveys_Taken'] >= 1]['Employee_ID'].tolist()

    # Flag rows that fall on a survey cycle month
    df_hc['_Is_Survey_Month'] = df_hc['Month_Date'].dt.month.isin(EOS_CYCLE_MONTHS)

    # Flag rows that have a raw score (responded to this specific cycle)
    df_hc['_Has_Raw_Score'] = df_hc['Intent_Stay_Raw'].notna()

    # Flag employees who have ever responded
    df_hc['_Ever_Responded'] = df_hc['Employee_ID'].isin(ever_responded)

    # Non-response = survey month + ever responded before + no score this cycle
    df_hc['EOS_Nonresponse'] = None  # default: not applicable (non-survey months)
    survey_month_mask = df_hc['_Is_Survey_Month']
    df_hc.loc[survey_month_mask, 'EOS_Nonresponse'] = 0  # default to responded
    df_hc.loc[
        survey_month_mask & df_hc['_Ever_Responded'] & ~df_hc['_Has_Raw_Score'],
        'EOS_Nonresponse'
    ] = 1

    # Convert to nullable integer so NaN stays NaN (not -9999 or similar)
    df_hc['EOS_Nonresponse'] = df_hc['EOS_Nonresponse'].astype('Int8')

    # Clean up internal helper columns
    df_hc.drop(
        columns=['Intent_Stay_Raw', '_Is_Survey_Month', '_Has_Raw_Score', '_Ever_Responded'],
        inplace=True,
        errors='ignore'
    )

else:
    print("WARNING: No EOS data loaded. All EOS columns will be null.")
    df_hc['Intent_Stay_EOS']  = None
    df_hc['EOS_Delta']        = None
    df_hc['Months_Since_Survey'] = None
    df_hc['EOS_Nonresponse']  = None

# =========================
# E. SAVE
# =========================
print("--- 5. Saving Master File ---")
print(f"\nEOS column summary:")
for col in ['Intent_Stay_EOS', 'EOS_Delta', 'Months_Since_Survey', 'EOS_Nonresponse']:
    if col in df_hc.columns:
        null_pct = df_hc[col].isna().mean() * 100
        print(f"  {col}: {null_pct:.1f}% null")

for c in df_hc.columns:
    if df_hc[c].dtype == 'object':
        df_hc[c] = df_hc[c].astype(str).replace('nan', None)

out_path = os.path.join(PATH_OUTPUT, 'Master_Headcount_Full.parquet')
df_hc.to_parquet(out_path, index=False)
print(f"\nDONE ✅  Saved: {out_path}")
