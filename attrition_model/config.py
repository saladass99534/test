"""
Attrition Prediction Pipeline — Central Configuration
======================================================
All paths, constants, mappings, and feature lists.
Change PREDICTION_HORIZON_MONTHS below to adjust the target window.
"""

import os

# ============================================================
# 1. PREDICTION HORIZON (Change this single value as needed)
# ============================================================
PREDICTION_HORIZON_MONTHS = 6

# ============================================================
# 2. FILE PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Raw data folders (relative to project root — adjust as needed)
PATH_HEADCOUNT   = os.path.join(BASE_DIR, '..', '01_Raw_Headcount')
PATH_PERFORMANCE = os.path.join(BASE_DIR, '..', '02_Raw_Performance')
PATH_PROMOTIONS  = os.path.join(BASE_DIR, '..', '03_Raw_Promotion')
PATH_OUTPUT      = os.path.join(BASE_DIR, '..', '04_Master_Data')

# Model artifacts output
PATH_MODEL_OUTPUT = os.path.join(BASE_DIR, 'model_artifacts')

# Header rows to skip (0-based index)
SKIP_HEADCOUNT   = 3
SKIP_PERFORMANCE = 1
SKIP_PROMOTIONS  = 2

# ============================================================
# 3. LEAKAGE COLUMNS — NEVER USE AS FEATURES
# ============================================================
LEAKAGE_COLUMNS = [
    'HC_Future_Term_Current Year_EE_Vol',
    'HC_Future_Term_Current Year_EE_Invol',
    'HC_Future_Term_Current Year_EE_Other',
    'HC_Future_Term_EE_Current_Year_1_0',
    'HC_Future_Term_EE',
    'Will_Churn_Eventually',         # Old target (leaky)
    'Is_Leaver_Row',                 # Direct indicator
    'Termination Date',              # Future info
    'Termination Month Number',      # Future info
    'Termination Reason Category',   # Only exists after leaving
    'Termination Reason',            # Only exists after leaving
]

# Columns used only for fairness auditing — NOT model features
PROTECTED_COLUMNS = [
    'Gender', 'US Ethnicity', 'US Ethnicity Short Name',
    'US White+ Flag', 'US Latinx+ Flag', 'US Black+ Flag',
    'US Asian+ Flag', 'US Other+ Flag',
    'Age', 'Generation', 'Veteran Status', 'Disability Status',
    'Date of Birth',
]

# ============================================================
# 4. PERFORMANCE RATING MAP
# ============================================================
PERFORMANCE_MAP = {
    'Significantly Outperformed': 5,
    'Exceeding': 5,
    'Outperformed': 4,
    'Strongly Performed': 3,
    'Tracking': 3,
    'Achieved Most Expectations': 2,
    'Needs Improvement': 1,
    'New Hire: Needs Improvement': 1,
}

# ============================================================
# 5. MANAGEMENT LEVEL MAPPING
# ============================================================
MANAGEMENT_LEVEL_MAP = {
    'Admin Business Coordinator': 'Admin',
    'Admin Business Lead': 'Admin',
    'Admin Business Partner': 'Admin',
    'Administrative Assistant': 'Admin',
    'Analyst': 'Analyst',
    'Associate': 'Associate',
    'CEO-Chairman': 'MD+',
    'Contingent Worker': 'Contingent Worker',
    'Director': 'Director',
    'Employee Fixed Term': 'Admin',
    'Executive Assistant': 'Admin',
    'Fund Partner': 'MD+',
    'Intern': 'Intern',
    'Managing Director': 'MD+',
    'Partner': 'MD+',
    'President': 'MD+',
    'Principal': 'Director',
    'Senior Advisor': 'MD+',
    'Senior Associate': 'Associate',
    'Senior Managing Director': 'MD+',
    'Specialist': 'Specialist',
    'Vice Chairman': 'MD+',
    'Vice President': 'Vice President',
    'Executive Director': 'Director',
}

# Expected years-to-promote benchmarks per level
LEVEL_BENCHMARKS = {
    'Analyst': 2.0,
    'Associate': 3.0,
    'Vice President': 4.0,
    'Director': 4.0,
    'MD+': 5.0,
    'Admin': 5.0,
    'Specialist': 3.0,
}

# ============================================================
# 6. MODEL FEATURES
# ============================================================
# Base numeric features used by the model
BASE_FEATURES = [
    # Tenure & Career
    'Total_Tenure_Years',
    'Time_In_Title_Current',
    'Cycles_In_Level',
    'Is_Stuck_In_Level',
    'Days_Since_Last_Promotion',
    'Consecutive_Low_Perf_Years',

    # Performance
    'Performance_Score',
    'Is_High_Performer',
    'Performance_Drop_Flag',
    'Perf_Score_Rolling_Avg_3Yr',
    'Perf_Trajectory_Slope',

    # Peer dynamics
    'Peer_Promo_Rate',
    'Peer_Pressure_Flag',

    # Manager intelligence
    'Manager_Past_Churn_Count',
    'Manager_Performance_Score',
    'Performer_vs_Mgr_Score',
    'Manager_Changed_Recently',
    'Manager_Span_Of_Control',

    # Org-level risk
    'Job_Family_Churn_Rate',
    'Org_Level_Churn_Rate',

    # Employment context
    'Is_Lateral_Hire',
    'Is_Campus_Hire',
    'Has_International_Assignment',
    'Is_Acquisition_Employee',
    'FTE_Value',
    'Is_Reorg_Affected',
    'Is_Transfer_Recent',

    # Temporal
    'Month_Sin',
    'Month_Cos',
    'Is_Post_Bonus_Window',
]

# Categorical features to one-hot encode
CATEGORICAL_FEATURES = [
    'Management_Level_Agg',
]

# Columns to carry through for display but NOT use as features
ID_COLUMNS = [
    'Employee_ID', 'Snapshot_Date', 'Year', 'Month',
    'Worker', 'Manager ID', 'Manager',
    'Management Level', 'Management_Level_Agg',
    'Level 2 Org', 'Level 3 Org', 'Level 4 Org',
    'Region', 'Country', 'City',
    'Job Family', 'Job Family Group',
]

# Target column name
TARGET_COLUMN = 'Will_Leave_Within_Horizon'

# ============================================================
# 7. MODEL HYPERPARAMETER DEFAULTS
# ============================================================
XGBOOST_DEFAULT_PARAMS = {
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'gamma': 0.1,
    'eval_metric': 'auc',
    'random_state': 42,
    'use_label_encoder': False,
}

# Cross-validation settings
CV_FOLDS = 5
OPTUNA_N_TRIALS = 50

# ============================================================
# 8. SURVIVAL ANALYSIS SETTINGS
# ============================================================
SURVIVAL_DURATION_COL = 'Tenure_Months_At_Snapshot'
SURVIVAL_EVENT_COL = 'Vol_Departure_Event'

# ============================================================
# 9. DATA QUALITY
# ============================================================
EXPECTED_HEADCOUNT_COLUMNS = [
    'Data As Of', 'Employee ID/Position ID', 'Management Level',
    'HC_CTT_EE', 'HC_LVO_EE', 'HC_LIN_EE', 'HC_LOT_EE',
    'Manager ID', 'Length of Service',
]
