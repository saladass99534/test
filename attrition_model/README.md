# Attrition Prediction Pipeline — Project Setup & Workflow Guide

## 📁 Project Structure

```
attrition_model/                    ← Project root
├── config.py                       ← Central configuration (paths, features, horizon)
├── data_loader.py                  ← Data ingestion, validation, target construction
├── feature_engine.py               ← Feature engineering (30+ features)
├── model_pipeline.py               ← XGBoost, Survival, SHAP, CV, fairness
├── dashboard.py                    ← Streamlit dashboard (5 tabs)
├── run_pipeline.py                 ← End-to-end orchestration
├── requirements.txt                ← Python dependencies
├── README.md                       ← This file
└── model_artifacts/                ← Generated model outputs (auto-created)
    ├── xgboost_model.joblib
    ├── logistic_baseline.joblib
    ├── feature_scaler.joblib
    ├── shap_values.parquet
    ├── shap_feature_importance.csv
    ├── predictions.parquet
    ├── full_predictions.parquet
    ├── cox_ph_summary.csv
    ├── kaplan_meier_by_level.csv
    ├── fairness_audit.csv
    └── model_params.json

../01_Raw_Headcount/                ← Raw headcount CSVs
../02_Raw_Performance/              ← Raw performance CSVs
../03_Raw_Promotion/                ← Raw promotion CSVs
../04_Master_Data/                  ← Processed master files (auto-created)
    ├── Master_Active_Headcount.parquet
    └── Model_Ready_Data.parquet
```

## 🔧 First-Time Setup

```bash
# 1. Navigate to project
cd attrition_model

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

## 🚀 Running the Pipeline

### Full Run (with hyperparameter tuning)
```bash
python run_pipeline.py
```

### Quick Run (skip tuning, faster)
```bash
python run_pipeline.py --no-tune
```

### Specify prediction year and threshold
```bash
python run_pipeline.py --year 2025 --threshold 0.4
```

### Re-run with existing master file (skip raw data loading)
```bash
python run_pipeline.py --skip-data --no-tune
```

### Launch Dashboard
```bash
streamlit run dashboard.py
```

## 🔁 Typical Workflow

```
┌─────────────────────────┐
│  1. Place raw CSVs in   │
│  the 3 raw folders      │
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│  2. Run pipeline:       │
│  python run_pipeline.py │
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│  3. Launch dashboard:   │
│  streamlit run           │
│  dashboard.py           │
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│  4. Review results:     │
│  - Overview tab         │
│  - Employee risk cards  │
│  - Manager scorecard    │
│  - Survival curves      │
│  - Fairness audit       │
└─────────────────────────┘
```

## ⚙️ Configuration Reference

All configuration is in **`config.py`**. Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PREDICTION_HORIZON_MONTHS` | `6` | How far ahead to predict voluntary departure |
| `PATH_HEADCOUNT` | `../01_Raw_Headcount/` | Raw headcount CSV folder |
| `PATH_PERFORMANCE` | `../02_Raw_Performance/` | Raw performance CSV folder |
| `PATH_PROMOTIONS` | `../03_Raw_Promotion/` | Raw promotion CSV folder |
| `CV_FOLDS` | `5` | Stratified K-Fold cross-validation folds |
| `OPTUNA_N_TRIALS` | `50` | Hyperparameter tuning iterations |

## 🔌 Adding New Data Sources

The pipeline is designed to be modular. To add engagement surveys or compensation data:

### Engagement Data
1. Open `data_loader.py`
2. Find `load_engagement_data()` function (currently a stub)
3. Replace the stub with your data loading logic:
```python
def load_engagement_data():
    df = pd.read_csv('path/to/engagement_surveys.csv')
    # Return with columns: Employee_ID, Survey_Date, Satisfaction_Score, Engagement_Score
    return df
```
4. The merge logic is already wired up in `build_master()`

### Compensation Data
1. Open `data_loader.py`
2. Find `load_compensation_data()` function
3. Replace with your loading logic:
```python
def load_compensation_data():
    df = pd.read_csv('path/to/compensation.csv')
    # Return with columns: Employee_ID, Year, Base_Salary, Bonus_Amount, Total_Comp
    return df
```

### Adding New Features
1. Open `feature_engine.py`
2. Add your feature logic in the appropriate section
3. Add the feature name to `BASE_FEATURES` in `config.py`
4. Re-run the pipeline

## 🛡️ Data Leakage Safeguards

The pipeline has built-in protections against data leakage:

1. **Leakage columns** listed in `config.LEAKAGE_COLUMNS` are automatically dropped
2. **Target is forward-looking**: Uses future voluntary departure within the horizon window
3. **Leaver rows excluded**: Rows where `HC_LVO_EE == 1` are never in the training/prediction set
4. **Org-level features are lagged**: Job family and org churn rates use prior-year data
5. **Manager churn counts are lagged**: Uses Year-1 data only

## ⚖️ Fairness & Ethics

- Protected attributes (Gender, Ethnicity, Age, etc.) are **never** used as model features
- They ARE used for the **fairness audit** to detect algorithmic bias
- The dashboard Fairness tab shows flag rate disparities across protected groups
- Groups flagged at >20% higher rate than overall are highlighted as potential bias

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `FileNotFoundError` for raw data | Check that CSVs are in the correct `01_Raw_Headcount/` etc. folders |
| `0 positive targets` | Increase `PREDICTION_HORIZON_MONTHS` or check data date ranges |
| SMOTE fails | Ensure at least 6 positive targets; reduce `k_neighbors` in `model_pipeline.py` |
| Dashboard shows stale data | Delete `model_artifacts/` and re-run pipeline |
| Parquet save fails | Pipeline auto-falls back to CSV format |
