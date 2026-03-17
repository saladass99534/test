"""
Attrition Risk Dashboard — Enhanced Streamlit Application
==========================================================
Replaces Cell 3 dashboard from the original notebook.
New features:
  - Individual employee SHAP risk cards
  - Manager scorecard
  - Survival curves (Kaplan-Meier)
  - Trend over time
  - Fairness audit tab
  - All original functionality preserved
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_auc_score, confusion_matrix

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    from lifelines import KaplanMeierFitter
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False

from config import (
    BASE_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN,
    PATH_MODEL_OUTPUT, PATH_OUTPUT, PREDICTION_HORIZON_MONTHS,
    ID_COLUMNS,
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Attrition Risk Dashboard",
    layout="wide",
    page_icon="📉",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                padding: 15px; border-radius: 10px; border: 1px solid #1f4068; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a2e; border-radius: 6px; padding: 8px 16px;
        border: 1px solid #1f4068; color: #e0e0e0;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0f3460 0%, #533483 100%);
        border-color: #e94560;
    }
    div[data-testid="stSidebarContent"] {
        background: linear-gradient(180deg, #0e1117 0%, #1a1a2e 100%);
    }
    h1, h2, h3 { color: #e0e0e0; }
    .risk-high { color: #e94560; font-weight: bold; }
    .risk-medium { color: #f5a623; font-weight: bold; }
    .risk-low { color: #00d4aa; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 1. LOAD DATA (Cached)
# ============================================================
@st.cache_data
def load_data():
    """Load the featured model-ready data."""
    paths = [
        os.path.join(PATH_OUTPUT, 'Model_Ready_Data.parquet'),
        'Model_Ready_Data.parquet',
        os.path.join(PATH_OUTPUT, 'Model_Ready_Data.csv'),
    ]
    for p in paths:
        if os.path.exists(p):
            if p.endswith('.parquet'):
                df = pd.read_parquet(p)
            else:
                df = pd.read_csv(p, low_memory=False)
            df['Snapshot_Date'] = pd.to_datetime(df['Snapshot_Date'])
            return df
    st.error("❌ No model-ready data found. Run `python run_pipeline.py` first.")
    st.stop()


@st.cache_resource
def load_model():
    """Load the trained XGBoost model."""
    model_path = os.path.join(PATH_MODEL_OUTPUT, 'xgboost_model.joblib')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


@st.cache_data
def load_shap_values():
    """Load pre-computed SHAP values."""
    shap_path = os.path.join(PATH_MODEL_OUTPUT, 'shap_values.parquet')
    if os.path.exists(shap_path):
        return pd.read_parquet(shap_path)
    return None


@st.cache_data
def load_fairness_report():
    """Load pre-computed fairness audit."""
    fair_path = os.path.join(PATH_MODEL_OUTPUT, 'fairness_audit.csv')
    if os.path.exists(fair_path):
        return pd.read_csv(fair_path)
    return None


@st.cache_data
def load_feature_importance():
    """Load SHAP-based feature importance."""
    imp_path = os.path.join(PATH_MODEL_OUTPUT, 'shap_feature_importance.csv')
    if os.path.exists(imp_path):
        return pd.read_csv(imp_path)
    return None


# ============================================================
# 2. SIDEBAR CONTROLS
# ============================================================
df = load_data()
model = load_model()

st.sidebar.title("📊 Controls")
st.sidebar.caption(f"Prediction Horizon: {PREDICTION_HORIZON_MONTHS} months")

# Year & Threshold
available_years = sorted(df['Year'].unique())
latest_year = max(available_years)
year = st.sidebar.selectbox(
    "Prediction Year", available_years,
    index=available_years.index(latest_year)
)
threshold = st.sidebar.slider("Risk Threshold", 0.1, 0.9, 0.5, 0.05)
st.sidebar.markdown("---")


# Filter function
def render_filter(label, col_name):
    if col_name not in df.columns:
        return []
    options = sorted(df[col_name].dropna().astype(str).unique())
    key_all = f"all_{col_name}"
    key_select = f"select_{col_name}"
    is_all = st.sidebar.checkbox(f"Select All {label}", value=True, key=key_all)
    if is_all:
        return options
    else:
        return st.sidebar.multiselect(f"Select {label}", options, default=[], key=key_select)


# Render filters
level_filter = render_filter("Management Level", "Management_Level_Agg")
l2_filter = render_filter("Level 2 Org", "Level 2 Org")
l3_filter = render_filter("Level 3 Org", "Level 3 Org")
l4_filter = render_filter("Level 4 Org", "Level 4 Org")
region_filter = render_filter("Region", "Region")
country_filter = render_filter("Country", "Country")
city_filter = render_filter("City", "City")

filters = {
    'Management_Level_Agg': level_filter,
    'Level 2 Org': l2_filter,
    'Level 3 Org': l3_filter,
    'Level 4 Org': l4_filter,
    'Region': region_filter,
    'Country': country_filter,
    'City': city_filter,
}


# ============================================================
# 3. DATA PROCESSING
# ============================================================
@st.cache_data
def get_xy(_df, year, filters_dict):
    dff = _df.copy()
    for col, allowed_vals in filters_dict.items():
        if col in dff.columns and allowed_vals and len(allowed_vals) > 0:
            dff = dff[dff[col].astype(str).isin(allowed_vals)]
    if dff.empty:
        return None, None, None, None, None, None

    all_features = BASE_FEATURES + CATEGORICAL_FEATURES
    available = [f for f in all_features if f in dff.columns]

    X_all = pd.get_dummies(dff[available], columns=CATEGORICAL_FEATURES, drop_first=True)
    y_all = dff[TARGET_COLUMN] if TARGET_COLUMN in dff.columns else pd.Series(0, index=dff.index)

    train_mask = dff['Year'] < year
    X_train = X_all[train_mask].fillna(0)
    y_train = y_all[train_mask].fillna(0)

    test_raw = dff[dff['Year'] == year].sort_values(
        ['Employee_ID', 'Snapshot_Date'], ascending=[True, False]
    )
    test_unique = test_raw.drop_duplicates(subset=['Employee_ID'], keep='first')

    X_test = pd.get_dummies(
        test_unique[available], columns=CATEGORICAL_FEATURES, drop_first=True
    )
    for c in X_train.columns:
        if c not in X_test.columns:
            X_test[c] = 0
    X_test = X_test[X_train.columns].fillna(0)

    y_test = test_unique[TARGET_COLUMN] if TARGET_COLUMN in test_unique.columns else pd.Series(0, index=test_unique.index)

    return X_train, y_train, X_test, y_test, test_unique, X_train.columns.tolist()


X_train, y_train, X_test, y_test, test_df, feat_names = get_xy(df, year, filters)

if X_train is None:
    st.error("⚠️ No data found for these filters. Please select more options.")
    st.stop()


# ============================================================
# 4. TRAIN / LOAD MODEL
# ============================================================
@st.cache_resource
def build_model(_X, _y):
    from xgboost import XGBClassifier
    n_pos = _y.sum()
    n_neg = len(_y) - n_pos
    scale_pos = n_neg / max(n_pos, 1)
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        eval_metric='auc', random_state=42, use_label_encoder=False
    )
    xgb.fit(_X, _y)
    return xgb


# Prefer saved model, fallback to training
if model is not None:
    # Ensure feature alignment
    for c in feat_names:
        if c not in X_test.columns:
            X_test[c] = 0
    try:
        X_test_aligned = X_test[model.get_booster().feature_names]
    except Exception:
        X_test_aligned = X_test
        model = build_model(X_train, y_train)
else:
    model = build_model(X_train, y_train)
    X_test_aligned = X_test

# Predict
try:
    y_prob = model.predict_proba(X_test_aligned)[:, 1]
except Exception:
    model = build_model(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]

y_pred = (y_prob > threshold).astype(int)

test_df = test_df.copy()
test_df['Predicted_Risk'] = y_prob
test_df['Predicted_Leaver'] = y_pred


# ============================================================
# 5. MAIN DASHBOARD
# ============================================================
st.title(f"📉 Attrition Risk Dashboard — {year}")
st.caption(f"Prediction horizon: {PREDICTION_HORIZON_MONTHS} months | Threshold: {threshold}")

# --- TABS ---
tab_overview, tab_employee, tab_manager, tab_survival, tab_fairness = st.tabs([
    "📊 Overview", "👤 Employee Risk Card", "👔 Manager Scorecard",
    "📈 Survival Curves", "⚖️ Fairness Audit"
])


# ============================================================
# TAB 1: OVERVIEW (Original + Enhanced)
# ============================================================
with tab_overview:
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    tp = ((y_pred == 1) & (y_test == 1)).sum()
    recall = tp / y_test.sum() if y_test.sum() > 0 else 0
    precision = tp / y_pred.sum() if y_pred.sum() > 0 else 0
    try:
        auc = roc_auc_score(y_test, y_prob)
    except (ValueError, Exception):
        auc = 0

    col1.metric("AUC", f"{auc:.1%}")
    col2.metric("Recall (Catch Rate)", f"{recall:.1%}")
    col3.metric("Precision (Hit Rate)", f"{precision:.1%}")
    col4.metric("Filtered Population", f"{len(test_df):,}")

    st.markdown("---")

    # Feature Importance (SHAP-based if available)
    st.subheader("1. Top Drivers of Attrition (SHAP)")
    imp_df = load_feature_importance()
    if imp_df is not None:
        imps = imp_df.head(10)
    else:
        imps = pd.DataFrame({
            'Feature': feat_names,
            'Mean_Abs_SHAP': model.feature_importances_
        }).sort_values('Mean_Abs_SHAP', ascending=False).head(10)

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    fig1.patch.set_facecolor('#0e1117')
    ax1.set_facecolor('#0e1117')
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(imps)))
    imp_col = 'Mean_Abs_SHAP' if 'Mean_Abs_SHAP' in imps.columns else 'Importance'
    bars = ax1.barh(
        imps['Feature'].values[::-1],
        imps[imp_col].values[::-1],
        color=colors[::-1], edgecolor='white', linewidth=0.5
    )
    ax1.set_xlabel("Importance (Mean |SHAP|)", color='white')
    ax1.tick_params(colors='white')
    for spine in ax1.spines.values():
        spine.set_color('#1f4068')
    st.pyplot(fig1)

    # Risk Distribution
    st.subheader("2. Risk Score Distribution")
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    fig2.patch.set_facecolor('#0e1117')
    ax2.set_facecolor('#0e1117')
    ax2.hist(y_prob, bins=50, color='#e94560', alpha=0.7, edgecolor='#0e1117')
    ax2.axvline(threshold, color='#00d4aa', linestyle='--', linewidth=2,
                label=f'Threshold {threshold}')
    ax2.set_xlabel("Predicted Risk Probability", color='white')
    ax2.set_ylabel("Count", color='white')
    ax2.tick_params(colors='white')
    ax2.legend(facecolor='#1a1a2e', edgecolor='#1f4068', labelcolor='white')
    for spine in ax2.spines.values():
        spine.set_color('#1f4068')
    st.pyplot(fig2)

    # Confusion Matrix
    st.subheader("3. Model Accuracy (Confusion Matrix)")
    cm = confusion_matrix(y_test, y_pred)
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    fig3.patch.set_facecolor('#0e1117')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                cbar_kws={'shrink': 0.8})
    ax3.set_xlabel("Predicted", color='white')
    ax3.set_ylabel("Actual", color='white')
    ax3.tick_params(colors='white')
    st.pyplot(fig3)

    # Action Lists
    st.markdown("---")
    st.header("📋 Action Lists")

    latest_date = test_df['Snapshot_Date'].max()
    st.caption(f"Showing employees active as of: **{latest_date.date()}**")

    active_mask = (test_df['Snapshot_Date'] == latest_date)
    active_df = test_df[active_mask].copy()

    high_risk_list = active_df[active_df['Predicted_Leaver'] == 1].sort_values(
        'Predicted_Risk', ascending=False
    )
    saveable_stars = high_risk_list[
        high_risk_list['Performance_Score'].isin([4, 5])
    ].copy() if 'Performance_Score' in high_risk_list.columns else pd.DataFrame()

    col_list1, col_list2 = st.columns(2)
    with col_list1:
        st.subheader("🔴 High Risk Employees")
        st.write(f"Total: {len(high_risk_list)}")
        display_cols = ['Employee_ID', 'Predicted_Risk', 'Management_Level_Agg', 'Total_Tenure_Years']
        display_cols = [c for c in display_cols if c in high_risk_list.columns]
        st.dataframe(high_risk_list[display_cols].head(100), use_container_width=True)
        if not high_risk_list.empty:
            st.download_button(
                "Download High Risk List",
                high_risk_list.to_csv(index=False),
                "High_Risk_List.csv"
            )

    with col_list2:
        st.subheader("⭐ Saveable Stars (Top Talent)")
        st.write(f"Total: {len(saveable_stars)}")
        st.caption("High Performers (Rating 4 or 5) flagged as High Risk")
        star_cols = ['Employee_ID', 'Predicted_Risk', 'Performance_Score', 'Manager_Changed_Recently']
        star_cols = [c for c in star_cols if c in saveable_stars.columns]
        st.dataframe(saveable_stars[star_cols].head(100), use_container_width=True)
        if not saveable_stars.empty:
            st.download_button(
                "Download Saveable Stars",
                saveable_stars.to_csv(index=False),
                "Saveable_Stars.csv"
            )


# ============================================================
# TAB 2: EMPLOYEE RISK CARD
# ============================================================
with tab_employee:
    st.subheader("👤 Individual Employee Risk Card")

    # Employee selector
    employee_list = test_df.sort_values('Predicted_Risk', ascending=False)['Employee_ID'].unique()
    selected_emp = st.selectbox(
        "Select Employee", employee_list,
        format_func=lambda x: f"{x} (Risk: {test_df[test_df['Employee_ID'] == x]['Predicted_Risk'].values[0]:.1%})"
    )

    if selected_emp:
        emp_data = test_df[test_df['Employee_ID'] == selected_emp].iloc[0]

        # Risk badge
        risk_score = emp_data['Predicted_Risk']
        if risk_score > 0.7:
            risk_class = "🔴 HIGH RISK"
            risk_color = "#e94560"
        elif risk_score > threshold:
            risk_class = "🟡 MEDIUM RISK"
            risk_color = "#f5a623"
        else:
            risk_class = "🟢 LOW RISK"
            risk_color = "#00d4aa"

        st.markdown(f"### {risk_class}: {risk_score:.1%}")

        # Key stats
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Tenure (Years)", f"{emp_data.get('Total_Tenure_Years', 'N/A')}")
        col_b.metric("Level", emp_data.get('Management_Level_Agg', 'N/A'))
        col_c.metric("Performance", f"{emp_data.get('Performance_Score', 'N/A')}")
        col_d.metric("Mgr Changed Recently", "Yes" if emp_data.get('Manager_Changed_Recently', 0) == 1 else "No")

        st.markdown("---")

        # SHAP waterfall for this employee
        shap_df = load_shap_values()
        if shap_df is not None and HAS_SHAP:
            st.subheader("Why This Employee Is Flagged (SHAP)")
            # Find this employee's row index
            emp_idx = test_df[test_df['Employee_ID'] == selected_emp].index
            if len(emp_idx) > 0:
                try:
                    emp_shap = shap_df.loc[emp_idx[0]]
                except (KeyError, IndexError):
                    emp_shap = None

                if emp_shap is not None:
                    # Top drivers
                    top_drivers = emp_shap.abs().nlargest(8)
                    fig_shap, ax_shap = plt.subplots(figsize=(10, 5))
                    fig_shap.patch.set_facecolor('#0e1117')
                    ax_shap.set_facecolor('#0e1117')
                    colors_shap = ['#e94560' if emp_shap[f] > 0 else '#00d4aa' for f in top_drivers.index]
                    ax_shap.barh(
                        top_drivers.index[::-1],
                        [emp_shap[f] for f in top_drivers.index[::-1]],
                        color=colors_shap[::-1], edgecolor='white', linewidth=0.5
                    )
                    ax_shap.set_xlabel("SHAP Value (→ increases risk | ← decreases risk)", color='white')
                    ax_shap.axvline(0, color='white', linewidth=0.5)
                    ax_shap.tick_params(colors='white')
                    for spine in ax_shap.spines.values():
                        spine.set_color('#1f4068')
                    st.pyplot(fig_shap)

                    st.caption("🔴 Red bars push risk UP | 🟢 Green bars push risk DOWN")
        else:
            st.info("Run the full pipeline to generate SHAP values for individual explanations.")

        # Employee feature snapshot
        st.subheader("Feature Snapshot")
        feature_display = {}
        for f in BASE_FEATURES:
            if f in emp_data.index:
                feature_display[f] = emp_data[f]
        st.dataframe(pd.Series(feature_display, name='Value'), use_container_width=True)


# ============================================================
# TAB 3: MANAGER SCORECARD
# ============================================================
with tab_manager:
    st.subheader("👔 Manager Scorecard")

    if 'Manager ID' in test_df.columns:
        # Get managers with enough reports
        mgr_counts = test_df['Manager ID'].value_counts()
        mgr_with_reports = mgr_counts[mgr_counts >= 2].index.tolist()

        if mgr_with_reports:
            # Show manager selector with risk info
            mgr_risk = test_df.groupby('Manager ID').agg(
                Team_Size=('Employee_ID', 'count'),
                Avg_Risk=('Predicted_Risk', 'mean'),
                High_Risk_Count=('Predicted_Leaver', 'sum'),
            ).reset_index()
            mgr_risk = mgr_risk[mgr_risk['Manager ID'].isin(mgr_with_reports)]
            mgr_risk = mgr_risk.sort_values('Avg_Risk', ascending=False)

            selected_mgr = st.selectbox(
                "Select Manager",
                mgr_risk['Manager ID'].values,
                format_func=lambda x: (
                    f"{x} | Team: {mgr_risk[mgr_risk['Manager ID']==x]['Team_Size'].values[0]} | "
                    f"Avg Risk: {mgr_risk[mgr_risk['Manager ID']==x]['Avg_Risk'].values[0]:.1%}"
                )
            )

            if selected_mgr:
                team = test_df[test_df['Manager ID'] == selected_mgr].copy()
                mgr_info = mgr_risk[mgr_risk['Manager ID'] == selected_mgr].iloc[0]

                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("Team Size", int(mgr_info['Team_Size']))
                col_m2.metric("Avg Risk", f"{mgr_info['Avg_Risk']:.1%}")
                col_m3.metric("High Risk Members", int(mgr_info['High_Risk_Count']))

                mgr_perf = team.get('Manager_Performance_Score', pd.Series([3]))
                col_m4.metric("Manager Perf Score", f"{mgr_perf.iloc[0]:.0f}")

                st.markdown("---")

                # Team risk distribution
                st.subheader("Team Risk Distribution")
                fig_team, ax_team = plt.subplots(figsize=(10, 3))
                fig_team.patch.set_facecolor('#0e1117')
                ax_team.set_facecolor('#0e1117')
                ax_team.hist(team['Predicted_Risk'], bins=20, color='#e94560',
                            alpha=0.7, edgecolor='#0e1117')
                ax_team.axvline(threshold, color='#00d4aa', linestyle='--',
                               linewidth=2, label=f'Threshold {threshold}')
                ax_team.set_xlabel("Risk Score", color='white')
                ax_team.set_ylabel("Count", color='white')
                ax_team.tick_params(colors='white')
                ax_team.legend(facecolor='#1a1a2e', edgecolor='#1f4068', labelcolor='white')
                for spine in ax_team.spines.values():
                    spine.set_color('#1f4068')
                st.pyplot(fig_team)

                # Team member table
                st.subheader("Team Members")
                team_cols = ['Employee_ID', 'Predicted_Risk', 'Management_Level_Agg',
                             'Total_Tenure_Years', 'Performance_Score', 'Predicted_Leaver']
                team_cols = [c for c in team_cols if c in team.columns]
                st.dataframe(
                    team[team_cols].sort_values('Predicted_Risk', ascending=False),
                    use_container_width=True
                )
        else:
            st.info("No managers with 2+ reports found in filtered data.")
    else:
        st.info("Manager ID column not available in data.")


# ============================================================
# TAB 4: SURVIVAL CURVES
# ============================================================
with tab_survival:
    st.subheader("📈 Retention Curves (Kaplan-Meier)")

    if HAS_LIFELINES and 'Total_Tenure_Years' in test_df.columns:
        km_group = st.selectbox(
            "Group By",
            ['Management_Level_Agg', 'Region', 'Level 4 Org']
        )

        if km_group in test_df.columns:
            fig_km, ax_km = plt.subplots(figsize=(10, 6))
            fig_km.patch.set_facecolor('#0e1117')
            ax_km.set_facecolor('#0e1117')

            kmf = KaplanMeierFitter()
            groups = test_df[km_group].dropna().unique()
            colors_km = plt.cm.Set2(np.linspace(0, 1, len(groups)))

            for i, group in enumerate(groups):
                mask = test_df[km_group] == group
                group_data = test_df[mask]
                if len(group_data) > 10:
                    durations = group_data['Total_Tenure_Years'].clip(lower=0.01)
                    events = group_data[TARGET_COLUMN] if TARGET_COLUMN in group_data.columns else pd.Series(0, index=group_data.index)
                    try:
                        kmf.fit(durations, event_observed=events, label=str(group))
                        kmf.plot_survival_function(ax=ax_km, ci_show=False,
                                                   color=colors_km[i])
                    except Exception:
                        continue

            ax_km.set_xlabel("Tenure (Years)", color='white')
            ax_km.set_ylabel("Retention Probability", color='white')
            ax_km.set_title(f"Retention by {km_group}", color='white')
            ax_km.tick_params(colors='white')
            ax_km.legend(facecolor='#1a1a2e', edgecolor='#1f4068',
                        labelcolor='white', fontsize=8, loc='lower left')
            for spine in ax_km.spines.values():
                spine.set_color('#1f4068')
            st.pyplot(fig_km)
        else:
            st.info(f"Column '{km_group}' not found in data.")
    else:
        st.info("Install `lifelines` and ensure tenure data exists for survival curves.")


# ============================================================
# TAB 5: FAIRNESS AUDIT
# ============================================================
with tab_fairness:
    st.subheader("⚖️ Fairness & Bias Audit")
    st.caption("Checks if the model disproportionately flags certain protected groups.")

    fairness_df = load_fairness_report()
    if fairness_df is not None and not fairness_df.empty:
        # Overall stats
        overall_flag_rate = y_pred.mean()
        st.metric("Overall Flag Rate", f"{overall_flag_rate:.1%}")

        # Group-level analysis
        for attr in fairness_df['Protected_Attribute'].unique():
            attr_data = fairness_df[fairness_df['Protected_Attribute'] == attr]
            if len(attr_data) < 2:
                continue

            st.subheader(f"📊 {attr}")

            fig_fair, ax_fair = plt.subplots(figsize=(10, 4))
            fig_fair.patch.set_facecolor('#0e1117')
            ax_fair.set_facecolor('#0e1117')

            bar_colors = []
            for _, row in attr_data.iterrows():
                if row.get('Rate_vs_Overall', 1) > 1.2:
                    bar_colors.append('#e94560')  # Biased
                elif row.get('Rate_vs_Overall', 1) < 0.8:
                    bar_colors.append('#f5a623')  # Under-represented
                else:
                    bar_colors.append('#00d4aa')  # Fair

            ax_fair.barh(
                attr_data['Group'].values[::-1],
                attr_data['Flag_Rate'].values[::-1],
                color=bar_colors[::-1], edgecolor='white', linewidth=0.5
            )
            ax_fair.axvline(overall_flag_rate, color='white', linestyle='--',
                          linewidth=1, label=f'Overall: {overall_flag_rate:.1%}')
            ax_fair.set_xlabel("Flag Rate", color='white')
            ax_fair.tick_params(colors='white')
            ax_fair.legend(facecolor='#1a1a2e', edgecolor='#1f4068', labelcolor='white')
            for spine in ax_fair.spines.values():
                spine.set_color('#1f4068')
            st.pyplot(fig_fair)

            st.dataframe(attr_data[['Group', 'N', 'Flag_Rate', 'Avg_Risk_Score',
                                    'Rate_vs_Overall']],
                        use_container_width=True)

        # Bias warnings
        biased = fairness_df[fairness_df.get('Rate_vs_Overall', pd.Series()) > 1.20]
        if not biased.empty:
            st.warning("⚠️ **Potential Bias Detected** — The following groups are flagged at >20% higher rate than average:")
            for _, row in biased.iterrows():
                st.write(f"  - **{row['Protected_Attribute']} = {row['Group']}**: "
                        f"Flag rate {row['Flag_Rate']:.1%} ({row['Rate_vs_Overall']:.1f}x overall)")
        else:
            st.success("✅ No significant bias detected across protected groups.")
    else:
        st.info("Run the full pipeline to generate the fairness audit report.")

        # In-dashboard fairness check (basic)
        st.markdown("---")
        st.subheader("Quick In-Dashboard Check")
        for col in ['Gender', 'Generation', 'US Ethnicity Short Name']:
            if col in test_df.columns:
                group_stats = test_df.groupby(col).agg(
                    N=('Employee_ID', 'count'),
                    Avg_Risk=('Predicted_Risk', 'mean'),
                    Flagged=('Predicted_Leaver', 'sum'),
                ).reset_index()
                group_stats['Flag_Rate'] = group_stats['Flagged'] / group_stats['N']
                st.write(f"**{col}:**")
                st.dataframe(group_stats, use_container_width=True)
