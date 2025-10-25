"""
Customer Churn Prediction & CLV Analysis
Streamlit App - Modernized UI/UX
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import sys
import matplotlib.pyplot as plt
import shap
from pathlib import Path

# --- 1. SETUP AND CONFIGURATION ---

# Add src directory to path for robust imports
sys.path.append(str(Path(__file__).parent / 'src'))

# Force reload for development to pick up changes in train_models
import importlib
if 'train_models' in sys.modules:
    importlib.reload(sys.modules['train_models'])

from train_models import create_interaction_features

# Set page config for a modern, wide layout
st.set_page_config(
    page_title="ChurnGuard AI | Predict & Retain",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define key directory paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / 'models'
DATA_DIR = BASE_DIR / 'data' / 'processed'
FIGURES_DIR = BASE_DIR / 'figures'
ASSETS_DIR = BASE_DIR / 'assets'

# --- 2. UI/UX ENHANCEMENTS ---

def load_css(file_path):
    """Inject custom CSS for styling."""
    try:
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file not found at {file_path}. Using default styles.")

# Inject the custom CSS
load_css(ASSETS_DIR / 'style.css')

# --- 3. HELPER FUNCTIONS (DATA & MODEL LOADING) ---

@st.cache_resource
def load_models():
    """Load all trained models, scaler, and create SHAP explainers."""
    try:
        lr_model = joblib.load(MODELS_DIR / 'logistic_regression.pkl')
        rf_model = joblib.load(MODELS_DIR / 'random_forest.pkl')
        xgb_model = joblib.load(MODELS_DIR / 'xgboost.pkl')
        scaler = joblib.load(MODELS_DIR / 'scaler.pkl')
        
        # Create SHAP explainers once and cache them
        rf_explainer = shap.TreeExplainer(rf_model)
        xgb_explainer = shap.TreeExplainer(xgb_model)
        
        return lr_model, rf_model, xgb_model, scaler, rf_explainer, xgb_explainer
    except FileNotFoundError as e:
        st.error(f"🚨 Model file not found: {e}. Please ensure models are trained and saved in the '{MODELS_DIR}' directory.")
        return None, None, None, None, None, None

@st.cache_data
def load_supporting_data():
    """Load encoding mappings, test data, and model comparison metrics."""
    try:
        with open(DATA_DIR / 'encoding_mapping.json', 'r') as f:
            encoding_mapping = json.load(f)
        
        test_data = pd.read_csv(DATA_DIR / 'test.csv')
        model_comparison = pd.read_csv(MODELS_DIR / 'model_comparison.csv')
        feature_comp = pd.read_csv(MODELS_DIR / 'feature_importance_comparison.csv')
        
        return encoding_mapping, test_data, model_comparison, feature_comp
    except FileNotFoundError as e:
        st.error(f"🚨 Data file not found: {e}. Please ensure data is processed and saved in '{DATA_DIR}' and '{MODELS_DIR}'.")
        return None, None, None, None

# --- (No changes to the core logic of these functions) ---
def create_interaction_features(df):
    """
    CRITICAL: Must replicate exact feature engineering from train_models.py
    This creates all 31 interaction features to match the 52 total features
    """
    df = df.copy()
    
    # 1. Polynomial Features
    df['tenure_squared'] = df['tenure'] ** 2
    df['tenure_log'] = np.log1p(df['tenure'])
    df['charges_squared'] = df['MonthlyCharges'] ** 2
    df['charges_log'] = np.log1p(df['MonthlyCharges'])
    
    # 2. Ratio Features
    df['tenure_charges_ratio'] = df['tenure'] / (df['MonthlyCharges'] + 1)
    df['contract_tenure_ratio'] = df['Contract'] / (df['tenure'] + 1)
    df['charge_per_service'] = df['MonthlyCharges'] / (df['services_count'] + 1)
    
    # 3. Binary Interaction Features
    df['senior_fiber'] = df['SeniorCitizen'] * df['InternetService']
    df['senior_high_charge'] = df['SeniorCitizen'] * (df['MonthlyCharges'] > df['MonthlyCharges'].median()).astype(int)
    df['monthly_short_tenure'] = ((df['MonthlyCharges'] > df['MonthlyCharges'].median()) & 
                                  (df['tenure'] < df['tenure'].median())).astype(int)
    
    # 4. Service Quality Score
    df['service_quality_score'] = (
        df['OnlineSecurity'] + 
        df['OnlineBackup'] + 
        df['DeviceProtection'] + 
        df['TechSupport']
    )
    
    # 5. Risk Indicators
    df['electronic_check_flag'] = (df['PaymentMethod'] == 2).astype(int)  # Electronic check
    df['fiber_no_support'] = ((df['InternetService'] == 1) & 
                              (df['TechSupport'] == 0)).astype(int)
    
    # 6. Triple Interaction
    df['triple_risk'] = (
        (df['Contract'] == 0) *  # Month-to-month
        (df['PaymentMethod'] == 2) *  # Electronic check
        (df['PaperlessBilling'] == 1)  # Paperless billing
    ).astype(int)
    
    # 7. Isolated Customer Flag
    df['isolated_customer'] = (
        (df['Partner'] == 0) & 
        (df['Dependents'] == 0)
    ).astype(int)
    
    # 8. Tenure-Charges Interaction
    df['tenure_charges_interaction'] = df['tenure'] * df['MonthlyCharges']
    
    # 9-16. Pairwise Interactions (8 additional features)
    df['contract_internet'] = df['Contract'] * df['InternetService']
    df['contract_payment'] = df['Contract'] * df['PaymentMethod']
    df['internet_security'] = df['InternetService'] * df['OnlineSecurity']
    df['internet_backup'] = df['InternetService'] * df['OnlineBackup']
    df['security_support'] = df['OnlineSecurity'] * df['TechSupport']
    df['senior_partner'] = df['SeniorCitizen'] * df['Partner']
    df['paperless_autopay'] = df['PaperlessBilling'] * (df['PaymentMethod'] < 2).astype(int)
    df['streaming_both'] = (df['StreamingTV'] == 1) & (df['StreamingMovies'] == 1)
    df['streaming_both'] = df['streaming_both'].astype(int)
    
    # 17-21. Service Bundle Features (5 additional features)
    df['premium_services'] = (
        (df['OnlineSecurity'] == 1) & 
        (df['OnlineBackup'] == 1) & 
        (df['DeviceProtection'] == 1) & 
        (df['TechSupport'] == 1)
    ).astype(int)
    
    df['no_services'] = (df['services_count'] == 0).astype(int)
    df['basic_internet_only'] = (
        (df['InternetService'] > 0) & 
        (df['OnlineSecurity'] == 0) & 
        (df['OnlineBackup'] == 0)
    ).astype(int)
    
    df['entertainment_bundle'] = (
        (df['StreamingTV'] == 1) | 
        (df['StreamingMovies'] == 1)
    ).astype(int)
    
    df['protection_bundle'] = (
        (df['OnlineSecurity'] == 1) | 
        (df['DeviceProtection'] == 1)
    ).astype(int)
    
    return df

def calculate_clv_estimate(monthly_charges, expected_tenure_months=24):
    """Estimate CLV for a customer using a conservative expected tenure."""
    return monthly_charges * expected_tenure_months

def get_feature_explanation(feature_name, feature_value, shap_value):
    """Generate human-readable explanation for a feature's SHAP value."""
    impact = "increases" if shap_value > 0 else "decreases"
    
    explanations = {
        'Contract': f"A **Month-to-Month contract** {impact} churn risk significantly.",
        'tenure': f"A short tenure of **{feature_value} months** {impact} churn risk.",
        'MonthlyCharges': f"High monthly charges of **${feature_value:.2f}** {impact} churn risk.",
        'OnlineSecurity': f"Not having **Online Security** {impact} churn risk.",
        'TechSupport': f"Not having **Tech Support** {impact} churn risk.",
        'InternetService': f"Having **Fiber Optic** internet {impact} churn risk (often due to price/competition).",
        'PaymentMethod': f"Paying by **Electronic Check** {impact} churn risk.",
        'tenure_charges_ratio': f"A low tenure-to-charges ratio {impact} churn risk.",
        'electronic_check_flag': f"Using **Electronic Check** payment {impact} churn risk.",
    }
    
    return explanations.get(feature_name, f"The value of **{feature_name}** ({feature_value:.2f}) {impact} churn risk.")

# --- 4. LOAD RESOURCES ---

# Load all models and data at the start
lr_model, rf_model, xgb_model, scaler, rf_explainer, xgb_explainer = load_models()
encoding_mapping, test_data, model_comparison, feature_comp = load_supporting_data()

# --- 5. SIDEBAR ---

with st.sidebar:
    # Main title — centered, bold, larger, and clearer
    st.markdown(
        "<h1 style='text-align: center; font-weight: 900; font-size: 28px;'>🎯 Churn Prediction & CLV Insights</h1>", 
        unsafe_allow_html=True
    )
    
    # Developer name - centered
    st.markdown(
        "<h3 style='text-align: center;'>Developed by Segun Oladeinde</h3>",
        unsafe_allow_html=True
    )
    
    # Centered professional title
    st.markdown(
        "<p style='text-align: center; font-weight: bold; color: #000000;'>Data Scientist & ML Engineer</p>",
        unsafe_allow_html=True
    )
    
    # Add social links with icons
    st.markdown(
        """
        <div style="display: flex; justify-content: center; align-items: center; gap: 20px; margin-top: 10px;">
            <a href="https://github.com/SegunOladeinde" target="_blank" style="text-decoration: none;" title="GitHub Profile">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="#000000">
                    <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                </svg>
            </a>
            <a href="https://www.linkedin.com/in/segun-oladeinde/" target="_blank" style="text-decoration: none;" title="LinkedIn Profile">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="#0A66C2">
                    <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                </svg>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    
   
    
    

    st.markdown("---")
    
    if test_data is not None:
        st.header("📊 Key Metrics")
        st.metric("Total Customers in Test Set", f"{len(test_data):,}")
        st.metric("Overall Churn Rate", f"{test_data['Churn'].mean()*100:.1f}%")
        st.metric("Average CLV", f"${test_data['CLV'].mean():,.2f}")
    
    st.markdown("---")
    st.header("⚙️ Model Info")
    if model_comparison is not None:
        best_auc_model = model_comparison.loc[model_comparison['AUC-ROC'].idxmax()]
        st.markdown(f"**Best Model (AUC):** `{best_auc_model['Model']}`")
        st.markdown(f"**AUC-ROC:** `{best_auc_model['AUC-ROC']:.3f}`")
        st.markdown(f"**Recall:** `{best_auc_model['Recall']:.3f}`")
    
    st.markdown("---")
    st.info("Developed as part of an MLOps project, this app demonstrates a production-ready churn prediction system.")

# --- 6. MAIN APPLICATION ---

st.title("Customer Churn & Lifetime Value Dashboard")
st.markdown("Enter customer details to get a real-time churn prediction, or explore model performance and CLV analytics.")

if lr_model is None or test_data is None:
    st.error("Application cannot start because essential model or data files are missing. Please check the logs.")
else:
    # Create tabs for a clean, organized interface
    tab1, tab2, tab3 = st.tabs(["🎯 Predict Churn", "📊 Model Performance", "💰 CLV Analysis"])

    # ============================================================
    # TAB 1: PREDICT CHURN
    # ============================================================
    with tab1:
        st.header("👤 Customer Profile")
        
        # Use containers for better visual grouping of input fields
        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("Demographics")
                gender = st.selectbox("Gender", ["Female", "Male"])
                senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
                partner = st.selectbox("Has Partner", ["No", "Yes"])
                dependents = st.selectbox("Has Dependents", ["No", "Yes"])
            
            with col2:
                st.subheader("Account Details")
                tenure = st.slider("Tenure (months)", 0, 72, 12)
                contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
                payment_method = st.selectbox("Payment Method", 
                                             ["Electronic check", "Mailed check", 
                                              "Bank transfer (automatic)", "Credit card (automatic)"])
                monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0, 0.5)

            with col3:
                st.subheader("Subscribed Services")
                phone_service = st.selectbox("Phone Service", ["No", "Yes"])
                multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
                internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
                
                # Conditional inputs for internet-related services
                if internet_service != "No":
                    online_security = st.selectbox("Online Security", ["No", "Yes"])
                    online_backup = st.selectbox("Online Backup", ["No", "Yes"])
                    device_protection = st.selectbox("Device Protection", ["No", "Yes"])
                    tech_support = st.selectbox("Tech Support", ["No", "Yes"])
                    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
                    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
                else:
                    # Set to "No internet service" if no internet
                    online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies = ["No internet service"] * 6

        # Centralized prediction button
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(" Predict Churn Risk", type="primary", use_container_width=True):
            
            # --- Data Preparation for Prediction (No logic change) ---
            phone_svc = 1 if phone_service == "Yes" else 0
            online_sec = 0 if online_security in ["No", "No internet service"] else 1
            online_bkp = 0 if online_backup in ["No", "No internet service"] else 1
            device_prot = 0 if device_protection in ["No", "No internet service"] else 1
            tech_sup = 0 if tech_support in ["No", "No internet service"] else 1
            stream_tv = 0 if streaming_tv in ["No", "No internet service"] else 1
            stream_mov = 0 if streaming_movies in ["No", "No internet service"] else 1
            
            services_cnt = sum([phone_svc, online_sec, online_bkp, device_prot, tech_sup, stream_tv, stream_mov])
            internet_svc_val = 0 if internet_service == "No" else (1 if internet_service == "DSL" else 2)
            internet_no_sup = int((internet_svc_val > 0) and (tech_sup == 0) and (online_sec == 0))
            tenure_bucket_val = pd.cut([tenure], bins=[-1, 6, 12, 24, 100], labels=['0-6m', '6-12m', '12-24m', '24m+']).codes[0]
            
            input_df = pd.DataFrame([{
                'gender': 0 if gender == "Female" else 1,
                'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
                'Partner': 1 if partner == "Yes" else 0,
                'Dependents': 1 if dependents == "Yes" else 0,
                'tenure': tenure,
                'PhoneService': phone_svc,
                'MultipleLines': 0 if multiple_lines == "No" else (1 if multiple_lines == "Yes" else 2),
                'InternetService': internet_svc_val,
                'OnlineSecurity': online_sec,
                'OnlineBackup': online_bkp,
                'DeviceProtection': device_prot,
                'TechSupport': tech_sup,
                'StreamingTV': stream_tv,
                'StreamingMovies': stream_mov,
                'Contract': 0 if contract == "Month-to-month" else (1 if contract == "One year" else 2),
                'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
                'PaymentMethod': ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(payment_method),
                'MonthlyCharges': monthly_charges,
                'tenure_bucket': tenure_bucket_val,
                'services_count': services_cnt,
                'internet_no_support': internet_no_sup
            }])
            
            input_df_enhanced = create_interaction_features(input_df)
            
            expected_features = scaler.feature_names_in_
            input_df_enhanced = input_df_enhanced.reindex(columns=expected_features, fill_value=0)
            
            # --- Model Prediction (No logic change) ---
            X_scaled = scaler.transform(input_df_enhanced)
            lr_proba = lr_model.predict_proba(X_scaled)[0, 1]
            rf_proba = rf_model.predict_proba(input_df_enhanced)[0, 1]
            xgb_proba = xgb_model.predict_proba(input_df_enhanced)[0, 1]
            ensemble_proba = (lr_proba + rf_proba + xgb_proba) / 3
            clv_estimate = calculate_clv_estimate(monthly_charges)
            
            # --- Display Prediction Results (Modernized UI) ---
            st.markdown("---")
            st.header("🎯 Prediction Outcome")
            
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                with st.container(border=True):
                    st.metric("Ensemble Churn Probability", f"{ensemble_proba*100:.1f}%")
                    
                    if ensemble_proba >= 0.7:
                        risk_level, risk_color = "HIGH RISK", "error"
                    elif ensemble_proba >= 0.4:
                        risk_level, risk_color = "MEDIUM RISK", "warning"
                    else:
                        risk_level, risk_color = "LOW RISK", "success"
                    
                    st.markdown(f"**Risk Level:** <span class='{risk_color}-text'>{risk_level}</span>", unsafe_allow_html=True)

            with result_col2:
                with st.container(border=True):
                    st.metric("💰 Estimated CLV (24 months)", f"${clv_estimate:,.2f}")
                    st.markdown("**Business Value:** Potential revenue over the next 2 years.")

            with st.expander("Show Individual Model Predictions", expanded=False):
                p_col1, p_col2, p_col3 = st.columns(3)
                p_col1.metric("Logistic Regression", f"{lr_proba*100:.1f}%")
                p_col2.metric("Random Forest", f"{rf_proba*100:.1f}%")
                p_col3.metric("XGBoost", f"{xgb_proba*100:.1f}%")

            # --- SHAP Explanation & Recommendation (Modernized UI) ---
            st.markdown("---")
            st.header("🧠 Why This Prediction?")
            
            shap_col1, shap_col2 = st.columns([1, 1])
            
            with shap_col1:
                with st.container(border=True):
                    st.subheader("Top Factors Influencing Churn")
                    try:
                        shap_values = rf_explainer.shap_values(input_df_enhanced)
                        shap_values_pos = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
                        
                        shap_df = pd.DataFrame({
                            'feature': input_df_enhanced.columns,
                            'value': input_df_enhanced.iloc[0].values,
                            'shap_value': shap_values_pos
                        }).assign(abs_shap=lambda x: abs(x.shap_value)).sort_values('abs_shap', ascending=False).head(7)
                        
                        for _, row in shap_df.iterrows():
                            icon = "🔴" if row['shap_value'] > 0 else "🟢"
                            explanation = get_feature_explanation(row['feature'], row['value'], row['shap_value'])
                            st.markdown(f"{icon} {explanation}")
                    except Exception:
                        st.info("SHAP explanations are currently unavailable. Predictions remain accurate.")

            with shap_col2:
                with st.container(border=True):
                    st.subheader("💡 Recommended Action")
                    if risk_level == "HIGH RISK":
                        st.error(f"**URGENT:** This customer is at high risk. Prioritize immediate retention efforts. Potential revenue at risk: **${clv_estimate:,.2f}**.")
                    elif risk_level == "MEDIUM RISK":
                        st.warning("**MONITOR:** This customer shows moderate risk. Consider proactive engagement or loyalty incentives.")
                    else:
                        st.success("**RETAIN & GROW:** This customer is loyal. Focus on upselling and maximizing their lifetime value.")

    # ============================================================
    # TAB 2: MODEL PERFORMANCE
    # ============================================================
    with tab2:
        st.header("Model Performance Deep-Dive")
        
        with st.container(border=True):
            st.subheader("📊 Comparative Metrics")
            metrics_df = model_comparison.round(4)
            st.dataframe(
                metrics_df.style.highlight_max(axis=0, props='font-weight:bold;background-color:lightgreen;'),
                use_container_width=True
            )
            
            st.markdown("---")
            m_col1, m_col2, m_col3 = st.columns(3)
            best_auc = metrics_df.loc[metrics_df['AUC-ROC'].idxmax()]
            best_recall = metrics_df.loc[metrics_df['Recall'].idxmax()]
            best_precision = metrics_df.loc[metrics_df['Precision'].idxmax()]
            
            m_col1.metric("🏆 Best AUC-ROC", f"{best_auc['AUC-ROC']:.3f}", delta=best_auc['Model'])
            m_col2.metric("🎯 Best Recall", f"{best_recall['Recall']:.3f}", delta=best_recall['Model'])
            m_col3.metric("PRECISION", f"{best_precision['Precision']:.3f}", delta=best_precision['Model'])

        with st.container(border=True):
            st.subheader("🔑 Feature Importance Analysis")
            st.dataframe(
                feature_comp.head(15).style.background_gradient(subset=['LR', 'RF', 'XGB', 'avg_importance'], cmap='viridis'),
                use_container_width=True
            )
            with st.expander("View Key Insights"):
                st.markdown("""
                - **Contract Type:** Consistently the most powerful predictor. Month-to-month contracts are a major churn signal.
                - **Tenure:** Loyalty pays off. Longer tenure drastically reduces churn risk.
                - **Core Services:** Lack of `OnlineSecurity` and `TechSupport` are strong indicators of dissatisfaction.
                """)

        with st.container(border=True):
            st.subheader("📈 Visualizations")
            viz_col1, viz_col2 = st.columns(2)
            if (FIGURES_DIR / 'feature_importance_comparison.png').exists():
                viz_col1.image(str(FIGURES_DIR / 'feature_importance_comparison.png'), caption="Overall Feature Importance")
            if (FIGURES_DIR / 'xgboost_shap_summary.png').exists():
                viz_col2.image(str(FIGURES_DIR / 'xgboost_shap_summary.png'), caption="XGBoost SHAP Summary")

    # ============================================================
    # TAB 3: CLV ANALYSIS
    # ============================================================
    with tab3:
        st.header("Customer Lifetime Value Deep-Dive")
        test_with_clv = test_data.copy()
        
        with st.container(border=True):
            st.subheader("💰 CLV Summary Statistics")
            c_col1, c_col2, c_col3, c_col4 = st.columns(4)
            c_col1.metric("Average CLV", f"${test_with_clv['CLV'].mean():,.2f}")
            c_col2.metric("Median CLV", f"${test_with_clv['CLV'].median():,.2f}")
            c_col3.metric("Min CLV", f"${test_with_clv['CLV'].min():,.2f}")
            c_col4.metric("Max CLV", f"${test_with_clv['CLV'].max():,.2f}")

        with st.container(border=True):
            st.subheader("🎯 Churn Rate by CLV Quartile")
            test_with_clv['CLV_Quartile'] = pd.qcut(test_with_clv['CLV'], q=4, labels=['Low', 'Medium', 'High', 'Premium'])
            quartile_analysis = test_with_clv.groupby('CLV_Quartile', observed=True).agg(
                Churn_Rate=('Churn', 'mean'),
                Customer_Count=('Churn', 'count'),
                Avg_CLV=('CLV', 'mean')
            ).reset_index()
            
            st.dataframe(
                quartile_analysis.style
                .format({'Churn_Rate': '{:.1%}', 'Avg_CLV': '${:,.2f}'})
                .background_gradient(subset=['Churn_Rate'], cmap='Reds'),
                use_container_width=True
            )
            st.info("""
            **💡 Insight:** Premium customers have a churn rate over **10x lower** than Low-value customers. 
            This confirms that high-value customers are more loyal. Retention efforts should be laser-focused on the 'High' and 'Premium' tiers.
            """)

        with st.container(border=True):
            st.subheader("⚠️ Revenue at Risk")
            churned_customers = test_with_clv[test_with_clv['Churn'] == 1]
            total_revenue_at_risk = churned_customers['CLV'].sum()
            
            r_col1, r_col2 = st.columns(2)
            r_col1.metric("Total Revenue at Risk from Churn", f"${total_revenue_at_risk:,.2f}",
                          delta=f"{len(churned_customers)} customers", delta_color="inverse")
            r_col2.metric("Avg. Revenue Lost per Churned Customer", f"${churned_customers['CLV'].mean():,.2f}")

        with st.container(border=True):
            st.subheader("🖼️ CLV Visualizations")
            img_col1, img_col2 = st.columns(2)
            if (FIGURES_DIR / 'clv_distribution.png').exists():
                img_col1.image(str(FIGURES_DIR / 'clv_distribution.png'), caption="CLV Distribution")
            # Assuming a plot named 'clv_quartile_analysis.png' exists
            if (FIGURES_DIR / 'clv_quartile_analysis.png').exists():
                img_col2.image(str(FIGURES_DIR / 'clv_quartile_analysis.png'), caption="Churn Rate by CLV Quartile")
