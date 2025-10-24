"""
Prediction Module for Customer Churn
Handles loading models and making predictions on new customer data
"""

import pandas as pd
import numpy as np
import joblib
import os
from train_models import create_interaction_features

# Paths
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')


def load_models():
    """
    Load all trained models and preprocessing objects
    
    Returns:
        tuple: (lr_model, rf_model, xgb_model, scaler)
    """
    lr_model = joblib.load(os.path.join(MODELS_DIR, 'logistic_regression.pkl'))
    rf_model = joblib.load(os.path.join(MODELS_DIR, 'random_forest.pkl'))
    xgb_model = joblib.load(os.path.join(MODELS_DIR, 'xgboost.pkl'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    
    return lr_model, rf_model, xgb_model, scaler


def prepare_input_data(customer_data):
    """
    Prepare customer data for prediction by creating all necessary features
    
    Args:
        customer_data (dict): Dictionary with customer information
        Expected keys: gender, SeniorCitizen, Partner, Dependents, tenure,
                      PhoneService, MultipleLines, InternetService, OnlineSecurity,
                      OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
                      StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
                      MonthlyCharges, services_count, internet_no_support
        
    Returns:
        pd.DataFrame: DataFrame with all 52 features in correct order
    """
    # Calculate tenure_bucket
    tenure_bucket_val = pd.cut(
        [customer_data['tenure']],
        bins=[-1, 6, 12, 24, 100],
        labels=['0-6m', '6-12m', '12-24m', '24m+']
    ).codes[0]
    
    # Build DataFrame with EXACT 21 columns used during training
    # (after dropping TotalCharges, CLV, ExpectedTenure, monthly_to_total_ratio)
    input_df = pd.DataFrame([{
        'gender': customer_data['gender'],
        'SeniorCitizen': customer_data['SeniorCitizen'],
        'Partner': customer_data['Partner'],
        'Dependents': customer_data['Dependents'],
        'tenure': customer_data['tenure'],
        'PhoneService': customer_data['PhoneService'],
        'MultipleLines': customer_data['MultipleLines'],
        'InternetService': customer_data['InternetService'],
        'OnlineSecurity': customer_data['OnlineSecurity'],
        'OnlineBackup': customer_data['OnlineBackup'],
        'DeviceProtection': customer_data['DeviceProtection'],
        'TechSupport': customer_data['TechSupport'],
        'StreamingTV': customer_data['StreamingTV'],
        'StreamingMovies': customer_data['StreamingMovies'],
        'Contract': customer_data['Contract'],
        'PaperlessBilling': customer_data['PaperlessBilling'],
        'PaymentMethod': customer_data['PaymentMethod'],
        'MonthlyCharges': customer_data['MonthlyCharges'],
        'tenure_bucket': tenure_bucket_val,
        'services_count': customer_data['services_count'],
        'internet_no_support': customer_data['internet_no_support']
    }])
    
    # Apply interaction features (adds 31 more features: 21 -> 52)
    input_df_enhanced = create_interaction_features(input_df)
    
    # Load scaler and reorder columns to match expected order
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    input_df_enhanced = input_df_enhanced[scaler.feature_names_in_]
    
    return input_df_enhanced


def predict_churn(customer_data):
    """
    Predict churn probability for a customer
    
    Args:
        customer_data (dict): Dictionary with customer information
            Required keys: gender, SeniorCitizen, Partner, Dependents, tenure,
            PhoneService, MultipleLines, InternetService, OnlineSecurity,
            OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
            StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
            MonthlyCharges, services_count, internet_no_support
            
    Returns:
        dict: Prediction results with keys:
            - lr_proba: Logistic Regression probability
            - rf_proba: Random Forest probability
            - xgb_proba: XGBoost probability
            - ensemble_proba: Average probability
            - risk_level: 'HIGH', 'MEDIUM', or 'LOW'
    """
    # Load models
    lr_model, rf_model, xgb_model, scaler = load_models()
    
    # Prepare input data
    input_df_enhanced = prepare_input_data(customer_data)
    
    # Make predictions
    # Logistic Regression uses scaled features
    X_scaled = scaler.transform(input_df_enhanced)
    lr_proba = lr_model.predict_proba(X_scaled)[0, 1]
    
    # Random Forest and XGBoost use unscaled features
    rf_proba = rf_model.predict_proba(input_df_enhanced)[0, 1]
    xgb_proba = xgb_model.predict_proba(input_df_enhanced)[0, 1]
    
    # Ensemble prediction (average)
    ensemble_proba = (lr_proba + rf_proba + xgb_proba) / 3
    
    # Determine risk level
    if ensemble_proba >= 0.7:
        risk_level = 'HIGH'
    elif ensemble_proba >= 0.4:
        risk_level = 'MEDIUM'
    else:
        risk_level = 'LOW'
    
    return {
        'lr_proba': lr_proba,
        'rf_proba': rf_proba,
        'xgb_proba': xgb_proba,
        'ensemble_proba': ensemble_proba,
        'risk_level': risk_level
    }


def calculate_clv(monthly_charges, expected_tenure_months=24):
    """
    Calculate Customer Lifetime Value estimate
    
    Args:
        monthly_charges (float): Monthly charges for the customer
        expected_tenure_months (int): Expected tenure in months (default: 24)
        
    Returns:
        float: Estimated CLV
    """
    return monthly_charges * expected_tenure_months


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("CUSTOMER CHURN PREDICTION - EXAMPLE")
    print("=" * 60)
    
    # High-risk customer example
    high_risk_customer = {
        'gender': 1,  # Male
        'SeniorCitizen': 1,  # Yes
        'Partner': 0,  # No
        'Dependents': 0,  # No
        'tenure': 1,  # 1 month (new customer)
        'PhoneService': 1,  # Yes
        'MultipleLines': 0,  # No
        'InternetService': 2,  # Fiber optic
        'OnlineSecurity': 0,  # No
        'OnlineBackup': 0,  # No
        'DeviceProtection': 0,  # No
        'TechSupport': 0,  # No
        'StreamingTV': 0,  # No
        'StreamingMovies': 0,  # No
        'Contract': 0,  # Month-to-month
        'PaperlessBilling': 1,  # Yes
        'PaymentMethod': 2,  # Electronic check
        'MonthlyCharges': 85.0,
        'services_count': 1,  # Only phone service
        'internet_no_support': 1  # Has internet but no support
    }
    
    print("\nHigh-Risk Customer Profile:")
    print(f"  Senior Citizen, New Customer (1 month)")
    print(f"  Fiber Optic, No Support Services")
    print(f"  Month-to-month Contract, Electronic Check Payment")
    print(f"  Monthly Charges: ${high_risk_customer['MonthlyCharges']:.2f}")
    
    # Make prediction
    result = predict_churn(high_risk_customer)
    
    print(f"\nPrediction Results:")
    print(f"  Logistic Regression: {result['lr_proba']*100:.1f}%")
    print(f"  Random Forest: {result['rf_proba']*100:.1f}%")
    print(f"  XGBoost: {result['xgb_proba']*100:.1f}%")
    print(f"  Ensemble (Average): {result['ensemble_proba']*100:.1f}%")
    print(f"  Risk Level: {result['risk_level']}")
    
    # Calculate CLV
    clv = calculate_clv(high_risk_customer['MonthlyCharges'])
    print(f"  Estimated CLV (24 months): ${clv:,.2f}")
    
    print("\n" + "=" * 60)
    print("✅ Prediction completed successfully!")
    print("=" * 60)
