"""
Model Training Script
Trains three churn prediction models:
1. Logistic Regression (baseline)
2. Random Forest
3. XGBoost

Includes light hyperparameter tuning and model evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, roc_curve, precision_recall_curve,
    precision_score, recall_score, f1_score, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Paths
PROCESSED_DATA_DIR = 'data/processed/'
MODELS_DIR = 'models/'
FIGURES_DIR = 'figures/'

# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_data():
    """Load the processed train, validation, and test sets."""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'train.csv'))
    val = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'val.csv'))
    test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'test.csv'))
    
    # Features to exclude from modeling (prevent data leakage)
    # TotalCharges can leak information about churn timing
    # monthly_to_total_ratio uses TotalCharges, so also leaks
    exclude_cols = ['Churn', 'CLV', 'ExpectedTenure', 'CLV_Quartile', 'TotalCharges', 'monthly_to_total_ratio']
    
    # Separate features and target
    X_train = train.drop(columns=[col for col in exclude_cols if col in train.columns])
    y_train = train['Churn']
    
    X_val = val.drop(columns=[col for col in exclude_cols if col in val.columns])
    y_val = val['Churn']
    
    X_test = test.drop(columns=[col for col in exclude_cols if col in test.columns])
    y_test = test['Churn']
    
    print(f"\n✅ Data loaded:")
    print(f"   Train: X{X_train.shape}, y{y_train.shape}")
    print(f"   Val:   X{X_val.shape}, y{y_val.shape}")
    print(f"   Test:  X{X_test.shape}, y{y_test.shape}")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Churn rate - Train: {y_train.mean()*100:.2f}%, Val: {y_val.mean()*100:.2f}%, Test: {y_test.mean()*100:.2f}%")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def create_interaction_features(X):
    """
    Create comprehensive interaction features to capture non-linear relationships.
    Advanced feature engineering to push model performance toward 80-85% AUC.
    
    Categories of features:
    1. Risk amplifiers (combinations that multiply churn risk)
    2. Protection factors (combinations that reduce churn)
    3. Behavioral patterns (usage and payment patterns)
    4. Price sensitivity indicators
    
    NOTE: Expects X to already have base engineered features:
    - tenure_bucket (from data_prep.py)
    - services_count (from data_prep.py)
    - internet_no_support (from data_prep.py)
    """
    X_enhanced = X.copy()
    
    # ========== BASIC INTERACTIONS (Original 5) ==========
    # Senior × Fiber Optic (InternetService=1 is Fiber optic)
    X_enhanced['senior_fiber'] = X['SeniorCitizen'] * (X['InternetService'] == 1).astype(int)
    
    # Senior × High Monthly Charges (>$70)
    X_enhanced['senior_high_charge'] = X['SeniorCitizen'] * (X['MonthlyCharges'] > 70).astype(int)
    
    # Month-to-month contract (0) × Short tenure (<12 months)
    X_enhanced['monthly_short_tenure'] = (X['Contract'] == 0) * (X['tenure'] < 12).astype(int)
    
    # Fiber × No Tech Support (frustrated customers)
    X_enhanced['fiber_no_support'] = (X['InternetService'] == 1) * (X['TechSupport'] == 0).astype(int)
    
    # High charges × internet_no_support flag
    X_enhanced['expensive_no_support'] = (X['MonthlyCharges'] > 70) * X['internet_no_support']
    
    # ========== ADVANCED INTERACTIONS (New) ==========
    
    # 1. Contract Type × Tenure Interactions
    X_enhanced['contract_tenure_ratio'] = X['Contract'] * np.log1p(X['tenure'])
    X_enhanced['monthly_new_customer'] = (X['Contract'] == 0) * (X['tenure'] <= 6).astype(int)
    X_enhanced['oneyear_midtenure'] = (X['Contract'] == 1) * ((X['tenure'] >= 12) & (X['tenure'] <= 24)).astype(int)
    
    # 2. Service Bundle Quality Score
    # More services usually mean loyalty, but not if they're having issues
    X_enhanced['service_quality_score'] = (
        X['services_count'] * 
        (1 + X['OnlineSecurity'] + X['OnlineBackup'] + X['TechSupport']) / 4
    )
    
    # 3. Payment Method Risk Flags
    # Electronic check (2) is historically risky
    X_enhanced['electronic_check_flag'] = (X['PaymentMethod'] == 2).astype(int)
    X_enhanced['auto_payment_flag'] = ((X['PaymentMethod'] == 0) | (X['PaymentMethod'] == 1)).astype(int)
    X_enhanced['risky_payment_high_charge'] = (X['PaymentMethod'] == 2) * (X['MonthlyCharges'] > 70).astype(int)
    
    # 4. Price Sensitivity Features
    X_enhanced['charge_per_service'] = X['MonthlyCharges'] / np.maximum(X['services_count'], 1)
    X_enhanced['high_charge_per_service'] = (X_enhanced['charge_per_service'] > 20).astype(int)
    X_enhanced['price_jumper'] = ((X['MonthlyCharges'] > 80) & (X['tenure'] < 12)).astype(int)
    
    # 5. Dependency & Support Patterns
    X_enhanced['isolated_customer'] = ((X['Partner'] == 0) & (X['Dependents'] == 0)).astype(int)
    X_enhanced['isolated_no_support'] = X_enhanced['isolated_customer'] * (X['TechSupport'] == 0)
    X_enhanced['senior_isolated'] = X['SeniorCitizen'] * X_enhanced['isolated_customer']
    
    # 6. Internet Service Quality Indicators
    X_enhanced['fiber_full_services'] = (X['InternetService'] == 1) * (X['services_count'] >= 5).astype(int)
    X_enhanced['internet_minimal_services'] = (
        ((X['InternetService'] == 0) | (X['InternetService'] == 1)) * 
        (X['services_count'] <= 2).astype(int)
    )
    
    # 7. Tenure-based Risk Buckets
    X_enhanced['very_new'] = (X['tenure'] <= 3).astype(int)
    X_enhanced['early_exit_window'] = ((X['tenure'] >= 4) & (X['tenure'] <= 12)).astype(int)
    X_enhanced['established'] = (X['tenure'] > 24).astype(int)
    
    # 8. Critical Risk Combinations (Triple interactions)
    X_enhanced['triple_risk'] = (
        X['SeniorCitizen'] * 
        (X['Contract'] == 0) * 
        (X['InternetService'] == 1)  # Senior + monthly + fiber
    ).astype(int)
    
    X_enhanced['payment_service_risk'] = (
        (X['PaymentMethod'] == 2) *  # Electronic check
        (X['Contract'] == 0) *        # Month-to-month
        (X['tenure'] < 12).astype(int)
    )
    
    # 9. Polynomial Features for Key Variables
    X_enhanced['tenure_squared'] = X['tenure'] ** 2
    X_enhanced['tenure_log'] = np.log1p(X['tenure'])
    X_enhanced['charges_squared'] = X['MonthlyCharges'] ** 2
    X_enhanced['charges_log'] = np.log1p(X['MonthlyCharges'])
    
    # 10. Tenure × Charges Interactions
    X_enhanced['tenure_charges_interaction'] = X['tenure'] * X['MonthlyCharges']
    X_enhanced['tenure_charges_ratio'] = X['tenure'] / np.maximum(X['MonthlyCharges'], 1)
    
    return X_enhanced


def scale_features(X_train, X_val, X_test):
    """
    Scale features for Logistic Regression (tree models don't need scaling).
    
    Returns:
        Scaled datasets and the scaler object
    """
    print("\n" + "=" * 60)
    print("FEATURE SCALING")
    print("=" * 60)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames to keep column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    print("\n✅ Features scaled using StandardScaler")
    print("   (Needed for Logistic Regression, not for tree models)")
    
    # Save scaler
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"   💾 Scaler saved: {scaler_path}")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def evaluate_model(y_true, y_pred, y_pred_proba, model_name="Model"):
    """
    Evaluate model performance and return metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        model_name: Name of the model
        
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'AUC-ROC': roc_auc_score(y_true, y_pred_proba)
    }
    
    return metrics


def train_logistic_regression(X_train, y_train, X_val, y_val):
    """
    Train Logistic Regression with light hyperparameter tuning.
    
    Returns:
        Trained model and validation metrics
    """
    print("\n" + "=" * 60)
    print("TRAINING: LOGISTIC REGRESSION")
    print("=" * 60)
    
    print("\n🔧 Hyperparameter tuning...")
    print("   Testing different C values (regularization strength)")
    
    # More comprehensive C values for better tuning
    c_values = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
    best_auc = 0
    best_c = None
    best_model = None
    
    for c in c_values:
        model = LogisticRegression(
            C=c,
            class_weight='balanced',  # Handle class imbalance
            max_iter=1000,
            random_state=42,
            solver='liblinear'  # Better for small datasets with L1/L2
        )
        model.fit(X_train, y_train)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_val_proba)
        
        print(f"   C={c:5.2f} → AUC-ROC: {auc:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            best_c = c
            best_model = model
    
    print(f"\n✅ Best C: {best_c} (AUC-ROC: {best_auc:.4f})")
    
    # Evaluate on validation set
    y_val_pred = best_model.predict(X_val)
    y_val_proba = best_model.predict_proba(X_val)[:, 1]
    
    metrics = evaluate_model(y_val, y_val_pred, y_val_proba, "Logistic Regression")
    
    print(f"\n📊 Validation Set Performance:")
    for key, value in metrics.items():
        if key != 'Model':
            print(f"   {key:12s}: {value:.4f}")
    
    # Save model
    model_path = os.path.join(MODELS_DIR, 'logistic_regression.pkl')
    joblib.dump(best_model, model_path)
    print(f"\n💾 Model saved: {model_path}")
    
    return best_model, metrics


def train_random_forest(X_train, y_train, X_val, y_val):
    """
    Train Random Forest with light hyperparameter tuning.
    
    Returns:
        Trained model and validation metrics
    """
    print("\n" + "=" * 60)
    print("TRAINING: RANDOM FOREST")
    print("=" * 60)
    
    print("\n🔧 Hyperparameter tuning...")
    print("   Testing combinations of max_depth, min_samples_leaf, and min_samples_split")
    
    # More comprehensive parameter grid
    param_grid = [
        {'max_depth': 8, 'min_samples_leaf': 1, 'min_samples_split': 2},
        {'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 2},
        {'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 5},
        {'max_depth': 15, 'min_samples_leaf': 1, 'min_samples_split': 2},
        {'max_depth': 15, 'min_samples_leaf': 2, 'min_samples_split': 5},
        {'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2},
        {'max_depth': None, 'min_samples_leaf': 2, 'min_samples_split': 5},
    ]
    
    best_auc = 0
    best_params = None
    best_model = None
    
    for params in param_grid:
        model = RandomForestClassifier(
            n_estimators=200,  # Increased from 100 for better performance
            max_depth=params['max_depth'],
            min_samples_leaf=params['min_samples_leaf'],
            min_samples_split=params['min_samples_split'],
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_val_proba)
        
        depth_str = str(params['max_depth']) if params['max_depth'] else "None"
        print(f"   max_depth={depth_str:4s}, min_samples_leaf={params['min_samples_leaf']}, "
              f"min_samples_split={params['min_samples_split']} → AUC-ROC: {auc:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            best_params = params
            best_model = model
    
    print(f"\n✅ Best params: {best_params} (AUC-ROC: {best_auc:.4f})")
    
    # Evaluate on validation set
    y_val_pred = best_model.predict(X_val)
    y_val_proba = best_model.predict_proba(X_val)[:, 1]
    
    metrics = evaluate_model(y_val, y_val_pred, y_val_proba, "Random Forest")
    
    print(f"\n📊 Validation Set Performance:")
    for key, value in metrics.items():
        if key != 'Model':
            print(f"   {key:12s}: {value:.4f}")
    
    # Save model
    model_path = os.path.join(MODELS_DIR, 'random_forest.pkl')
    joblib.dump(best_model, model_path)
    print(f"\n💾 Model saved: {model_path}")
    
    return best_model, metrics


def train_xgboost(X_train, y_train, X_val, y_val):
    """
    Train XGBoost with light hyperparameter tuning.
    
    Returns:
        Trained model and validation metrics
    """
    print("\n" + "=" * 60)
    print("TRAINING: XGBOOST")
    print("=" * 60)
    
    print("\n🔧 Hyperparameter tuning...")
    print("   Testing combinations of max_depth, learning_rate, and n_estimators")
    
    # Calculate scale_pos_weight for class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"   scale_pos_weight: {scale_pos_weight:.2f} (to handle {y_train.mean()*100:.1f}% churn rate)")
    
    # More comprehensive parameter grid
    param_grid = [
        {'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 100},
        {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100},
        {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 200},
        {'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 100},
        {'max_depth': 5, 'learning_rate': 0.1, 'n_estimators': 100},
        {'max_depth': 5, 'learning_rate': 0.1, 'n_estimators': 200},
        {'max_depth': 7, 'learning_rate': 0.05, 'n_estimators': 100},
        {'max_depth': 7, 'learning_rate': 0.1, 'n_estimators': 100},
    ]
    
    best_auc = 0
    best_params = None
    best_model = None
    
    for params in param_grid:
        model = XGBClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss',
            use_label_encoder=False
        )
        model.fit(X_train, y_train)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_val_proba)
        
        print(f"   max_depth={params['max_depth']}, lr={params['learning_rate']:.2f}, "
              f"n_est={params['n_estimators']:3} → AUC-ROC: {auc:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            best_params = params
            best_model = model
    
    print(f"\n✅ Best params: {best_params} (AUC-ROC: {best_auc:.4f})")
    
    # Evaluate on validation set
    y_val_pred = best_model.predict(X_val)
    y_val_proba = best_model.predict_proba(X_val)[:, 1]
    
    metrics = evaluate_model(y_val, y_val_pred, y_val_proba, "XGBoost")
    
    print(f"\n📊 Validation Set Performance:")
    for key, value in metrics.items():
        if key != 'Model':
            print(f"   {key:12s}: {value:.4f}")
    
    # Save model
    model_path = os.path.join(MODELS_DIR, 'xgboost.pkl')
    joblib.dump(best_model, model_path)
    print(f"\n💾 Model saved: {model_path}")
    
    return best_model, metrics


def compare_models(all_metrics):
    """
    Create a comparison table of all models.
    
    Args:
        all_metrics: List of metric dictionaries
    """
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (Validation Set)")
    print("=" * 60)
    
    metrics_df = pd.DataFrame(all_metrics)
    print("\n" + metrics_df.to_string(index=False))
    
    # Save comparison
    comparison_path = os.path.join(MODELS_DIR, 'model_comparison.csv')
    metrics_df.to_csv(comparison_path, index=False)
    print(f"\n💾 Comparison saved: {comparison_path}")
    
    # Highlight best model
    best_auc_idx = metrics_df['AUC-ROC'].idxmax()
    best_model = metrics_df.loc[best_auc_idx, 'Model']
    best_auc = metrics_df.loc[best_auc_idx, 'AUC-ROC']
    
    print(f"\n🏆 Best Model (by AUC-ROC): {best_model} ({best_auc:.4f})")


def main():
    """Main training pipeline."""
    print("\n" + "=" * 60)
    print("🤖 MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    # Add interaction features
    print("\n" + "=" * 60)
    print("CREATING ADVANCED INTERACTION FEATURES")
    print("=" * 60)
    print("\nAdding 30+ interaction features to capture complex patterns:")
    print("\n📊 Feature Categories:")
    print("  1. Contract × Tenure interactions (3 features)")
    print("  2. Service quality indicators (2 features)")
    print("  3. Payment method risk flags (3 features)")
    print("  4. Price sensitivity features (3 features)")
    print("  5. Customer isolation patterns (3 features)")
    print("  6. Internet service quality (2 features)")
    print("  7. Tenure-based risk buckets (3 features)")
    print("  8. Triple-interaction risk factors (2 features)")
    print("  9. Polynomial transformations (4 features)")
    print(" 10. Tenure × Charges interactions (2 features)")
    
    X_train_enhanced = create_interaction_features(X_train)
    X_val_enhanced = create_interaction_features(X_val)
    X_test_enhanced = create_interaction_features(X_test)
    
    print(f"\n✅ Features increased: {X_train.shape[1]} → {X_train_enhanced.shape[1]}")
    print(f"   Total engineered features: {X_train_enhanced.shape[1] - X_train.shape[1]}")
    
    # Scale features (for Logistic Regression)
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train_enhanced, X_val_enhanced, X_test_enhanced
    )
    
    # Train models
    all_metrics = []
    
    # 1. Logistic Regression (uses scaled data with interaction features)
    lr_model, lr_metrics = train_logistic_regression(X_train_scaled, y_train, X_val_scaled, y_val)
    all_metrics.append(lr_metrics)
    
    # 2. Random Forest (uses enhanced original data)
    rf_model, rf_metrics = train_random_forest(X_train_enhanced, y_train, X_val_enhanced, y_val)
    all_metrics.append(rf_metrics)
    
    # 3. XGBoost (uses enhanced original data)
    xgb_model, xgb_metrics = train_xgboost(X_train_enhanced, y_train, X_val_enhanced, y_val)
    all_metrics.append(xgb_metrics)
    
    # Compare models
    compare_models(all_metrics)
    
    print("\n" + "=" * 60)
    print("✅ MODEL TRAINING COMPLETE!")
    print("=" * 60)
    


if __name__ == "__main__":
    main()
