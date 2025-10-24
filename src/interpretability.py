"""
Model Interpretability Script
Implements SHAP explanations for tree models and coefficient analysis for Logistic Regression.

For the Streamlit app, we need:
1. Global feature importance (which features matter most overall)
2. Local explanations (why a specific customer is predicted to churn)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import os

# Paths
MODELS_DIR = 'models/'
PROCESSED_DATA_DIR = 'data/processed/'
FIGURES_DIR = 'figures/'

os.makedirs(FIGURES_DIR, exist_ok=True)


def load_models_and_data():
    """Load trained models and test data."""
    print("=" * 60)
    print("LOADING MODELS AND DATA")
    print("=" * 60)
    
    # Load models
    lr_model = joblib.load(os.path.join(MODELS_DIR, 'logistic_regression.pkl'))
    rf_model = joblib.load(os.path.join(MODELS_DIR, 'random_forest.pkl'))
    xgb_model = joblib.load(os.path.join(MODELS_DIR, 'xgboost.pkl'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    
    # Load test data
    test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'test.csv'))
    
    # Prepare features (same exclusions as training)
    exclude_cols = ['Churn', 'CLV', 'ExpectedTenure', 'CLV_Quartile', 'TotalCharges', 'monthly_to_total_ratio']
    X_test = test.drop(columns=[col for col in exclude_cols if col in test.columns])
    y_test = test['Churn']
    
    # Add interaction features (must match training)
    from train_models import create_interaction_features
    X_test_enhanced = create_interaction_features(X_test)
    
    # Scale for Logistic Regression
    X_test_scaled = scaler.transform(X_test_enhanced)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test_enhanced.columns, index=X_test_enhanced.index)
    
    print(f"\n✅ Loaded:")
    print(f"   Models: Logistic Regression, Random Forest, XGBoost")
    print(f"   Test data: {X_test_enhanced.shape}")
    
    return lr_model, rf_model, xgb_model, X_test_enhanced, X_test_scaled, y_test


def explain_logistic_regression(model, X, feature_names):
    """
    Get feature importance from Logistic Regression coefficients.
    
    For linear models, coefficients tell us feature importance directly.
    We use absolute standardized coefficients.
    """
    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION: Coefficient Analysis")
    print("=" * 60)
    
    # Get coefficients
    coefficients = model.coef_[0]
    
    # Calculate standardized importance: |coef| × std(feature)
    # This makes coefficients comparable across different scales
    feature_std = X.std(axis=0)
    importance = np.abs(coefficients * feature_std)
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients),
        'std': feature_std,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("\n📊 Top 15 Most Important Features:")
    print(importance_df.head(15)[['feature', 'coefficient', 'importance']].to_string(index=False))
    
    # Save importance
    importance_path = os.path.join(MODELS_DIR, 'lr_feature_importance.csv')
    importance_df.to_csv(importance_path, index=False)
    print(f"\n💾 Saved: {importance_path}")
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(20)
    
    colors = ['red' if c < 0 else 'green' for c in top_features['coefficient']]
    plt.barh(range(len(top_features)), top_features['importance'], color=colors, alpha=0.7)
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Absolute Standardized Coefficient (Importance)')
    plt.title('Logistic Regression: Top 20 Feature Importances\n(Red = Increases Churn, Green = Decreases Churn)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    plot_path = os.path.join(FIGURES_DIR, 'lr_feature_importance.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved plot: {plot_path}")
    plt.close()
    
    return importance_df


def explain_tree_models_shap(model, X, model_name, sample_size=500):
    """
    Use SHAP TreeExplainer for Random Forest and XGBoost.
    
    SHAP values explain how much each feature contributed to a prediction.
    Positive SHAP = pushes prediction toward churn.
    Negative SHAP = pushes prediction away from churn.
    """
    print("\n" + "=" * 60)
    print(f"{model_name.upper()}: SHAP Analysis")
    print("=" * 60)
    
    # Sample data for faster computation (SHAP can be slow)
    if len(X) > sample_size:
        print(f"\n⏱️  Sampling {sample_size} observations for faster SHAP computation...")
        X_sample = X.sample(n=sample_size, random_state=42)
    else:
        X_sample = X
    
    # Create SHAP explainer
    print("🔧 Creating SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    print("🔧 Computing SHAP values...")
    shap_values = explainer.shap_values(X_sample)
    
    # For binary classification, shap_values might be a list [class0, class1]
    # We want the SHAP values for the positive class (churn = 1)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Churn class
    
    print("✅ SHAP values computed!")
    
    # Calculate mean absolute SHAP values for global importance
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)
    
    print("\n📊 Top 15 Most Important Features (by mean |SHAP|):")
    print(feature_importance.head(15).to_string(index=False))
    
    # Save importance
    importance_path = os.path.join(MODELS_DIR, f'{model_name.lower().replace(" ", "_")}_shap_importance.csv')
    feature_importance.to_csv(importance_path, index=False)
    print(f"\n💾 Saved: {importance_path}")
    
    # Create SHAP summary plot (beeswarm)
    print("\n📈 Creating SHAP summary plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=20)
    plt.title(f'{model_name}: Top 20 Features by SHAP Importance')
    plt.tight_layout()
    
    plot_path = os.path.join(FIGURES_DIR, f'{model_name.lower().replace(" ", "_")}_shap_summary.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved plot: {plot_path}")
    plt.close()
    
    # Create detailed SHAP beeswarm plot
    print("📈 Creating SHAP beeswarm plot...")
    plt.figure(figsize=(10, 10))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
    plt.title(f'{model_name}: SHAP Value Distribution (Top 20 Features)')
    plt.tight_layout()
    
    beeswarm_path = os.path.join(FIGURES_DIR, f'{model_name.lower().replace(" ", "_")}_shap_beeswarm.png')
    plt.savefig(beeswarm_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved plot: {beeswarm_path}")
    plt.close()
    
    # Save explainer for use in Streamlit app
    explainer_path = os.path.join(MODELS_DIR, f'{model_name.lower().replace(" ", "_")}_explainer.pkl')
    joblib.dump(explainer, explainer_path)
    print(f"💾 Saved explainer: {explainer_path}")
    
    return feature_importance, explainer, shap_values


def compare_feature_importance(lr_importance, rf_importance, xgb_importance):
    """
    Compare feature importance across all three models.
    """
    print("\n" + "=" * 60)
    print("COMPARING FEATURE IMPORTANCE ACROSS MODELS")
    print("=" * 60)
    
    # Normalize importances to 0-1 scale for comparison
    lr_norm = lr_importance.copy()
    lr_norm['importance_norm'] = lr_norm['importance'] / lr_norm['importance'].max()
    lr_norm = lr_norm[['feature', 'importance_norm']].rename(columns={'importance_norm': 'LR'})
    
    rf_norm = rf_importance.copy()
    rf_norm['importance_norm'] = rf_norm['mean_abs_shap'] / rf_norm['mean_abs_shap'].max()
    rf_norm = rf_norm[['feature', 'importance_norm']].rename(columns={'importance_norm': 'RF'})
    
    xgb_norm = xgb_importance.copy()
    xgb_norm['importance_norm'] = xgb_norm['mean_abs_shap'] / xgb_norm['mean_abs_shap'].max()
    xgb_norm = xgb_norm[['feature', 'importance_norm']].rename(columns={'importance_norm': 'XGB'})
    
    # Merge all
    comparison = lr_norm.merge(rf_norm, on='feature', how='outer').merge(xgb_norm, on='feature', how='outer')
    comparison = comparison.fillna(0)
    
    # Calculate average importance across models
    comparison['avg_importance'] = comparison[['LR', 'RF', 'XGB']].mean(axis=1)
    comparison = comparison.sort_values('avg_importance', ascending=False)
    
    print("\n📊 Top 20 Features Across All Models:")
    print(comparison.head(20).to_string(index=False))
    
    # Save comparison
    comparison_path = os.path.join(MODELS_DIR, 'feature_importance_comparison.csv')
    comparison.to_csv(comparison_path, index=False)
    print(f"\n💾 Saved: {comparison_path}")
    
    # Create comparison visualization
    plt.figure(figsize=(12, 8))
    top_features = comparison.head(15)
    
    x = np.arange(len(top_features))
    width = 0.25
    
    plt.barh(x - width, top_features['LR'], width, label='Logistic Regression', alpha=0.8)
    plt.barh(x, top_features['RF'], width, label='Random Forest', alpha=0.8)
    plt.barh(x + width, top_features['XGB'], width, label='XGBoost', alpha=0.8)
    
    plt.yticks(x, top_features['feature'])
    plt.xlabel('Normalized Importance')
    plt.title('Feature Importance Comparison Across Models (Top 15)')
    plt.legend()
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    plot_path = os.path.join(FIGURES_DIR, 'feature_importance_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved plot: {plot_path}")
    plt.close()
    
    return comparison


def create_local_explanation_example(explainer, model, X_sample, model_name):
    """
    Create an example local explanation for a single high-risk customer.
    This demonstrates what we'll show in the Streamlit app.
    """
    print("\n" + "=" * 60)
    print(f"{model_name.upper()}: Example Local Explanation")
    print("=" * 60)
    
    # Find a high-risk prediction
    predictions = model.predict_proba(X_sample)[:, 1]
    high_risk_idx = predictions.argmax()
    
    customer = X_sample.iloc[high_risk_idx:high_risk_idx+1]
    churn_prob = predictions[high_risk_idx]
    
    print(f"\n👤 Example Customer:")
    print(f"   Predicted Churn Probability: {churn_prob*100:.1f}%")
    
    # Calculate SHAP values for this customer
    if 'Tree' in str(type(explainer)):
        shap_values = explainer.shap_values(customer)
        if isinstance(shap_values, list):
            shap_values = shap_values[1][0]
        else:
            shap_values = shap_values[0]
    else:
        return
    
    # Get top contributors
    feature_contributions = pd.DataFrame({
        'feature': X_sample.columns,
        'value': customer.iloc[0].values,
        'shap_value': shap_values
    }).sort_values('shap_value', key=abs, ascending=False)
    
    print("\n📊 Top 10 Features Driving This Prediction:")
    print(feature_contributions.head(10)[['feature', 'value', 'shap_value']].to_string(index=False))
    
    # Create waterfall plot
    print("\n📈 Creating SHAP waterfall plot...")
    plt.figure(figsize=(10, 6))
    
    # Handle expected value - it might be an array for multi-class or single value
    expected_val = explainer.expected_value if hasattr(explainer, 'expected_value') else 0
    if isinstance(expected_val, np.ndarray):
        expected_val = float(expected_val[0]) if len(expected_val) > 0 else 0.0
    
    shap.waterfall_plot(shap.Explanation(
        values=shap_values,
        base_values=expected_val,
        data=customer.iloc[0].values,
        feature_names=X_sample.columns.tolist()
    ), max_display=15, show=False)
    plt.title(f'{model_name}: Local Explanation for High-Risk Customer')
    plt.tight_layout()
    
    plot_path = os.path.join(FIGURES_DIR, f'{model_name.lower().replace(" ", "_")}_local_example.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved plot: {plot_path}")
    plt.close()


def main():
    """Main execution."""
    print("=" * 60)
    print("🔍 MODEL INTERPRETABILITY PIPELINE")
    print("=" * 60)
    
    # Load everything
    lr_model, rf_model, xgb_model, X_test, X_test_scaled, y_test = load_models_and_data()
    
    # 1. Explain Logistic Regression (coefficient analysis)
    lr_importance = explain_logistic_regression(lr_model, X_test_scaled, X_test.columns)
    
    # 2. Explain Random Forest (SHAP)
    rf_importance, rf_explainer, rf_shap_values = explain_tree_models_shap(
        rf_model, X_test, "Random Forest"
    )
    
    # 3. Explain XGBoost (SHAP)
    xgb_importance, xgb_explainer, xgb_shap_values = explain_tree_models_shap(
        xgb_model, X_test, "XGBoost"
    )
    
    # 4. Compare across models
    comparison = compare_feature_importance(lr_importance, rf_importance, xgb_importance)
    
    # 5. Create example local explanations
    create_local_explanation_example(rf_explainer, rf_model, X_test.sample(100, random_state=42), "Random Forest")
    create_local_explanation_example(xgb_explainer, xgb_model, X_test.sample(100, random_state=42), "XGBoost")
    
    print("\n" + "=" * 60)
    print("✅ INTERPRETABILITY ANALYSIS COMPLETE!")
    print("=" * 60)
    print("\n📁 Generated Files:")
    print("   • Feature importance CSVs (models/)")
    print("   • SHAP summary plots (figures/)")
    print("   • SHAP explainers for Streamlit (models/)")
    print("\n🔑 Key Insights:")
    print("   Check the comparison plot to see which features are")
    print("   consistently important across all models.")
    print("\nNext step: Build Streamlit app (app.py)")


if __name__ == "__main__":
    main()
