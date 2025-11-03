"""
Data Preparation Script - Getting Our Data Ready!

This is like organizing a messy closet before a big event. We:
1. Open the boxes (load raw data)
2. Clean dirty clothes (fix errors and missing info)
3. Add accessories (create new useful features)
4. Label everything (convert text to numbers AI can understand)
5. Sort into piles (split into practice, homework, and test sets)
6. Store neatly (save everything for later use)

Think of it like meal prep on Sunday - do all the hard work once, eat easy all week!
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# Define paths
RAW_DATA_PATH = 'data/raw/Telco-Customer-Churn.csv'
PROCESSED_DATA_DIR = 'data/processed/'

def load_and_explore_data():
    """
    Opens the customer data file and takes a first look at what's inside.
    
    Like opening a recipe book and checking: How many recipes are there? What ingredients
    do we need? Are any pages ripped or missing? What's the most popular dish?
    
    What you get back: A table of all the customer information
    """
    print("=" * 60)
    print("LOADING AND EXPLORING RAW DATA")
    print("=" * 60)
    
    # Load the CSV file
    df = pd.read_csv(RAW_DATA_PATH)
    
    print(f"\n📊 Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    print("\n📋 Column Names and Data Types:")
    print(df.dtypes)
    
    print("\n🔍 First 5 Rows:")
    print(df.head())
    
    print("\n📈 Basic Statistics:")
    print(df.describe())
    
    print("\n❓ Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("No missing values detected (as NaN)")
    else:
        print(missing[missing > 0])
    
    print("\n🎯 Target Variable Distribution (Churn):")
    print(df['Churn'].value_counts())
    print(f"Churn Rate: {(df['Churn'] == 'Yes').sum() / len(df) * 100:.2f}%")
    
    return df


def clean_data(df):
    """
    Fixes errors and fills in missing information in our customer data.
    
    Like proofreading an essay - fixing typos, filling in blank answers, and making
    sure all the numbers make sense. If someone's bill says " " (blank), we figure
    out what it should be or fill it with a smart guess.
    
    What you give it: Messy customer data table
    What you get back: Clean customer data table ready to use
    """
    print("\n" + "=" * 60)
    print("CLEANING DATA")
    print("=" * 60)
    
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Investigate TotalCharges issue
    print("\n🔍 Investigating TotalCharges column...")
    print(f"Data type: {df_clean['TotalCharges'].dtype}")
    
    # Check for non-numeric values
    # Try converting to numeric - errors will become NaN
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    
    # Now check for missing values
    total_charges_missing = df_clean['TotalCharges'].isnull().sum()
    print(f"Missing values in TotalCharges: {total_charges_missing}")
    
    if total_charges_missing > 0:
        # Let's look at these problematic rows
        print("\n📋 Examining rows with missing TotalCharges:")
        problematic_rows = df_clean[df_clean['TotalCharges'].isnull()]
        print(problematic_rows[['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges']].head())
        
        # Common approach: If tenure is 0 or very low and TotalCharges is missing,
        # it's likely a new customer, so TotalCharges should be close to MonthlyCharges
        print("\n✅ Imputation Strategy:")
        print("For customers with missing TotalCharges:")
        print("  - If tenure = 0: TotalCharges = 0")
        print("  - Otherwise: TotalCharges = MonthlyCharges × tenure")
        
        # Apply the strategy
        mask = df_clean['TotalCharges'].isnull()
        df_clean.loc[mask & (df_clean['tenure'] == 0), 'TotalCharges'] = 0
        df_clean.loc[mask & (df_clean['tenure'] > 0), 'TotalCharges'] = \
            df_clean.loc[mask & (df_clean['tenure'] > 0), 'MonthlyCharges'] * \
            df_clean.loc[mask & (df_clean['tenure'] > 0), 'tenure']
        
        print(f"\n✅ Fixed! Missing TotalCharges now: {df_clean['TotalCharges'].isnull().sum()}")
    
    # Verify data types
    print("\n📊 Data types after cleaning:")
    print(df_clean.dtypes)
    
    return df_clean


def engineer_features(df):
    """
    Creates brand new information from the data we already have.
    
    Like a chef creating a special sauce from basic ingredients. We combine existing
    info in clever ways to help our AI make better guesses. For example: Instead of
    just knowing someone has been a customer for 8 months, we group them as "6-12 months"
    customer - this helps spot patterns easier.
    
    What you give it: Clean customer data
    What you get back: Same data PLUS 4 new clever columns that help predict churn
    """
    print("\n" + "=" * 60)
    print("ENGINEERING FEATURES")
    print("=" * 60)
    
    df_engineered = df.copy()
    
    # 1. Tenure Buckets
    print("\n1️⃣ Creating tenure_bucket...")
    df_engineered['tenure_bucket'] = pd.cut(
        df_engineered['tenure'],
        bins=[-1, 6, 12, 24, 100],
        labels=['0-6m', '6-12m', '12-24m', '24m+']
    )
    print(f"   Distribution:\n{df_engineered['tenure_bucket'].value_counts().sort_index()}")
    
    # 2. Services Count
    print("\n2️⃣ Creating services_count...")
    # List of service columns (Yes/No values)
    service_columns = [
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    
    # Count 'Yes' values across service columns
    # We'll count 'Yes' as 1, everything else as 0
    df_engineered['services_count'] = 0
    for col in service_columns:
        if col in df_engineered.columns:
            df_engineered['services_count'] += (df_engineered[col] == 'Yes').astype(int)
    
    print(f"   Range: {df_engineered['services_count'].min()} to {df_engineered['services_count'].max()}")
    print(f"   Average: {df_engineered['services_count'].mean():.2f} services per customer")
    
    # 3. Monthly to Total Ratio
    print("\n3️⃣ Creating monthly_to_total_ratio...")
    # This ratio helps identify if charges have changed over time
    # Expected value ≈ 1/tenure (if charges were constant)
    # If ratio is high, monthly charges might have increased recently
    df_engineered['monthly_to_total_ratio'] = df_engineered['MonthlyCharges'] / \
        df_engineered[['TotalCharges', 'MonthlyCharges']].max(axis=1).replace(0, 1)
    
    print(f"   Range: {df_engineered['monthly_to_total_ratio'].min():.3f} to {df_engineered['monthly_to_total_ratio'].max():.3f}")
    
    # 4. Internet but no support flag
    print("\n4️⃣ Creating internet_no_support flag...")
    df_engineered['internet_no_support'] = (
        (df_engineered['InternetService'] != 'No') &
        (df_engineered['TechSupport'] == 'No')
    ).astype(int)
    
    print(f"   Customers with internet but no tech support: {df_engineered['internet_no_support'].sum()}")
    
    print("\n✅ Feature engineering complete!")
    print(f"   Total features now: {df_engineered.shape[1]}")
    
    return df_engineered


def create_interaction_features(df):
    """
    Combines two pieces of information to create super-smart new clues.
    
    Like a detective saying: "People who are seniors AND have fancy internet are more
    likely to have problems." Instead of looking at age and internet separately, we
    look at them TOGETHER. This helps our AI spot hidden patterns that are easy to miss.
    
    What you give it: Customer data with basic info
    What you get back: Same data PLUS 10 new super-smart combination columns
    """
    print("\n" + "=" * 60)
    print("CREATING INTERACTION FEATURES (OPTIONAL ENHANCEMENT)")
    print("=" * 60)
    
    df_interact = df.copy()
    
    # Track original column count
    original_cols = df_interact.shape[1]
    
    # 1. Senior Citizen × Fiber Optic (high-risk combo)
    print("\n1️⃣ Creating senior_fiber_optic interaction...")
    df_interact['senior_fiber_optic'] = (
        (df_interact['SeniorCitizen'] == 1) & 
        (df_interact['InternetService'] == 'Fiber optic')
    ).astype(int)
    print(f"   Customers: {df_interact['senior_fiber_optic'].sum()}")
    
    # 2. Month-to-month contract × Short tenure (churn risk)
    print("\n2️⃣ Creating month_to_month_short_tenure interaction...")
    df_interact['month_to_month_short_tenure'] = (
        (df_interact['Contract'] == 'Month-to-month') & 
        (df_interact['tenure'] <= 12)
    ).astype(int)
    print(f"   Customers: {df_interact['month_to_month_short_tenure'].sum()}")
    
    # 3. High charges with low services (poor value perception)
    print("\n3️⃣ Creating high_charges_low_services interaction...")
    median_charges = df_interact['MonthlyCharges'].median()
    median_services = df_interact['services_count'].median()
    df_interact['high_charges_low_services'] = (
        (df_interact['MonthlyCharges'] > median_charges) & 
        (df_interact['services_count'] <= median_services)
    ).astype(int)
    print(f"   Customers: {df_interact['high_charges_low_services'].sum()}")
    
    # 4. Fiber optic without online security (security concern)
    print("\n4️⃣ Creating fiber_no_security interaction...")
    df_interact['fiber_no_security'] = (
        (df_interact['InternetService'] == 'Fiber optic') & 
        (df_interact['OnlineSecurity'] == 'No')
    ).astype(int)
    print(f"   Customers: {df_interact['fiber_no_security'].sum()}")
    
    # 5. Electronic check payment with month-to-month (high churn combo)
    print("\n5️⃣ Creating electronic_check_monthly interaction...")
    df_interact['electronic_check_monthly'] = (
        (df_interact['PaymentMethod'] == 'Electronic check') & 
        (df_interact['Contract'] == 'Month-to-month')
    ).astype(int)
    print(f"   Customers: {df_interact['electronic_check_monthly'].sum()}")
    
    # 6. No phone service but has multiple lines (data inconsistency flag)
    print("\n6️⃣ Creating no_phone_multiple_lines interaction...")
    df_interact['no_phone_multiple_lines'] = (
        (df_interact['PhoneService'] == 'No') & 
        (df_interact['MultipleLines'] == 'Yes')
    ).astype(int)
    print(f"   Customers: {df_interact['no_phone_multiple_lines'].sum()}")
    
    # 7. Partner + Dependents (family stability - lower churn)
    print("\n7️⃣ Creating family_stability interaction...")
    df_interact['family_stability'] = (
        (df_interact['Partner'] == 'Yes') & 
        (df_interact['Dependents'] == 'Yes')
    ).astype(int)
    print(f"   Customers: {df_interact['family_stability'].sum()}")
    
    # 8. No tech support + No online security (lack of support services)
    print("\n8️⃣ Creating no_support_services interaction...")
    df_interact['no_support_services'] = (
        (df_interact['TechSupport'] == 'No') & 
        (df_interact['OnlineSecurity'] == 'No')
    ).astype(int)
    print(f"   Customers: {df_interact['no_support_services'].sum()}")
    
    # 9. Numeric interactions: tenure × monthly charges (total value indicator)
    print("\n9️⃣ Creating tenure_charges_product (numeric)...")
    df_interact['tenure_charges_product'] = df_interact['tenure'] * df_interact['MonthlyCharges']
    print(f"   Range: {df_interact['tenure_charges_product'].min():.2f} to {df_interact['tenure_charges_product'].max():.2f}")
    
    # 10. Contract type × services count (commitment level indicator)
    print("\n🔟 Creating contract_services_ratio (numeric)...")
    # Encode contract temporarily for multiplication (Month-to-month=0, One year=1, Two year=2)
    contract_numeric = df_interact['Contract'].map({
        'Month-to-month': 0,
        'One year': 1,
        'Two year': 2
    }).fillna(0)
    df_interact['contract_services_ratio'] = contract_numeric * df_interact['services_count']
    print(f"   Range: {df_interact['contract_services_ratio'].min():.2f} to {df_interact['contract_services_ratio'].max():.2f}")
    
    new_features = df_interact.shape[1] - original_cols
    print(f"\n✅ Interaction features complete!")
    print(f"   New features added: {new_features}")
    print(f"   Total features now: {df_interact.shape[1]}")
    
    return df_interact


def encode_categorical_variables(df):
    """
    Converts words into numbers so our AI can understand them.
    
    Like translating a book from English to Math. "Yes" becomes 1, "No" becomes 0,
    "Male" becomes 1, "Female" becomes 0. AI can only do math, not read words, so
    we need to give everything a number! We also save a dictionary so we can translate back later.
    
    What you give it: Data with words like "Yes", "No", "Male", "Female"
    What you get back: Same data but all words replaced with numbers, PLUS a translation dictionary
    """
    print("\n" + "=" * 60)
    print("ENCODING CATEGORICAL VARIABLES")
    print("=" * 60)
    
    df_encoded = df.copy()
    encoding_mapping = {}
    
    # Drop customerID - not useful for modeling
    df_encoded = df_encoded.drop('customerID', axis=1)
    
    # Identify categorical columns (object type or category)
    categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"\n📋 Categorical columns to encode: {len(categorical_cols)}")
    print(categorical_cols)
    
    # Encode each categorical column
    print("\n🔄 Encoding process:")
    for col in categorical_cols:
        if col == 'Churn':
            # Special handling for target variable
            # No -> 0, Yes -> 1 (this is what we want to predict)
            df_encoded[col] = (df_encoded[col] == 'Yes').astype(int)
            encoding_mapping[col] = {'No': 0, 'Yes': 1}
            print(f"   ✓ {col}: No=0, Yes=1 (target variable)")
        else:
            # Use LabelEncoder for other columns
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            
            # Store the mapping for documentation
            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            encoding_mapping[col] = mapping
            
            # Print mapping for reference
            mapping_str = ", ".join([f"{k}={v}" for k, v in sorted(mapping.items(), key=lambda x: x[1])])
            print(f"   ✓ {col}: {mapping_str}")
    
    print(f"\n✅ Encoding complete! All variables are now numeric.")
    print(f"   Shape: {df_encoded.shape}")
    
    # Save encoding mapping to a file for reference
    import json
    encoding_file = 'data/processed/encoding_mapping.json'
    os.makedirs(os.path.dirname(encoding_file), exist_ok=True)
    
    # Convert numpy types to native Python types for JSON serialization
    encoding_mapping_serializable = {}
    for key, value in encoding_mapping.items():
        encoding_mapping_serializable[key] = {k: int(v) if isinstance(v, (np.integer, np.int32, np.int64)) else v 
                                               for k, v in value.items()}
    
    with open(encoding_file, 'w') as f:
        json.dump(encoding_mapping_serializable, f, indent=2)
    print(f"   📝 Encoding mapping saved to: {encoding_file}")
    
    return df_encoded, encoding_mapping


def calculate_clv(df):
    """
    Figures out how much money each customer is worth to the company over time.
    
    Like calculating: "If Sarah pays $50/month and stays for 24 months, she's worth $1,200!"
    For customers who left, we use how long they actually stayed. For happy customers still
    here, we guess they'll stay 3 more years (36 months) because that's normal.
    
    What you give it: Customer data with monthly bills and how long they've been customers
    What you get back: Same data PLUS a new "CLV" column showing each customer's total worth
    """
    print("\n" + "=" * 60)
    print("CALCULATING CUSTOMER LIFETIME VALUE (CLV)")
    print("=" * 60)
    
    df_with_clv = df.copy()
    
    # Define our assumption
    ADDITIONAL_TENURE_FOR_ACTIVE = 36  # months
    
    print(f"\n📐 CLV Assumption:")
    print(f"   - Churned customers (Churn=1): ExpectedTenure = actual tenure")
    print(f"   - Active customers (Churn=0): ExpectedTenure = tenure + {ADDITIONAL_TENURE_FOR_ACTIVE} months")
    print(f"   - Formula: CLV = MonthlyCharges × ExpectedTenure")
    
    # Calculate Expected Tenure
    df_with_clv['ExpectedTenure'] = df_with_clv['tenure'].copy()
    active_customers_mask = df_with_clv['Churn'] == 0
    df_with_clv.loc[active_customers_mask, 'ExpectedTenure'] += ADDITIONAL_TENURE_FOR_ACTIVE
    
    # Calculate CLV
    df_with_clv['CLV'] = df_with_clv['MonthlyCharges'] * df_with_clv['ExpectedTenure']
    
    print(f"\n📊 CLV Statistics:")
    print(f"   Range: ${df_with_clv['CLV'].min():.2f} to ${df_with_clv['CLV'].max():.2f}")
    print(f"   Mean: ${df_with_clv['CLV'].mean():.2f}")
    print(f"   Median: ${df_with_clv['CLV'].median():.2f}")
    
    # Create CLV quartiles for segmentation
    df_with_clv['CLV_Quartile'] = pd.qcut(
        df_with_clv['CLV'], 
        q=4, 
        labels=['Low', 'Medium', 'High', 'Premium']
    )
    
    print(f"\n🎯 CLV Quartiles:")
    quartile_counts = df_with_clv['CLV_Quartile'].value_counts().sort_index()
    for quartile, count in quartile_counts.items():
        quartile_data = df_with_clv[df_with_clv['CLV_Quartile'] == quartile]
        churn_rate = (quartile_data['Churn'].sum() / len(quartile_data)) * 100
        avg_clv = quartile_data['CLV'].mean()
        print(f"   {quartile}: {count} customers, Avg CLV=${avg_clv:.2f}, Churn Rate={churn_rate:.1f}%")
    
    print(f"\n✅ CLV calculation complete!")
    
    return df_with_clv


def split_and_save_data(df):
    """
    Divides our customer data into 3 piles and saves them as files.
    
    Like dividing flashcards for studying:
    - 60% Practice pile: For teaching our AI the patterns
    - 20% Quiz pile: For checking if AI learned correctly while studying
    - 20% Final exam pile: For testing AI on totally new data it's never seen
    
    We make sure each pile has a fair mix of "left" and "stayed" customers!
    
    What you give it: Complete customer data
    What you get back: 3 separate piles (train, validation, test) saved as CSV files
    """
    print("\n" + "=" * 60)
    print("SPLITTING AND SAVING DATA")
    print("=" * 60)
    
    # Separate features and target
    # Keep CLV info separate for analysis but not as features for modeling
    X = df.drop(['Churn', 'CLV', 'ExpectedTenure', 'CLV_Quartile'], axis=1)
    y = df['Churn']
    clv_info = df[['CLV', 'ExpectedTenure', 'CLV_Quartile']]
    
    print(f"\n📊 Data shape before split:")
    print(f"   Features (X): {X.shape}")
    print(f"   Target (y): {y.shape}")
    
    # First split: 60% train, 40% temp (which will become 20% val + 20% test)
    X_train, X_temp, y_train, y_temp, clv_train, clv_temp = train_test_split(
        X, y, clv_info,
        test_size=0.40,
        random_state=42,
        stratify=y
    )
    
    # Second split: Split the 40% into 50%-50% (which gives us 20%-20% of original)
    X_val, X_test, y_val, y_test, clv_val, clv_test = train_test_split(
        X_temp, y_temp, clv_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )
    
    print(f"\n✅ Split complete:")
    print(f"   Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
    print(f"   Val:   {X_val.shape[0]} samples ({X_val.shape[0]/len(df)*100:.1f}%)")
    print(f"   Test:  {X_test.shape[0]} samples ({X_test.shape[0]/len(df)*100:.1f}%)")
    
    # Check stratification worked
    print(f"\n🎯 Churn rate verification (should be ~26.5% in all sets):")
    print(f"   Train: {y_train.mean()*100:.2f}%")
    print(f"   Val:   {y_val.mean()*100:.2f}%")
    print(f"   Test:  {y_test.mean()*100:.2f}%")
    
    # Combine back for saving (features + target + CLV info)
    train_df = pd.concat([X_train, y_train, clv_train], axis=1)
    val_df = pd.concat([X_val, y_val, clv_val], axis=1)
    test_df = pd.concat([X_test, y_test, clv_test], axis=1)
    
    # Save to CSV
    train_path = os.path.join(PROCESSED_DATA_DIR, 'train.csv')
    val_path = os.path.join(PROCESSED_DATA_DIR, 'val.csv')
    test_path = os.path.join(PROCESSED_DATA_DIR, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n💾 Data saved:")
    print(f"   {train_path}")
    print(f"   {val_path}")
    print(f"   {test_path}")
    
    print(f"\n✅ Data preparation complete!")
    print(f"   Total features: {X_train.shape[1]}")
    print(f"   Feature names: {list(X_train.columns)}")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    # Run the exploration
    df = load_and_explore_data()
    
    # Clean the data
    df_clean = clean_data(df)
    
    # Engineer features
    df_features = engineer_features(df_clean)
    
    # Create interaction features (OPTIONAL ENHANCEMENT from Project.md Section 7)
    df_interactions = create_interaction_features(df_features)
    
    # Encode categorical variables
    df_encoded, encoding_map = encode_categorical_variables(df_interactions)
    
    # Calculate CLV
    df_with_clv = calculate_clv(df_encoded)
    
    # Split and save
    train_df, val_df, test_df = split_and_save_data(df_with_clv)
    
    print("\n" + "=" * 60)
    print("🎉 DATA PREPARATION PIPELINE COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Perform CLV analysis (src/clv_analysis.py)")
    print("  2. Train models (src/train_models.py)")
    print("  3. Build Streamlit app (app.py)")
