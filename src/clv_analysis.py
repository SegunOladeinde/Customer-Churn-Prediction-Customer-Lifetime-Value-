"""
CLV Analysis Script
Analyzes Customer Lifetime Value and its relationship with churn.
Creates visualizations for business insights and the Streamlit app.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for professional-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# Define paths
PROCESSED_DATA_DIR = 'data/processed/'
FIGURES_DIR = 'figures/'

# Create figures directory if it doesn't exist
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_processed_data():
    """Load the processed train, val, and test datasets."""
    print("=" * 60)
    print("LOADING PROCESSED DATA")
    print("=" * 60)
    
    train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'train.csv'))
    val = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'val.csv'))
    test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'test.csv'))
    
    # Combine all for CLV analysis (we're analyzing the full dataset)
    full_data = pd.concat([train, val, test], axis=0, ignore_index=True)
    
    print(f"\n✅ Data loaded:")
    print(f"   Train: {len(train)} samples")
    print(f"   Val:   {len(val)} samples")
    print(f"   Test:  {len(test)} samples")
    print(f"   Total: {len(full_data)} samples")
    
    return full_data, train, val, test


def analyze_clv_distribution(df):
    """
    Analyze and visualize CLV distribution.
    
    Args:
        df (pd.DataFrame): Full dataset with CLV
    """
    print("\n" + "=" * 60)
    print("CLV DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    # Basic statistics
    print(f"\n📊 CLV Statistics:")
    print(f"   Mean:   ${df['CLV'].mean():,.2f}")
    print(f"   Median: ${df['CLV'].median():,.2f}")
    print(f"   Std:    ${df['CLV'].std():,.2f}")
    print(f"   Min:    ${df['CLV'].min():,.2f}")
    print(f"   Max:    ${df['CLV'].max():,.2f}")
    
    # Create CLV distribution histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(df['CLV'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Customer Lifetime Value ($)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Number of Customers', fontsize=12, fontweight='bold')
    axes[0].set_title('CLV Distribution', fontsize=14, fontweight='bold')
    axes[0].axvline(df['CLV'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${df["CLV"].mean():,.0f}')
    axes[0].axvline(df['CLV'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: ${df["CLV"].median():,.0f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot by Churn status
    churn_labels = {0: 'Active (No Churn)', 1: 'Churned'}
    df['Churn_Label'] = df['Churn'].map(churn_labels)
    
    axes[1].boxplot([df[df['Churn'] == 0]['CLV'], df[df['Churn'] == 1]['CLV']], 
                    labels=['Active', 'Churned'],
                    patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7))
    axes[1].set_ylabel('Customer Lifetime Value ($)', fontsize=12, fontweight='bold')
    axes[1].set_title('CLV by Churn Status', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'clv_distribution.png'), dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {os.path.join(FIGURES_DIR, 'clv_distribution.png')}")
    plt.close()
    
    # CLV comparison by churn
    print(f"\n💡 CLV by Churn Status:")
    for churn_val in [0, 1]:
        churn_label = churn_labels[churn_val]
        clv_subset = df[df['Churn'] == churn_val]['CLV']
        print(f"   {churn_label}:")
        print(f"     Mean CLV: ${clv_subset.mean():,.2f}")
        print(f"     Median CLV: ${clv_subset.median():,.2f}")


def analyze_clv_quartiles(df):
    """
    Analyze churn rate by CLV quartile.
    This is the KEY business insight!
    
    Args:
        df (pd.DataFrame): Full dataset with CLV
    """
    print("\n" + "=" * 60)
    print("CLV QUARTILE ANALYSIS")
    print("=" * 60)
    
    # Calculate quartile statistics
    quartile_stats = []
    for quartile in ['Low', 'Medium', 'High', 'Premium']:
        subset = df[df['CLV_Quartile'] == quartile]
        stats = {
            'Quartile': quartile,
            'Customers': len(subset),
            'Avg_CLV': subset['CLV'].mean(),
            'Churn_Rate': (subset['Churn'].sum() / len(subset)) * 100,
            'Churned_Count': subset['Churn'].sum(),
            'Active_Count': len(subset) - subset['Churn'].sum()
        }
        quartile_stats.append(stats)
    
    quartile_df = pd.DataFrame(quartile_stats)
    
    print("\n📊 CLV Quartile Summary:")
    print(quartile_df.to_string(index=False))
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Churn Rate by Quartile (BAR CHART)
    quartiles_ordered = ['Low', 'Medium', 'High', 'Premium']
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']  # Red to Blue gradient
    
    axes[0, 0].bar(quartiles_ordered, quartile_df['Churn_Rate'], color=colors, edgecolor='black', alpha=0.8)
    axes[0, 0].set_xlabel('CLV Quartile', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Churn Rate (%)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('🔥 Churn Rate by CLV Quartile', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (quartile, rate) in enumerate(zip(quartiles_ordered, quartile_df['Churn_Rate'])):
        axes[0, 0].text(i, rate + 2, f'{rate:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    # 2. Average CLV by Quartile
    axes[0, 1].bar(quartiles_ordered, quartile_df['Avg_CLV'], color=colors, edgecolor='black', alpha=0.8)
    axes[0, 1].set_xlabel('CLV Quartile', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Average CLV ($)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('💰 Average CLV by Quartile', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (quartile, clv) in enumerate(zip(quartiles_ordered, quartile_df['Avg_CLV'])):
        axes[0, 1].text(i, clv + 200, f'${clv:,.0f}', ha='center', fontsize=10, fontweight='bold')
    
    # 3. Customer Count by Quartile and Churn Status (STACKED BAR)
    x_pos = np.arange(len(quartiles_ordered))
    width = 0.6
    
    axes[1, 0].bar(x_pos, quartile_df['Active_Count'], width, label='Active', color='#2ca02c', edgecolor='black', alpha=0.8)
    axes[1, 0].bar(x_pos, quartile_df['Churned_Count'], width, bottom=quartile_df['Active_Count'], 
                   label='Churned', color='#d62728', edgecolor='black', alpha=0.8)
    
    axes[1, 0].set_xlabel('CLV Quartile', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Number of Customers', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('👥 Customer Distribution by Quartile', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(quartiles_ordered)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Key Insight Text Box
    axes[1, 1].axis('off')
    
    insight_text = f"""
    🎯 KEY BUSINESS INSIGHTS
    
    1. PREMIUM CUSTOMERS (Top 25%)
       • Average CLV: ${quartile_df.iloc[3]['Avg_CLV']:,.0f}
       • Churn Rate: {quartile_df.iloc[3]['Churn_Rate']:.1f}%
       • 💡 Very loyal! Minimal risk.
    
    2. HIGH-VALUE CUSTOMERS (50-75%)
       • Average CLV: ${quartile_df.iloc[2]['Avg_CLV']:,.0f}
       • Churn Rate: {quartile_df.iloc[2]['Churn_Rate']:.1f}%
       • 🎯 Best retention ROI target!
    
    3. LOW-VALUE CUSTOMERS (Bottom 25%)
       • Average CLV: ${quartile_df.iloc[0]['Avg_CLV']:,.0f}
       • Churn Rate: {quartile_df.iloc[0]['Churn_Rate']:.1f}%
       • ⚠️ High churn, low value.
    
    RECOMMENDATION:
    Focus retention efforts on HIGH and 
    PREMIUM segments for maximum impact.
    """
    
    axes[1, 1].text(0.1, 0.9, insight_text, transform=axes[1, 1].transAxes,
                   fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'clv_quartile_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {os.path.join(FIGURES_DIR, 'clv_quartile_analysis.png')}")
    plt.close()


def generate_insights(df):
    """
    Generate written business insights based on CLV analysis.
    
    Args:
        df (pd.DataFrame): Full dataset with CLV
    """
    print("\n" + "=" * 60)
    print("BUSINESS INSIGHTS")
    print("=" * 60)
    
    insights = []
    
    # Insight 1: CLV-Churn Relationship
    premium_churn = df[df['CLV_Quartile'] == 'Premium']['Churn'].mean() * 100
    low_churn = df[df['CLV_Quartile'] == 'Low']['Churn'].mean() * 100
    
    insight1 = f"""
    1. Strong Inverse Relationship: CLV ↑, Churn ↓
       Premium customers (top 25% CLV) have a churn rate of {premium_churn:.1f}%, 
       while low-value customers have a {low_churn:.1f}% churn rate. 
       This {low_churn/premium_churn:.1f}x difference shows that high-value customers 
       are significantly more loyal.
    """
    insights.append(insight1)
    
    # Insight 2: Retention Priority
    high_premium_customers = len(df[(df['CLV_Quartile'] == 'High') | (df['CLV_Quartile'] == 'Premium')])
    high_premium_revenue = df[(df['CLV_Quartile'] == 'High') | (df['CLV_Quartile'] == 'Premium')]['CLV'].sum()
    total_revenue = df['CLV'].sum()
    revenue_pct = (high_premium_revenue / total_revenue) * 100
    
    insight2 = f"""
    2. Focus on the Top 50%
       The High and Premium segments ({high_premium_customers:,} customers) represent 
       {revenue_pct:.1f}% of total customer value. These customers should be the 
       primary focus of retention campaigns, as saving even a few prevents 
       significant revenue loss.
    """
    insights.append(insight2)
    
    # Insight 3: Low-Value Segment Strategy
    low_customers = len(df[df['CLV_Quartile'] == 'Low'])
    low_revenue_pct = (df[df['CLV_Quartile'] == 'Low']['CLV'].sum() / total_revenue) * 100
    
    insight3 = f"""
    3. Reconsider Low-Value Retention Efforts
       Low-value customers ({low_customers:,}) contribute only {low_revenue_pct:.1f}% 
       of total value but have a {low_churn:.1f}% churn rate. Instead of expensive 
       retention campaigns, consider cost-effective self-service support or 
       letting natural churn occur while focusing resources on high-value segments.
    """
    insights.append(insight3)
    
    # Print all insights
    for insight in insights:
        print(insight)
    
    # Save to file
    insights_file = os.path.join(PROCESSED_DATA_DIR, 'clv_insights.txt')
    with open(insights_file, 'w', encoding='utf-8') as f:
        f.write("CUSTOMER LIFETIME VALUE ANALYSIS - KEY INSIGHTS\n")
        f.write("=" * 60 + "\n\n")
        for insight in insights:
            f.write(insight + "\n")
    
    print(f"\n✅ Insights saved to: {insights_file}")


def main():
    """Main execution function."""
    print("\n" + "=" * 60)
    print("🔍 CUSTOMER LIFETIME VALUE (CLV) ANALYSIS")
    print("=" * 60)
    
    # Load data
    full_data, train, val, test = load_processed_data()
    
    # Perform analyses
    analyze_clv_distribution(full_data)
    analyze_clv_quartiles(full_data)
    generate_insights(full_data)
    
    print("\n" + "=" * 60)
    print("✅ CLV ANALYSIS COMPLETE!")
    print("=" * 60)
    print("\nGenerated files:")
    print(f"  • {os.path.join(FIGURES_DIR, 'clv_distribution.png')}")
    print(f"  • {os.path.join(FIGURES_DIR, 'clv_quartile_analysis.png')}")
    print(f"  • {os.path.join(PROCESSED_DATA_DIR, 'clv_insights.txt')}")
    print("\nNext step: Train models (src/train_models.py)")


if __name__ == "__main__":
    main()
