#!/usr/bin/env python3
"""
Demographic Segmentation Analysis for DCD Timing Bottleneck

This script controls for demographic factors and segments the timing analysis
to ensure the 5.3x timing bottleneck finding isn't confounded by patient characteristics.

Tests:
1. Does the timing effect hold within each demographic group?
2. Are certain groups systematically getting shorter windows?
3. Is there a pure timing effect after controlling for demographics?
4. Do demographics interact with timing (e.g., timing matters more for certain groups)?

Author: ODE Research Team
Date: November 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path.home() / 'physionet.org/files/orchid/2.1.1'
OUTPUT_DIR = Path.home() / 'results'
FIGURES_DIR = OUTPUT_DIR / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_data():
    """Load ORCHID referrals data"""
    print("Loading ORCHID data...")
    df = pd.read_csv(DATA_DIR / 'OPOReferrals.csv', low_memory=False)
    print(f"Loaded {len(df):,} referrals\n")
    return df

def calculate_time_windows(df):
    """Calculate time windows for analysis"""
    print("Calculating time windows...")
    
    # Convert time columns to datetime
    time_cols = [col for col in df.columns if 'time_' in col.lower()]
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Calculate windows
    if 'time_referred' in df.columns and 'time_asystole' in df.columns:
        df['window_hours_dcd'] = (df['time_asystole'] - df['time_referred']).dt.total_seconds() / 3600
    
    if 'time_referred' in df.columns and 'time_brain_death' in df.columns:
        df['window_hours_dbd'] = (df['time_brain_death'] - df['time_referred']).dt.total_seconds() / 3600
    
    return df

def categorize_demographics(df):
    """Create demographic categories"""
    print("Categorizing demographics...")
    
    # Age groups
    df['age_group'] = pd.cut(df['age'], 
                              bins=[0, 18, 35, 50, 65, 120],
                              labels=['0-17', '18-34', '35-49', '50-64', '65+'])
    
    # BMI categories (if height and weight available)
    if 'height_in' in df.columns and 'weight_kg' in df.columns:
        df['bmi'] = df['weight_kg'] / (df['height_in'] * 0.0254) ** 2
        df['bmi_category'] = pd.cut(df['bmi'],
                                     bins=[0, 18.5, 25, 30, 100],
                                     labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    
    # Window length categories
    if 'window_hours_dcd' in df.columns:
        df['window_category'] = pd.cut(df['window_hours_dcd'],
                                        bins=[0, 12, 24, 48, 72, 1000],
                                        labels=['<12hr', '12-24hr', '24-48hr', '48-72hr', '>72hr'])
    
    return df

# ============================================================================
# ANALYSIS 1: TIMING EFFECT WITHIN DEMOGRAPHIC GROUPS
# ============================================================================

def analyze_timing_by_demographics(df):
    """Test if timing effect holds within each demographic group"""
    print("\n" + "="*80)
    print("ANALYSIS 1: TIMING EFFECT WITHIN DEMOGRAPHIC GROUPS")
    print("="*80)
    
    # Filter to DCD MSCs
    dcd_mscs = df[(df['brain_death'] == 0) & (df['is_msc'] == True)].copy()
    
    if len(dcd_mscs) == 0:
        print("No DCD MSCs found. Skipping analysis.")
        return None
    
    print(f"\nAnalyzing {len(dcd_mscs):,} DCD MSCs\n")
    
    results = []
    
    # Demographic variables to segment by
    demo_vars = []
    if 'race' in dcd_mscs.columns:
        demo_vars.append(('race', 'Race'))
    if 'gender' in dcd_mscs.columns:
        demo_vars.append(('gender', 'Gender'))
    if 'age_group' in dcd_mscs.columns:
        demo_vars.append(('age_group', 'Age Group'))
    if 'bmi_category' in dcd_mscs.columns:
        demo_vars.append(('bmi_category', 'BMI Category'))
    if 'cause_of_death_opo' in dcd_mscs.columns:
        demo_vars.append(('cause_of_death_opo', 'Cause of Death'))
    if 'opo' in dcd_mscs.columns:
        demo_vars.append(('opo', 'OPO'))
    
    for var, label in demo_vars:
        print(f"\n{label}:")
        print("-" * 60)
        
        # Get value counts
        value_counts = dcd_mscs[var].value_counts()
        
        # Only analyze groups with sufficient sample size
        for value in value_counts[value_counts >= 100].index:
            subset = dcd_mscs[dcd_mscs[var] == value]
            
            if 'window_category' not in subset.columns:
                continue
            
            # Calculate approach rates by window category
            window_approach = subset.groupby('window_category')['approached'].agg(['mean', 'count'])
            
            if len(window_approach) < 2:
                continue
            
            # Calculate effect size (ratio of highest to lowest)
            if window_approach['mean'].max() > 0 and window_approach['mean'].min() > 0:
                effect_size = window_approach['mean'].max() / window_approach['mean'].min()
            else:
                effect_size = np.nan
            
            # Test correlation between window length and approach rate
            subset_clean = subset[subset['window_hours_dcd'].notna() & subset['approached'].notna()]
            if len(subset_clean) > 10:
                corr, p_value = stats.spearmanr(subset_clean['window_hours_dcd'], 
                                                 subset_clean['approached'])
            else:
                corr, p_value = np.nan, np.nan
            
            results.append({
                'Demographic': label,
                'Value': value,
                'N': len(subset),
                'Effect Size': effect_size,
                'Correlation': corr,
                'P-value': p_value
            })
            
            print(f"  {value}: N={len(subset):,}, Effect={effect_size:.2f}x, ρ={corr:.3f}, p={p_value:.4f}")
    
    # Create summary dataframe
    results_df = pd.DataFrame(results)
    
    # Save results
    output_file = OUTPUT_DIR / 'timing_effect_by_demographics.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved results: {output_file}")
    
    return results_df

# ============================================================================
# ANALYSIS 2: DEMOGRAPHIC DISPARITIES IN WINDOW LENGTH
# ============================================================================

def analyze_window_disparities(df):
    """Test if certain demographic groups systematically get shorter windows"""
    print("\n" + "="*80)
    print("ANALYSIS 2: DEMOGRAPHIC DISPARITIES IN WINDOW LENGTH")
    print("="*80)
    
    dcd_mscs = df[(df['brain_death'] == 0) & (df['is_msc'] == True)].copy()
    
    if 'window_hours_dcd' not in dcd_mscs.columns:
        print("Window data not available. Skipping analysis.")
        return None
    
    print(f"\nAnalyzing window length disparities across demographics\n")
    
    disparities = []
    
    # Test each demographic variable
    demo_vars = []
    if 'race' in dcd_mscs.columns:
        demo_vars.append(('race', 'Race'))
    if 'gender' in dcd_mscs.columns:
        demo_vars.append(('gender', 'Gender'))
    if 'age_group' in dcd_mscs.columns:
        demo_vars.append(('age_group', 'Age Group'))
    
    for var, label in demo_vars:
        print(f"\n{label}:")
        print("-" * 60)
        
        # Calculate median window by group
        window_by_group = dcd_mscs.groupby(var)['window_hours_dcd'].agg(['median', 'mean', 'count'])
        window_by_group = window_by_group[window_by_group['count'] >= 50]  # Minimum sample size
        
        if len(window_by_group) < 2:
            continue
        
        print(window_by_group)
        
        # Test for significant differences (Kruskal-Wallis)
        groups = [dcd_mscs[dcd_mscs[var] == val]['window_hours_dcd'].dropna() 
                  for val in window_by_group.index]
        
        if len(groups) >= 2:
            h_stat, p_value = stats.kruskal(*groups)
            
            disparities.append({
                'Demographic': label,
                'H-statistic': h_stat,
                'P-value': p_value,
                'Significant': p_value < 0.05
            })
            
            print(f"\nKruskal-Wallis Test: H={h_stat:.2f}, p={p_value:.4f}")
            if p_value < 0.05:
                print("✓ SIGNIFICANT DISPARITY DETECTED")
            else:
                print("○ No significant disparity")
    
    # Create summary
    disparities_df = pd.DataFrame(disparities)
    
    # Save results
    output_file = OUTPUT_DIR / 'window_length_disparities.csv'
    disparities_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved results: {output_file}")
    
    return disparities_df

# ============================================================================
# ANALYSIS 3: REGRESSION WITH DEMOGRAPHIC CONTROLS
# ============================================================================

def regression_with_controls(df):
    """Logistic regression to isolate pure timing effect"""
    print("\n" + "="*80)
    print("ANALYSIS 3: REGRESSION WITH DEMOGRAPHIC CONTROLS")
    print("="*80)
    
    dcd_mscs = df[(df['brain_death'] == 0) & (df['is_msc'] == True)].copy()
    
    if 'window_hours_dcd' not in dcd_mscs.columns:
        print("Window data not available. Skipping analysis.")
        return None
    
    print("\nFitting logistic regression: P(approached) ~ window + demographics\n")
    
    # Prepare data
    model_data = dcd_mscs[['approached', 'window_hours_dcd', 'age', 'gender', 'race']].copy()
    model_data = model_data.dropna()
    
    if len(model_data) < 100:
        print("Insufficient data for regression. Skipping.")
        return None
    
    # Create dummy variables
    model_data = pd.get_dummies(model_data, columns=['gender', 'race'], drop_first=True)
    
    # Standardize continuous variables
    model_data['window_std'] = (model_data['window_hours_dcd'] - model_data['window_hours_dcd'].mean()) / model_data['window_hours_dcd'].std()
    model_data['age_std'] = (model_data['age'] - model_data['age'].mean()) / model_data['age'].std()
    
    # Fit logistic regression (using statsmodels if available, otherwise sklearn)
    try:
        import statsmodels.api as sm
        
        # Prepare X and y
        X = model_data.drop(['approached', 'window_hours_dcd', 'age'], axis=1)
        X = sm.add_constant(X)
        y = model_data['approached']
        
        # Fit model
        model = sm.Logit(y, X).fit(disp=0)
        
        print(model.summary())
        
        # Extract key results
        window_coef = model.params['window_std']
        window_pval = model.pvalues['window_std']
        window_or = np.exp(window_coef)
        
        print(f"\n{'='*60}")
        print(f"TIMING EFFECT (controlling for demographics):")
        print(f"  Coefficient: {window_coef:.4f}")
        print(f"  Odds Ratio: {window_or:.4f}")
        print(f"  P-value: {window_pval:.6f}")
        print(f"  Interpretation: 1 SD increase in window → {(window_or-1)*100:.1f}% increase in odds of approach")
        print(f"{'='*60}")
        
        return model
        
    except ImportError:
        print("statsmodels not available. Using sklearn for basic analysis.")
        from sklearn.linear_model import LogisticRegression
        
        X = model_data.drop(['approached', 'window_hours_dcd', 'age'], axis=1)
        y = model_data['approached']
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        
        # Get window coefficient (first feature)
        window_coef = model.coef_[0][0]
        
        print(f"\nTiming coefficient: {window_coef:.4f}")
        print("(Use statsmodels for full regression output)")
        
        return model

# ============================================================================
# ANALYSIS 4: INTERACTION EFFECTS
# ============================================================================

def analyze_interactions(df):
    """Test if timing effect varies by demographic group"""
    print("\n" + "="*80)
    print("ANALYSIS 4: TIMING × DEMOGRAPHIC INTERACTIONS")
    print("="*80)
    
    dcd_mscs = df[(df['brain_death'] == 0) & (df['is_msc'] == True)].copy()
    
    if 'window_hours_dcd' not in dcd_mscs.columns:
        print("Window data not available. Skipping analysis.")
        return None
    
    print("\nTesting if timing effect varies by demographic characteristics\n")
    
    # Test race × timing interaction
    if 'race' in dcd_mscs.columns:
        print("Race × Timing Interaction:")
        print("-" * 60)
        
        # Get major race categories
        race_counts = dcd_mscs['race'].value_counts()
        major_races = race_counts[race_counts >= 200].index[:3]  # Top 3 races
        
        for race in major_races:
            subset = dcd_mscs[dcd_mscs['race'] == race]
            clean = subset[subset['window_hours_dcd'].notna() & subset['approached'].notna()]
            
            if len(clean) > 50:
                corr, p_val = stats.spearmanr(clean['window_hours_dcd'], clean['approached'])
                print(f"  {race}: ρ={corr:.3f}, p={p_val:.4f} (N={len(clean):,})")
    
    # Test age × timing interaction
    if 'age_group' in dcd_mscs.columns:
        print("\nAge × Timing Interaction:")
        print("-" * 60)
        
        for age_group in dcd_mscs['age_group'].dropna().unique():
            subset = dcd_mscs[dcd_mscs['age_group'] == age_group]
            clean = subset[subset['window_hours_dcd'].notna() & subset['approached'].notna()]
            
            if len(clean) > 50:
                corr, p_val = stats.spearmanr(clean['window_hours_dcd'], clean['approached'])
                print(f"  {age_group}: ρ={corr:.3f}, p={p_val:.4f} (N={len(clean):,})")
    
    print("\n" + "="*80)

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualizations(df):
    """Create demographic segmentation visualizations"""
    print("\nCreating visualizations...")
    
    dcd_mscs = df[(df['brain_death'] == 0) & (df['is_msc'] == True)].copy()
    
    if 'window_category' not in dcd_mscs.columns or 'race' not in dcd_mscs.columns:
        print("Insufficient data for visualizations.")
        return
    
    # Figure: Timing effect by race
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Get top 4 races by count
    race_counts = dcd_mscs['race'].value_counts()
    top_races = race_counts.head(4).index
    
    # Panel A: Approach rate by window category, stratified by race
    for race in top_races:
        subset = dcd_mscs[dcd_mscs['race'] == race]
        approach_by_window = subset.groupby('window_category')['approached'].mean()
        axes[0].plot(range(len(approach_by_window)), approach_by_window.values, 
                     marker='o', label=race, linewidth=2)
    
    axes[0].set_xlabel('Window Length Category')
    axes[0].set_ylabel('Approach Rate')
    axes[0].set_title('Timing Effect by Race')
    axes[0].legend()
    axes[0].set_xticks(range(5))
    axes[0].set_xticklabels(['<12hr', '12-24hr', '24-48hr', '48-72hr', '>72hr'], rotation=45)
    axes[0].grid(True, alpha=0.3)
    
    # Panel B: Window length distribution by race
    window_data = [dcd_mscs[dcd_mscs['race'] == race]['window_hours_dcd'].dropna() 
                   for race in top_races]
    axes[1].boxplot(window_data, labels=top_races)
    axes[1].set_ylabel('Window Length (hours)')
    axes[1].set_xlabel('Race')
    axes[1].set_title('Window Length Distribution by Race')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = FIGURES_DIR / 'demographic_timing_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved figure: {output_file}")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run complete demographic segmentation analysis"""
    print("\n" + "="*80)
    print("DEMOGRAPHIC SEGMENTATION ANALYSIS")
    print("Testing if timing bottleneck is confounded by patient characteristics")
    print("="*80)
    
    # Load data
    df = load_data()
    
    # Calculate time windows
    df = calculate_time_windows(df)
    
    # Categorize demographics
    df = categorize_demographics(df)
    
    # Check if MSC labels exist
    if 'is_msc' not in df.columns:
        print("\n⚠ WARNING: MSC labels not found.")
        print("Please run script 03_msc_sensitivity.py first to generate MSC classifications.")
        return
    
    # Run analyses
    results1 = analyze_timing_by_demographics(df)
    results2 = analyze_window_disparities(df)
    results3 = regression_with_controls(df)
    analyze_interactions(df)
    
    # Create visualizations
    create_visualizations(df)
    
    # Final summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nKey Questions:")
    print("1. Does timing effect hold within demographic groups? → See timing_effect_by_demographics.csv")
    print("2. Do certain groups get shorter windows? → See window_length_disparities.csv")
    print("3. Is there a pure timing effect after controls? → See regression output above")
    print("4. Do demographics interact with timing? → See interaction analysis above")
    print("\n" + "="*80)

if __name__ == '__main__':
    main()
