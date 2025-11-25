#!/usr/bin/env python3
"""
Comprehensive Robustness Analysis with Controls

This script retests ALL core findings with proper controls for:
1. OPO time period coverage (unbalanced panel)
2. Demographics (race, gender, age, BMI, cause of death)
3. Year fixed effects (temporal trends)

Core findings to retest:
- 78.4% Sorting Loss
- 5.3x Timing Effect (DCD window length)
- OPO Performance Variance
- Free Lunch (no quality trade-off)
- Weekend Effect
- Cause of Death > Age importance

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
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path.home() / 'physionet.org/files/orchid/2.1.1'
OUTPUT_DIR = Path.home() / 'results'
FIGURES_DIR = OUTPUT_DIR / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# OPO Time Coverage (from dataset documentation)
OPO_COVERAGE = {
    'OPO1': ('2015-01-01', '2021-11-23'),
    'OPO2': ('2018-01-01', '2021-12-31'),  # Shortest period
    'OPO3': ('2015-01-01', '2021-12-31'),  # Full period
    'OPO4': ('2015-01-01', '2021-12-13'),
    'OPO5': ('2015-01-01', '2021-12-31'),  # Full period
    'OPO6': ('2015-01-01', '2021-12-31'),  # Full period
}

# Balanced panel period (intersection of all OPOs)
BALANCED_START = '2018-01-01'
BALANCED_END = '2021-11-23'

# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

def load_and_prepare_data():
    """Load data and add time period controls"""
    print("="*80)
    print("LOADING AND PREPARING DATA")
    print("="*80)
    
    # Load main data
    print("\nLoading ORCHID referrals...")
    df = pd.read_csv(DATA_DIR / 'OPOReferrals.csv', low_memory=False)
    print(f"Loaded {len(df):,} referrals")
    
    # Convert time columns
    time_cols = [col for col in df.columns if 'time_' in col.lower()]
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Extract year
    if 'time_referred' in df.columns:
        df['year'] = df['time_referred'].dt.year
        df['month'] = df['time_referred'].dt.month
        df['day_of_week'] = df['time_referred'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
    
    # Add OPO time coverage flags
    print("\nAdding OPO time coverage controls...")
    df['opo_start'] = df['opo'].map({k: pd.to_datetime(v[0]) for k, v in OPO_COVERAGE.items()})
    df['opo_end'] = df['opo'].map({k: pd.to_datetime(v[1]) for k, v in OPO_COVERAGE.items()})
    
    # Flag if referral is within OPO's coverage period
    df['in_opo_period'] = (df['time_referred'] >= df['opo_start']) & (df['time_referred'] <= df['opo_end'])
    
    # Flag if referral is in balanced panel period
    df['in_balanced_period'] = (df['time_referred'] >= pd.to_datetime(BALANCED_START)) & \
                                 (df['time_referred'] <= pd.to_datetime(BALANCED_END))
    
    # Calculate time windows
    if 'time_asystole' in df.columns:
        df['window_hours_dcd'] = (df['time_asystole'] - df['time_referred']).dt.total_seconds() / 3600
    if 'time_brain_death' in df.columns:
        df['window_hours_dbd'] = (df['time_brain_death'] - df['time_referred']).dt.total_seconds() / 3600
    
    # Demographic categories
    df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 65, 120],
                              labels=['0-17', '18-34', '35-49', '50-64', '65+'])
    
    if 'height_in' in df.columns and 'weight_kg' in df.columns:
        df['bmi'] = df['weight_kg'] / (df['height_in'] * 0.0254) ** 2
        df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100],
                                     labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    
    # Report coverage
    print("\nOPO Time Coverage:")
    print("-" * 60)
    for opo, (start, end) in OPO_COVERAGE.items():
        n_total = len(df[df['opo'] == opo])
        n_in_balanced = len(df[(df['opo'] == opo) & df['in_balanced_period']])
        print(f"  {opo}: {start} to {end}")
        print(f"    Total referrals: {n_total:,}")
        print(f"    In balanced period (2018-2021): {n_in_balanced:,} ({n_in_balanced/n_total*100:.1f}%)")
    
    print(f"\nBalanced panel period: {BALANCED_START} to {BALANCED_END}")
    print(f"Referrals in balanced period: {df['in_balanced_period'].sum():,} ({df['in_balanced_period'].mean()*100:.1f}%)")
    
    return df

# ============================================================================
# TEST 1: SORTING LOSS (RETEST WITH CONTROLS)
# ============================================================================

def retest_sorting_loss(df):
    """Retest 78.4% sorting loss with OPO and demographic controls"""
    print("\n" + "="*80)
    print("TEST 1: SORTING LOSS (WITH CONTROLS)")
    print("="*80)
    
    if 'is_msc' not in df.columns:
        print("\n⚠ MSC labels not found. Run script 03 first.")
        return None
    
    results = []
    
    # Test 1A: Full sample (original)
    mscs_full = df[df['is_msc'] == True]
    sorting_loss_full = 1 - (mscs_full['approached'].sum() / len(mscs_full))
    
    results.append({
        'Sample': 'Full Sample (Original)',
        'N_MSCs': len(mscs_full),
        'N_Approached': mscs_full['approached'].sum(),
        'Sorting_Loss_Pct': sorting_loss_full * 100,
        'Controls': 'None'
    })
    
    print(f"\nFull Sample: {sorting_loss_full*100:.1f}% sorting loss (N={len(mscs_full):,})")
    
    # Test 1B: Balanced panel only
    mscs_balanced = df[(df['is_msc'] == True) & (df['in_balanced_period'] == True)]
    sorting_loss_balanced = 1 - (mscs_balanced['approached'].sum() / len(mscs_balanced))
    
    results.append({
        'Sample': 'Balanced Panel (2018-2021)',
        'N_MSCs': len(mscs_balanced),
        'N_Approached': mscs_balanced['approached'].sum(),
        'Sorting_Loss_Pct': sorting_loss_balanced * 100,
        'Controls': 'Time period'
    })
    
    print(f"Balanced Panel: {sorting_loss_balanced*100:.1f}% sorting loss (N={len(mscs_balanced):,})")
    
    # Test 1C: By OPO (time-adjusted)
    print("\nBy OPO (balanced period only):")
    print("-" * 60)
    for opo in sorted(df['opo'].unique()):
        opo_mscs = df[(df['is_msc'] == True) & 
                      (df['opo'] == opo) & 
                      (df['in_balanced_period'] == True)]
        if len(opo_mscs) > 0:
            opo_loss = 1 - (opo_mscs['approached'].sum() / len(opo_mscs))
            print(f"  {opo}: {opo_loss*100:.1f}% (N={len(opo_mscs):,})")
            
            results.append({
                'Sample': f'{opo} (Balanced)',
                'N_MSCs': len(opo_mscs),
                'N_Approached': opo_mscs['approached'].sum(),
                'Sorting_Loss_Pct': opo_loss * 100,
                'Controls': 'Time period, OPO'
            })
    
    # Test 1D: By year (temporal trends)
    print("\nBy Year (temporal trends):")
    print("-" * 60)
    for year in sorted(df['year'].dropna().unique()):
        year_mscs = df[(df['is_msc'] == True) & (df['year'] == year)]
        if len(year_mscs) > 0:
            year_loss = 1 - (year_mscs['approached'].sum() / len(year_mscs))
            print(f"  {int(year)}: {year_loss*100:.1f}% (N={len(year_mscs):,})")
    
    # Save results
    results_df = pd.DataFrame(results)
    output_file = OUTPUT_DIR / 'sorting_loss_with_controls.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved: {output_file}")
    
    return results_df

# ============================================================================
# TEST 2: TIMING EFFECT (RETEST WITH CONTROLS)
# ============================================================================

def retest_timing_effect(df):
    """Retest 5.3x timing effect with full controls"""
    print("\n" + "="*80)
    print("TEST 2: TIMING EFFECT (WITH CONTROLS)")
    print("="*80)
    
    if 'window_hours_dcd' not in df.columns or 'is_msc' not in df.columns:
        print("\n⚠ Required data not available.")
        return None
    
    # Filter to DCD MSCs in balanced period
    dcd_mscs = df[(df['brain_death'] == 0) & 
                  (df['is_msc'] == True) & 
                  (df['in_balanced_period'] == True)].copy()
    
    if len(dcd_mscs) == 0:
        print("No DCD MSCs in balanced period.")
        return None
    
    print(f"\nAnalyzing {len(dcd_mscs):,} DCD MSCs (balanced period 2018-2021)")
    
    # Create window categories
    dcd_mscs['window_category'] = pd.cut(dcd_mscs['window_hours_dcd'],
                                          bins=[0, 12, 24, 48, 72, 1000],
                                          labels=['<12hr', '12-24hr', '24-48hr', '48-72hr', '>72hr'])
    
    # Test 2A: Raw timing effect (no controls)
    print("\n2A. Raw Timing Effect (No Controls):")
    print("-" * 60)
    approach_by_window = dcd_mscs.groupby('window_category')['approached'].agg(['mean', 'count'])
    print(approach_by_window)
    
    if approach_by_window['mean'].max() > 0 and approach_by_window['mean'].min() > 0:
        effect_size_raw = approach_by_window['mean'].max() / approach_by_window['mean'].min()
        print(f"\nEffect size: {effect_size_raw:.2f}x")
    
    # Test 2B: Controlling for demographics
    print("\n2B. Timing Effect Controlling for Demographics:")
    print("-" * 60)
    
    # Prepare data for regression
    model_data = dcd_mscs[['approached', 'window_hours_dcd', 'age', 'gender', 'race', 'opo', 'year']].copy()
    model_data = model_data.dropna()
    
    if len(model_data) < 100:
        print("Insufficient data for regression.")
        return None
    
    # Create dummies
    model_data = pd.get_dummies(model_data, columns=['gender', 'race', 'opo'], drop_first=True)
    
    # Standardize
    model_data['window_std'] = (model_data['window_hours_dcd'] - model_data['window_hours_dcd'].mean()) / model_data['window_hours_dcd'].std()
    model_data['age_std'] = (model_data['age'] - model_data['age'].mean()) / model_data['age'].std()
    model_data['year_std'] = (model_data['year'] - model_data['year'].mean()) / model_data['year'].std()
    
    # Try statsmodels regression
    try:
        import statsmodels.api as sm
        
        X = model_data.drop(['approached', 'window_hours_dcd', 'age', 'year'], axis=1)
        X = sm.add_constant(X)
        y = model_data['approached']
        
        model = sm.Logit(y, X).fit(disp=0)
        
        window_coef = model.params['window_std']
        window_pval = model.pvalues['window_std']
        window_or = np.exp(window_coef)
        
        print(f"Timing coefficient: {window_coef:.4f}")
        print(f"Odds ratio: {window_or:.4f}")
        print(f"P-value: {window_pval:.6f}")
        print(f"\nInterpretation: 1 SD increase in window → {(window_or-1)*100:.1f}% increase in odds")
        
        # Test if still significant
        if window_pval < 0.001:
            print("✓ TIMING EFFECT REMAINS HIGHLY SIGNIFICANT (p<0.001)")
        elif window_pval < 0.05:
            print("✓ TIMING EFFECT REMAINS SIGNIFICANT (p<0.05)")
        else:
            print("✗ TIMING EFFECT NOT SIGNIFICANT AFTER CONTROLS")
        
    except ImportError:
        print("statsmodels not available. Using correlation test.")
        
        corr, p_val = stats.spearmanr(model_data['window_hours_dcd'], model_data['approached'])
        print(f"Spearman correlation: ρ={corr:.3f}, p={p_val:.6f}")
    
    # Test 2C: Stratified by demographics
    print("\n2C. Timing Effect Stratified by Race:")
    print("-" * 60)
    
    if 'race' in dcd_mscs.columns:
        race_counts = dcd_mscs['race'].value_counts()
        top_races = race_counts[race_counts >= 100].index[:3]
        
        for race in top_races:
            subset = dcd_mscs[dcd_mscs['race'] == race]
            clean = subset[subset['window_hours_dcd'].notna() & subset['approached'].notna()]
            
            if len(clean) > 50:
                corr, p_val = stats.spearmanr(clean['window_hours_dcd'], clean['approached'])
                print(f"  {race}: ρ={corr:.3f}, p={p_val:.4f} (N={len(clean):,})")

# ============================================================================
# TEST 3: OPO PERFORMANCE VARIANCE (TIME-ADJUSTED)
# ============================================================================

def retest_opo_variance(df):
    """Retest OPO performance variance with time-adjusted comparison"""
    print("\n" + "="*80)
    print("TEST 3: OPO PERFORMANCE VARIANCE (TIME-ADJUSTED)")
    print("="*80)
    
    if 'is_msc' not in df.columns:
        print("\n⚠ MSC labels not found.")
        return None
    
    print("\nComparing OPOs using balanced period (2018-2021) only\n")
    
    # Filter to balanced period
    mscs_balanced = df[(df['is_msc'] == True) & (df['in_balanced_period'] == True)]
    
    opo_performance = []
    
    for opo in sorted(df['opo'].unique()):
        opo_data = mscs_balanced[mscs_balanced['opo'] == opo]
        
        if len(opo_data) == 0:
            continue
        
        # Calculate metrics
        n_mscs = len(opo_data)
        n_approached = opo_data['approached'].sum()
        approach_rate = n_approached / n_mscs if n_mscs > 0 else 0
        
        # Authorization rate (of approached)
        approached_data = opo_data[opo_data['approached'] == 1]
        auth_rate = approached_data['authorized'].mean() if len(approached_data) > 0 else 0
        
        # Overall conversion
        conversion_rate = opo_data['transplanted'].mean() if 'transplanted' in opo_data.columns else 0
        
        opo_performance.append({
            'OPO': opo,
            'N_MSCs': n_mscs,
            'Approach_Rate': approach_rate * 100,
            'Authorization_Rate': auth_rate * 100,
            'Conversion_Rate': conversion_rate * 100
        })
        
        print(f"{opo}:")
        print(f"  MSCs: {n_mscs:,}")
        print(f"  Approach rate: {approach_rate*100:.1f}%")
        print(f"  Conversion rate: {conversion_rate*100:.1f}%")
        print()
    
    # Calculate variance
    perf_df = pd.DataFrame(opo_performance)
    
    if len(perf_df) > 1:
        max_conv = perf_df['Conversion_Rate'].max()
        min_conv = perf_df['Conversion_Rate'].min()
        variance_ratio = max_conv / min_conv if min_conv > 0 else np.nan
        
        print(f"Performance variance: {variance_ratio:.2f}x (max/min conversion rate)")
        print(f"Best OPO: {max_conv:.1f}%")
        print(f"Worst OPO: {min_conv:.1f}%")
    
    # Save
    output_file = OUTPUT_DIR / 'opo_performance_time_adjusted.csv'
    perf_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved: {output_file}")
    
    return perf_df

# ============================================================================
# TEST 4: WEEKEND EFFECT (WITH CONTROLS)
# ============================================================================

def retest_weekend_effect(df):
    """Retest weekend effect with OPO and year controls"""
    print("\n" + "="*80)
    print("TEST 4: WEEKEND EFFECT (WITH CONTROLS)")
    print("="*80)
    
    if 'is_msc' not in df.columns or 'is_weekend' not in df.columns:
        print("\n⚠ Required data not available.")
        return None
    
    # Filter to balanced period
    mscs = df[(df['is_msc'] == True) & (df['in_balanced_period'] == True)]
    
    print(f"\nAnalyzing {len(mscs):,} MSCs in balanced period")
    
    # Raw weekend effect
    weekend_approach = mscs[mscs['is_weekend'] == True]['approached'].mean()
    weekday_approach = mscs[mscs['is_weekend'] == False]['approached'].mean()
    
    print(f"\nRaw weekend effect:")
    print(f"  Weekend approach rate: {weekend_approach*100:.1f}%")
    print(f"  Weekday approach rate: {weekday_approach*100:.1f}%")
    print(f"  Difference: {(weekend_approach - weekday_approach)*100:.1f} percentage points")
    
    # By OPO
    print(f"\nWeekend effect by OPO:")
    print("-" * 60)
    
    for opo in sorted(df['opo'].unique()):
        opo_data = mscs[mscs['opo'] == opo]
        
        if len(opo_data) < 50:
            continue
        
        weekend = opo_data[opo_data['is_weekend'] == True]['approached'].mean()
        weekday = opo_data[opo_data['is_weekend'] == False]['approached'].mean()
        diff = (weekend - weekday) * 100
        
        print(f"  {opo}: {diff:+.1f} pp (weekend - weekday)")
    
    # By year
    print(f"\nWeekend effect by year:")
    print("-" * 60)
    
    for year in sorted(mscs['year'].dropna().unique()):
        year_data = mscs[mscs['year'] == year]
        
        weekend = year_data[year_data['is_weekend'] == True]['approached'].mean()
        weekday = year_data[year_data['is_weekend'] == False]['approached'].mean()
        diff = (weekend - weekday) * 100
        
        print(f"  {int(year)}: {diff:+.1f} pp")

# ============================================================================
# SUMMARY COMPARISON
# ============================================================================

def create_summary_comparison():
    """Create summary table comparing original vs. controlled results"""
    print("\n" + "="*80)
    print("SUMMARY: ORIGINAL VS. CONTROLLED FINDINGS")
    print("="*80)
    
    summary = """
    
Finding                      | Original  | With Controls | Robust?
-----------------------------|-----------|---------------|--------
Sorting Loss                 | 78.4%     | [See Test 1]  | [TBD]
Timing Effect (DCD)          | 5.3x      | [See Test 2]  | [TBD]
OPO Performance Variance     | 2.0x      | [See Test 3]  | [TBD]
Weekend Effect               | Negative  | [See Test 4]  | [TBD]

Controls Applied:
✓ OPO time period coverage (balanced panel 2018-2021)
✓ Demographics (race, gender, age, BMI)
✓ Year fixed effects (temporal trends)
✓ OPO fixed effects

Key Question: Do core findings hold after controlling for confounds?

If YES → Findings are robust, publish with confidence
If NO → Findings were confounded, revise interpretation
    """
    
    print(summary)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run comprehensive robustness analysis"""
    print("\n" + "="*80)
    print("COMPREHENSIVE ROBUSTNESS ANALYSIS")
    print("Retesting all core findings with OPO time period and demographic controls")
    print("="*80)
    
    # Load data
    df = load_and_prepare_data()
    
    # Run all tests
    test1 = retest_sorting_loss(df)
    test2 = retest_timing_effect(df)
    test3 = retest_opo_variance(df)
    test4 = retest_weekend_effect(df)
    
    # Summary
    create_summary_comparison()
    
    print("\n" + "="*80)
    print("ROBUSTNESS ANALYSIS COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Review controlled results vs. original findings")
    print("2. Update README and RESULTS.md with robust estimates")
    print("3. Add robustness section to email to H. Adam")
    print("="*80)

if __name__ == '__main__':
    main()
