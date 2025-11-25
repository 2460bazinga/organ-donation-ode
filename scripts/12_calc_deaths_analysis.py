#!/usr/bin/env python3
"""
CalcDeaths Analysis and CALC-Adjusted OPO Performance

This script analyzes the CalcDeaths.csv file and calculates CALC-adjusted
OPO performance metrics according to the OPO Final Rule.

Key metrics:
- Referral Rate = Referrals / CALC Deaths
- Donation Rate = Transplants / CALC Deaths  
- Conversion Efficiency = Transplants / Referrals

This provides the proper denominator for OPO performance comparison,
accounting for DSA size and demographics.

Author: ODE Research Team
Date: November 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
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
# PART 1: EXPLORE CALCDEATHS FILE
# ============================================================================

def explore_calc_deaths():
    """Comprehensive exploration of CalcDeaths.csv"""
    print("="*80)
    print("PART 1: CALCDEATHS FILE EXPLORATION")
    print("="*80)
    
    # Load file
    print("\nLoading CalcDeaths.csv...")
    calc_file = DATA_DIR / 'CalcDeaths.csv'
    
    if not calc_file.exists():
        print(f"ERROR: {calc_file} not found!")
        return None
    
    calc = pd.read_csv(calc_file)
    print(f"Loaded {len(calc):,} rows\n")
    
    # Display structure
    print("File Structure:")
    print("-" * 60)
    print(calc.head(20))
    print()
    
    print("Column Names:")
    print("-" * 60)
    for col in calc.columns:
        print(f"  - {col}")
    print()
    
    print("Data Types:")
    print("-" * 60)
    print(calc.dtypes)
    print()
    
    print("Summary Statistics:")
    print("-" * 60)
    print(calc.describe())
    print()
    
    # Check for key columns
    print("Key Information:")
    print("-" * 60)
    
    if 'opo' in calc.columns or 'OPO' in calc.columns:
        opo_col = 'opo' if 'opo' in calc.columns else 'OPO'
        print(f"OPOs present: {sorted(calc[opo_col].unique())}")
        print(f"Number of OPOs: {calc[opo_col].nunique()}")
    
    if 'year' in calc.columns or 'Year' in calc.columns:
        year_col = 'year' if 'year' in calc.columns else 'Year'
        print(f"Years covered: {sorted(calc[year_col].unique())}")
        print(f"Year range: {calc[year_col].min()} - {calc[year_col].max()}")
    
    # Check for CALC death estimates
    calc_cols = [col for col in calc.columns if 'calc' in col.lower() or 'death' in col.lower()]
    if calc_cols:
        print(f"\nCALC death columns found:")
        for col in calc_cols:
            print(f"  - {col}: mean={calc[col].mean():.0f}, range=[{calc[col].min():.0f}, {calc[col].max():.0f}]")
    
    print("\n" + "="*80)
    
    return calc

# ============================================================================
# PART 2: MERGE WITH REFERRALS DATA
# ============================================================================

def merge_with_referrals(calc):
    """Merge CalcDeaths with referrals data"""
    print("\n" + "="*80)
    print("PART 2: MERGING CALCDEATHS WITH REFERRALS")
    print("="*80)
    
    # Load referrals
    print("\nLoading OPOReferrals.csv...")
    referrals = pd.read_csv(DATA_DIR / 'OPOReferrals.csv', low_memory=False)
    print(f"Loaded {len(referrals):,} referrals")
    
    # Extract year from referrals
    if 'time_referred' in referrals.columns:
        referrals['time_referred'] = pd.to_datetime(referrals['time_referred'], errors='coerce')
        referrals['year'] = referrals['time_referred'].dt.year
    
    # Aggregate referrals by OPO-year
    print("\nAggregating referrals by OPO-year...")
    
    agg_dict = {
        'patient_id': 'count',  # Total referrals
    }
    
    if 'approached' in referrals.columns:
        agg_dict['approached'] = 'sum'
    if 'authorized' in referrals.columns:
        agg_dict['authorized'] = 'sum'
    if 'procured' in referrals.columns:
        agg_dict['procured'] = 'sum'
    if 'transplanted' in referrals.columns:
        agg_dict['transplanted'] = 'sum'
    
    ref_by_opo_year = referrals.groupby(['opo', 'year']).agg(agg_dict).reset_index()
    ref_by_opo_year.rename(columns={'patient_id': 'referrals'}, inplace=True)
    
    print(f"Created {len(ref_by_opo_year)} OPO-year observations")
    print("\nSample:")
    print(ref_by_opo_year.head(10))
    
    # Merge with CalcDeaths
    print("\nMerging with CalcDeaths...")
    
    # Standardize column names for merge
    calc_merge = calc.copy()
    if 'OPO' in calc_merge.columns:
        calc_merge.rename(columns={'OPO': 'opo'}, inplace=True)
    if 'Year' in calc_merge.columns:
        calc_merge.rename(columns={'Year': 'year'}, inplace=True)
    
    merged = ref_by_opo_year.merge(calc_merge, on=['opo', 'year'], how='left')
    
    print(f"Merged dataset: {len(merged)} rows")
    print(f"Rows with CALC deaths: {merged['calc_deaths'].notna().sum() if 'calc_deaths' in merged.columns else 'N/A'}")
    
    print("\nMerged data sample:")
    print(merged.head(10))
    
    return merged

# ============================================================================
# PART 3: CALCULATE CALC-ADJUSTED METRICS
# ============================================================================

def calculate_calc_adjusted_metrics(merged):
    """Calculate CALC-adjusted OPO performance metrics"""
    print("\n" + "="*80)
    print("PART 3: CALC-ADJUSTED OPO PERFORMANCE METRICS")
    print("="*80)
    
    # Determine which CALC column to use
    if 'calc_deaths' in merged.columns:
        calc_col = 'calc_deaths'
    elif 'CALC_deaths' in merged.columns:
        calc_col = 'CALC_deaths'
    else:
        print("\nERROR: No CALC deaths column found!")
        return None
    
    print(f"\nUsing '{calc_col}' as denominator")
    
    # Calculate rates
    merged['referral_rate'] = (merged['referrals'] / merged[calc_col]) * 1000  # Per 1000 CALC deaths
    
    if 'approached' in merged.columns:
        merged['approach_rate'] = (merged['approached'] / merged[calc_col]) * 1000
    
    if 'authorized' in merged.columns:
        merged['authorization_rate'] = (merged['authorized'] / merged[calc_col]) * 1000
    
    if 'transplanted' in merged.columns:
        merged['donation_rate'] = (merged['transplanted'] / merged[calc_col]) * 1000
    
    # Conversion efficiency (internal to referrals)
    merged['conversion_efficiency'] = merged['transplanted'] / merged['referrals'] * 100  # Percent
    
    # Display results
    print("\nCALC-Adjusted Metrics by OPO (averaged across years):")
    print("-" * 80)
    
    opo_summary = merged.groupby('opo').agg({
        calc_col: 'sum',
        'referrals': 'sum',
        'transplanted': 'sum' if 'transplanted' in merged.columns else 'first',
        'referral_rate': 'mean',
        'donation_rate': 'mean' if 'donation_rate' in merged.columns else 'first',
        'conversion_efficiency': 'mean'
    }).round(2)
    
    print(opo_summary)
    
    # Performance variance
    if 'donation_rate' in opo_summary.columns:
        print("\nPerformance Variance:")
        print("-" * 60)
        print(f"Best OPO donation rate: {opo_summary['donation_rate'].max():.2f} per 1000 CALC deaths")
        print(f"Worst OPO donation rate: {opo_summary['donation_rate'].min():.2f} per 1000 CALC deaths")
        print(f"Variance ratio: {opo_summary['donation_rate'].max() / opo_summary['donation_rate'].min():.2f}x")
    
    # Decomposition
    print("\nDonation Rate Decomposition:")
    print("-" * 60)
    print("Donation Rate = Referral Rate × Conversion Efficiency")
    print()
    
    for opo in opo_summary.index:
        ref_rate = opo_summary.loc[opo, 'referral_rate']
        conv_eff = opo_summary.loc[opo, 'conversion_efficiency']
        donation_rate = opo_summary.loc[opo, 'donation_rate'] if 'donation_rate' in opo_summary.columns else 0
        
        print(f"{opo}:")
        print(f"  Referral Rate: {ref_rate:.2f} per 1000 CALC deaths")
        print(f"  Conversion Efficiency: {conv_eff:.2f}%")
        print(f"  Donation Rate: {donation_rate:.2f} per 1000 CALC deaths")
        print()
    
    # Save results
    output_file = OUTPUT_DIR / 'calc_adjusted_opo_performance.csv'
    opo_summary.to_csv(output_file)
    print(f"✓ Saved: {output_file}")
    
    return merged, opo_summary

# ============================================================================
# PART 4: TEMPORAL TRENDS
# ============================================================================

def analyze_temporal_trends(merged):
    """Analyze how CALC-adjusted metrics change over time"""
    print("\n" + "="*80)
    print("PART 4: TEMPORAL TRENDS IN CALC-ADJUSTED METRICS")
    print("="*80)
    
    if 'donation_rate' not in merged.columns:
        print("\nDonation rate not available. Skipping temporal analysis.")
        return
    
    print("\nDonation Rate by Year (all OPOs combined):")
    print("-" * 60)
    
    yearly = merged.groupby('year').agg({
        'calc_deaths': 'sum',
        'referrals': 'sum',
        'transplanted': 'sum'
    })
    
    yearly['donation_rate'] = (yearly['transplanted'] / yearly['calc_deaths']) * 1000
    
    print(yearly)
    
    # Test for trend
    from scipy import stats
    years = yearly.index.values
    rates = yearly['donation_rate'].values
    
    if len(years) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(years, rates)
        
        print(f"\nTrend Analysis:")
        print(f"  Slope: {slope:.3f} per year")
        print(f"  R²: {r_value**2:.3f}")
        print(f"  P-value: {p_value:.4f}")
        
        if p_value < 0.05:
            direction = "increasing" if slope > 0 else "decreasing"
            print(f"  ✓ SIGNIFICANT {direction.upper()} TREND")
        else:
            print(f"  ○ No significant trend")

# ============================================================================
# PART 5: VISUALIZATION
# ============================================================================

def create_visualizations(merged, opo_summary):
    """Create visualizations of CALC-adjusted metrics"""
    print("\n" + "="*80)
    print("PART 5: VISUALIZATIONS")
    print("="*80)
    
    if 'donation_rate' not in merged.columns or 'referral_rate' not in merged.columns:
        print("\nInsufficient data for visualizations.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel A: Donation rate by OPO
    opo_summary['donation_rate'].plot(kind='bar', ax=axes[0, 0], color='steelblue')
    axes[0, 0].set_title('Donation Rate by OPO (per 1000 CALC Deaths)', fontweight='bold')
    axes[0, 0].set_ylabel('Donation Rate')
    axes[0, 0].set_xlabel('OPO')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Panel B: Referral rate vs Conversion efficiency
    axes[0, 1].scatter(opo_summary['referral_rate'], opo_summary['conversion_efficiency'], 
                       s=200, alpha=0.6, color='coral')
    
    for opo in opo_summary.index:
        axes[0, 1].annotate(opo, 
                           (opo_summary.loc[opo, 'referral_rate'], 
                            opo_summary.loc[opo, 'conversion_efficiency']),
                           fontsize=10, ha='center')
    
    axes[0, 1].set_xlabel('Referral Rate (per 1000 CALC Deaths)')
    axes[0, 1].set_ylabel('Conversion Efficiency (%)')
    axes[0, 1].set_title('Referral Rate vs Conversion Efficiency', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Panel C: Donation rate over time
    yearly_avg = merged.groupby('year')['donation_rate'].mean()
    axes[1, 0].plot(yearly_avg.index, yearly_avg.values, marker='o', linewidth=2, color='darkgreen')
    axes[1, 0].set_xlabel('Year')
    axes[1, 0].set_ylabel('Donation Rate (per 1000 CALC Deaths)')
    axes[1, 0].set_title('Donation Rate Trend Over Time', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Panel D: OPO performance decomposition
    x = np.arange(len(opo_summary))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, opo_summary['referral_rate'], width, label='Referral Rate', alpha=0.8)
    axes[1, 1].bar(x + width/2, opo_summary['donation_rate'], width, label='Donation Rate', alpha=0.8)
    
    axes[1, 1].set_xlabel('OPO')
    axes[1, 1].set_ylabel('Rate (per 1000 CALC Deaths)')
    axes[1, 1].set_title('Referral Rate vs Donation Rate by OPO', fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(opo_summary.index)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_file = FIGURES_DIR / 'calc_adjusted_opo_performance.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved figure: {output_file}")
    plt.close()

# ============================================================================
# PART 6: KEY INSIGHTS
# ============================================================================

def summarize_insights(merged, opo_summary):
    """Summarize key insights from CALC-adjusted analysis"""
    print("\n" + "="*80)
    print("KEY INSIGHTS: CALC-ADJUSTED OPO PERFORMANCE")
    print("="*80)
    
    if 'donation_rate' not in opo_summary.columns:
        print("\nInsufficient data for insights.")
        return
    
    print("\n1. PROPER DENOMINATOR")
    print("-" * 60)
    print("Using CALC deaths as denominator accounts for:")
    print("  ✓ DSA size (population)")
    print("  ✓ Demographics (age, cause of death)")
    print("  ✓ Geographic variation")
    print("  ✓ Referral practice differences")
    
    print("\n2. OPO PERFORMANCE VARIANCE")
    print("-" * 60)
    best_opo = opo_summary['donation_rate'].idxmax()
    worst_opo = opo_summary['donation_rate'].idxmin()
    variance = opo_summary['donation_rate'].max() / opo_summary['donation_rate'].min()
    
    print(f"Best OPO ({best_opo}): {opo_summary.loc[best_opo, 'donation_rate']:.2f} per 1000 CALC deaths")
    print(f"Worst OPO ({worst_opo}): {opo_summary.loc[worst_opo, 'donation_rate']:.2f} per 1000 CALC deaths")
    print(f"Performance gap: {variance:.2f}x")
    
    print("\n3. DECOMPOSITION: WHERE IS THE VARIANCE?")
    print("-" * 60)
    
    ref_rate_variance = opo_summary['referral_rate'].max() / opo_summary['referral_rate'].min()
    conv_eff_variance = opo_summary['conversion_efficiency'].max() / opo_summary['conversion_efficiency'].min()
    
    print(f"Referral rate variance: {ref_rate_variance:.2f}x")
    print(f"Conversion efficiency variance: {conv_eff_variance:.2f}x")
    
    if ref_rate_variance > conv_eff_variance:
        print("\n→ REFERRAL RATE drives most of the variance")
        print("  Interpretation: OPOs differ more in getting referrals than converting them")
        print("  Implication: Focus on hospital engagement, not just OPO efficiency")
    else:
        print("\n→ CONVERSION EFFICIENCY drives most of the variance")
        print("  Interpretation: OPOs differ more in converting referrals than getting them")
        print("  Implication: Focus on OPO processes, not just hospital engagement")
    
    print("\n4. COMPARISON TO ORIGINAL FINDINGS")
    print("-" * 60)
    print("Original analysis (referrals as denominator):")
    print("  - Confounded by DSA size and referral practices")
    print("  - Cannot distinguish OPO performance from hospital behavior")
    print()
    print("CALC-adjusted analysis (CALC deaths as denominator):")
    print("  - Separates referral capture from conversion efficiency")
    print("  - Aligns with OPO Final Rule metrics")
    print("  - Enables fair OPO comparison")
    
    print("\n" + "="*80)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run comprehensive CalcDeaths analysis"""
    print("\n" + "="*80)
    print("CALCDEATHS ANALYSIS & CALC-ADJUSTED OPO PERFORMANCE")
    print("="*80)
    
    # Part 1: Explore CalcDeaths file
    calc = explore_calc_deaths()
    
    if calc is None:
        print("\nERROR: Could not load CalcDeaths.csv")
        return
    
    # Part 2: Merge with referrals
    merged = merge_with_referrals(calc)
    
    # Part 3: Calculate CALC-adjusted metrics
    merged, opo_summary = calculate_calc_adjusted_metrics(merged)
    
    # Part 4: Temporal trends
    analyze_temporal_trends(merged)
    
    # Part 5: Visualizations
    create_visualizations(merged, opo_summary)
    
    # Part 6: Key insights
    summarize_insights(merged, opo_summary)
    
    print("\n" + "="*80)
    print("CALCDEATHS ANALYSIS COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Integrate CALC-adjusted metrics into Script 11 (comprehensive robustness)")
    print("2. Update README with CALC-adjusted OPO performance findings")
    print("3. Revise email to H. Adam to mention proper denominator usage")
    print("="*80)

if __name__ == '__main__':
    main()
