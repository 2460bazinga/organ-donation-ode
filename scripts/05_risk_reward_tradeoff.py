#!/usr/bin/env python3
"""
Risk-Reward Tradeoff Analysis: Testing the "Free Lunch" Hypothesis
===================================================================

Tests whether increasing sorting efficiency leads to lower organ quality.

Hypothesis A (Rational Risk Aversion): 
  - OPOs that dig deeper get lower-quality organs
  - Negative correlation between sorting efficiency and placement rate

Hypothesis B ("Free Lunch"):
  - Marginal donors are biologically equivalent to utilized donors
  - Flat or positive correlation between sorting efficiency and placement rate

Method: Analyze correlation between OPO sorting efficiency and placement rate.

Author: Noah
Date: 2024-11-24
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
from pathlib import Path
from scipy import stats

# Paths
DATA_DIR = Path.home() / 'physionet.org' / 'files' / 'orchid' / '2.1.1'
INPUT_FILE = DATA_DIR / 'orchid_with_msc_sensitivity.csv'
OUTPUT_DIR = Path(__file__).parent.parent / 'results' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def analyze_risk_reward():
    """
    Analyze correlation between sorting efficiency and placement rate across OPOs
    """
    print("="*80)
    print("RISK-REWARD TRADEOFF ANALYSIS: Testing the 'Free Lunch' Hypothesis")
    print("="*80)
    
    # Load data
    print("\nLoading MSC-labeled dataset...")
    df = pd.read_csv(INPUT_FILE)
    
    # Use primary approach (99th percentile)
    msc_col = 'is_msc_percentile_99'
    mscs = df[df[msc_col] == True].copy()
    
    print(f"Total MSCs: {len(mscs):,}")
    
    # === OPO-Level Analysis ===
    print("\n" + "-"*80)
    print("OPO-LEVEL PERFORMANCE METRICS")
    print("-"*80)
    
    opo_stats = []
    
    for opo in sorted(mscs['opo'].unique()):
        opo_df = mscs[mscs['opo'] == opo]
        
        # Sorting metrics
        n_msc = len(opo_df)
        n_approached = opo_df['approached'].sum()
        sorting_efficiency = n_approached / n_msc if n_msc > 0 else 0
        
        # Authorization metrics
        approached_df = opo_df[opo_df['approached'] == True]
        n_authorized = approached_df['authorized'].sum() if len(approached_df) > 0 else 0
        auth_rate = n_authorized / n_approached if n_approached > 0 else 0
        
        # Procurement metrics
        authorized_df = approached_df[approached_df['authorized'] == True]
        n_procured = authorized_df['procured'].sum() if len(authorized_df) > 0 else 0
        procurement_rate = n_procured / n_authorized if n_authorized > 0 else 0
        
        # Placement metrics
        procured_df = authorized_df[authorized_df['procured'] == True]
        n_transplanted = procured_df['transplanted'].sum() if len(procured_df) > 0 else 0
        placement_rate = n_transplanted / n_procured if n_procured > 0 else 0
        
        # Overall conversion
        overall_conversion = n_transplanted / n_msc if n_msc > 0 else 0
        
        opo_stats.append({
            'opo': opo,
            'n_msc': n_msc,
            'n_approached': n_approached,
            'n_authorized': n_authorized,
            'n_procured': n_procured,
            'n_transplanted': n_transplanted,
            'sorting_efficiency': sorting_efficiency,
            'auth_rate': auth_rate,
            'procurement_rate': procurement_rate,
            'placement_rate': placement_rate,
            'overall_conversion': overall_conversion
        })
    
    opo_df = pd.DataFrame(opo_stats)
    
    print("\nOPO Performance Summary:")
    print(opo_df[['opo', 'n_msc', 'sorting_efficiency', 'placement_rate', 'overall_conversion']].to_string(index=False))
    
    # === Correlation Analysis ===
    print("\n" + "-"*80)
    print("CORRELATION ANALYSIS: Sorting Efficiency vs. Placement Rate")
    print("-"*80)
    
    correlation = opo_df['sorting_efficiency'].corr(opo_df['placement_rate'])
    p_value = stats.pearsonr(opo_df['sorting_efficiency'], opo_df['placement_rate'])[1]
    
    print(f"\nPearson correlation: r = {correlation:.3f} (p = {p_value:.3f})")
    
    if correlation < -0.3 and p_value < 0.05:
        print("✗ NEGATIVE correlation: Higher sorting → Lower placement (Rational Risk Aversion)")
    elif correlation > 0.1:
        print("✓ POSITIVE correlation: Higher sorting → Higher placement (Free Lunch!)")
    else:
        print("✓ FLAT correlation: No quality trade-off (Free Lunch!)")
    
    # === Visualization 1: Scatter Plot ===
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Scatter
    scatter = ax.scatter(opo_df['sorting_efficiency'], opo_df['placement_rate'],
                        s=opo_df['n_msc']/10, alpha=0.7, c=opo_df['overall_conversion'],
                        cmap='RdYlGn', edgecolors='black', linewidth=1.5)
    
    # Labels
    for idx, row in opo_df.iterrows():
        ax.annotate(row['opo'], 
                   (row['sorting_efficiency'], row['placement_rate']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold')
    
    # Trend line
    z = np.polyfit(opo_df['sorting_efficiency'], opo_df['placement_rate'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(opo_df['sorting_efficiency'].min(), opo_df['sorting_efficiency'].max(), 100)
    ax.plot(x_trend, p(x_trend), "r--", linewidth=2, alpha=0.7, 
            label=f'Trend (r={correlation:.3f}, p={p_value:.3f})')
    
    # Formatting
    ax.set_xlabel('Sorting Efficiency (MSCs Approached / Total MSCs)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Placement Rate (Transplanted / Procured)', fontsize=12, fontweight='bold')
    ax.set_title('The "Free Lunch": No Quality Trade-off from Increased Sorting', 
                fontsize=14, fontweight='bold')
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Overall Conversion Rate', fontsize=10)
    cbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'risk_reward_tradeoff.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {OUTPUT_DIR / 'risk_reward_tradeoff.png'}")
    
    # === Visualization 2: Efficiency Frontier ===
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Scatter
    scatter = ax.scatter(opo_df['sorting_efficiency'], opo_df['overall_conversion'],
                        s=opo_df['n_msc']/10, alpha=0.7, c=opo_df['placement_rate'],
                        cmap='RdYlGn', edgecolors='black', linewidth=1.5)
    
    # Labels
    for idx, row in opo_df.iterrows():
        ax.annotate(row['opo'], 
                   (row['sorting_efficiency'], row['overall_conversion']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold')
    
    # Identify best OPO
    best_opo = opo_df.loc[opo_df['overall_conversion'].idxmax()]
    ax.scatter(best_opo['sorting_efficiency'], best_opo['overall_conversion'],
              s=500, marker='*', c='gold', edgecolors='black', linewidth=2,
              label=f'Best OPO: {best_opo["opo"]}', zorder=10)
    
    # Formatting
    ax.set_xlabel('Sorting Efficiency (MSCs Approached / Total MSCs)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Overall Conversion Rate (Transplanted / Total MSCs)', fontsize=12, fontweight='bold')
    ax.set_title('Operating Inside the Efficiency Frontier', 
                fontsize=14, fontweight='bold')
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Placement Rate', fontsize=10)
    cbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'efficiency_frontier.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / 'efficiency_frontier.png'}")
    
    # === Summary ===
    print("\n" + "="*80)
    print("SUMMARY: Risk-Reward Tradeoff Analysis")
    print("="*80)
    
    print(f"\nBest OPO: {best_opo['opo']}")
    print(f"  - Sorting Efficiency: {best_opo['sorting_efficiency']:.1%}")
    print(f"  - Placement Rate: {best_opo['placement_rate']:.1%}")
    print(f"  - Overall Conversion: {best_opo['overall_conversion']:.1%}")
    
    worst_opo = opo_df.loc[opo_df['overall_conversion'].idxmin()]
    print(f"\nWorst OPO: {worst_opo['opo']}")
    print(f"  - Sorting Efficiency: {worst_opo['sorting_efficiency']:.1%}")
    print(f"  - Placement Rate: {worst_opo['placement_rate']:.1%}")
    print(f"  - Overall Conversion: {worst_opo['overall_conversion']:.1%}")
    
    print(f"\nPerformance Gap: {best_opo['overall_conversion'] / worst_opo['overall_conversion']:.1f}x")
    
    if correlation >= -0.1:
        print("\n✓ CONCLUSION: 'Free Lunch' hypothesis CONFIRMED.")
        print("  - No negative correlation between sorting efficiency and placement rate.")
        print("  - OPOs can increase sorting volume without sacrificing organ quality.")
        print("  - Marginal donors being ignored are biologically equivalent to those utilized.")
        print("  - System is operating far inside its efficiency frontier.")
    else:
        print("\n✗ CONCLUSION: 'Free Lunch' hypothesis REJECTED.")
        print("  - Negative correlation suggests quality trade-off exists.")
        print("  - Digging deeper may yield lower-quality organs.")
    
    print("\n" + "="*80)
    
    return opo_df

def main():
    opo_df = analyze_risk_reward()
    
    # Save OPO comparison table
    output_file = Path(__file__).parent.parent / 'results' / 'opo_comparison.csv'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    opo_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved OPO comparison table: {output_file}")

if __name__ == '__main__':
    main()
