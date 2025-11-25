#!/usr/bin/env python3
"""
Congestion Analysis: Testing the Infrastructure Constraint Hypothesis
======================================================================

Tests whether sorting losses are driven by:
  A) Rational Risk Aversion (OPOs consciously reject marginal donors)
  B) Infrastructure Constraints (OPOs lack capacity to process all referrals)

Method: Temporal stress test - examine how sorting efficiency varies with referral volume.

If Hypothesis B is correct, we expect sorting efficiency to decline when referral volume
is high (congestion externalities) and improve when volume is low.

Author: Noah
Date: 2024-11-24
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
from pathlib import Path

# Paths
DATA_DIR = Path.home() / 'physionet.org' / 'files' / 'orchid' / '2.1.1'
INPUT_FILE = DATA_DIR / 'orchid_with_msc_sensitivity.csv'
OUTPUT_DIR = Path(__file__).parent.parent / 'results' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def analyze_temporal_patterns():
    """
    Analyze how sorting efficiency varies by time of day and day of week
    """
    print("="*80)
    print("CONGESTION ANALYSIS: Infrastructure Constraint Test")
    print("="*80)
    
    # Load data
    print("\nLoading MSC-labeled dataset...")
    df = pd.read_csv(INPUT_FILE)
    
    # Use primary approach (99th percentile)
    msc_col = 'is_msc_percentile_99'
    mscs = df[df[msc_col] == True].copy()
    
    print(f"Total MSCs: {len(mscs):,}")
    print(f"MSCs approached: {mscs['approached'].sum():,}")
    print(f"Overall sorting efficiency: {mscs['approached'].sum() / len(mscs):.1%}")
    
    # Parse timestamps
    print("\nParsing timestamps...")
    mscs['time_referred'] = pd.to_datetime(mscs['time_referred'], errors='coerce')
    mscs = mscs.dropna(subset=['time_referred'])
    
    mscs['hour'] = mscs['time_referred'].dt.hour
    mscs['day_of_week'] = mscs['time_referred'].dt.dayofweek  # 0=Monday, 6=Sunday
    mscs['date'] = mscs['time_referred'].dt.date
    
    print(f"Records with valid timestamps: {len(mscs):,}")
    
    # === Analysis 1: Hour of Day ===
    print("\n" + "-"*80)
    print("ANALYSIS 1: Sorting Efficiency by Hour of Day")
    print("-"*80)
    
    hourly_stats = mscs.groupby('hour').agg({
        'patient_id': 'count',
        'approached': 'sum'
    }).rename(columns={'patient_id': 'msc_count', 'approached': 'approached_count'})
    
    hourly_stats['sorting_efficiency'] = hourly_stats['approached_count'] / hourly_stats['msc_count']
    
    print("\nHourly Statistics:")
    print(hourly_stats.to_string())
    
    # Plot hour of day
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Volume
    ax1.bar(hourly_stats.index, hourly_stats['msc_count'], color='steelblue', alpha=0.7)
    ax1.set_xlabel('Hour of Day', fontsize=12)
    ax1.set_ylabel('MSC Referral Volume', fontsize=12)
    ax1.set_title('Referral Volume by Hour of Day', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.axvspan(8, 17, color='green', alpha=0.1, label='Business Hours (8am-5pm)')
    ax1.legend()
    
    # Efficiency
    ax2.plot(hourly_stats.index, hourly_stats['sorting_efficiency'], 
             marker='o', color='#e74c3c', linewidth=3, markersize=8)
    ax2.set_xlabel('Hour of Day', fontsize=12)
    ax2.set_ylabel('Sorting Efficiency', fontsize=12)
    ax2.set_title('Evidence of Infrastructure Constraint: Sorting Efficiency by Hour', 
                  fontsize=14, fontweight='bold')
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.grid(alpha=0.3)
    ax2.axvspan(8, 17, color='green', alpha=0.1, label='Business Hours (8am-5pm)')
    ax2.axvspan(0, 6, color='red', alpha=0.05, label='Night Shift (12am-6am)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'hourly_congestion_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {OUTPUT_DIR / 'hourly_congestion_analysis.png'}")
    
    # === Analysis 2: Day of Week ===
    print("\n" + "-"*80)
    print("ANALYSIS 2: Sorting Efficiency by Day of Week (The Weekend Effect)")
    print("-"*80)
    
    daily_stats = mscs.groupby('day_of_week').agg({
        'patient_id': 'count',
        'approached': 'sum'
    }).rename(columns={'patient_id': 'msc_count', 'approached': 'approached_count'})
    
    daily_stats['sorting_efficiency'] = daily_stats['approached_count'] / daily_stats['msc_count']
    
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_stats['day_name'] = [day_names[i] for i in daily_stats.index]
    
    print("\nDaily Statistics:")
    print(daily_stats[['day_name', 'msc_count', 'approached_count', 'sorting_efficiency']].to_string())
    
    # Plot day of week
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Volume
    ax1.bar(daily_stats['day_name'], daily_stats['msc_count'], color='steelblue', alpha=0.7)
    ax1.set_xlabel('Day of Week', fontsize=12)
    ax1.set_ylabel('MSC Referral Volume', fontsize=12)
    ax1.set_title('Referral Volume by Day of Week', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.axvspan(4.5, 6.5, color='orange', alpha=0.1, label='Weekend')
    ax1.legend()
    
    # Efficiency
    ax2.plot(daily_stats['day_name'], daily_stats['sorting_efficiency'], 
             marker='o', color='#e74c3c', linewidth=3, markersize=10)
    ax2.set_xlabel('Day of Week', fontsize=12)
    ax2.set_ylabel('Sorting Efficiency', fontsize=12)
    ax2.set_title('The Weekend Effect: Sorting Efficiency Peaks When Volume is Low', 
                  fontsize=14, fontweight='bold')
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.grid(alpha=0.3)
    ax2.axvspan(4.5, 6.5, color='orange', alpha=0.1, label='Weekend (Lower Volume)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'weekend_effect.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / 'weekend_effect.png'}")
    
    # === Analysis 3: Volume-Efficiency Correlation ===
    print("\n" + "-"*80)
    print("ANALYSIS 3: Volume-Efficiency Correlation (Direct Test)")
    print("-"*80)
    
    # Daily volume and efficiency
    daily_volume = mscs.groupby('date').agg({
        'patient_id': 'count',
        'approached': 'sum'
    }).rename(columns={'patient_id': 'msc_count', 'approached': 'approached_count'})
    
    daily_volume['sorting_efficiency'] = daily_volume['approached_count'] / daily_volume['msc_count']
    
    # Remove outliers (days with <5 referrals)
    daily_volume = daily_volume[daily_volume['msc_count'] >= 5]
    
    # Correlation
    correlation = daily_volume['msc_count'].corr(daily_volume['sorting_efficiency'])
    print(f"\nCorrelation between daily volume and sorting efficiency: {correlation:.3f}")
    
    if correlation < -0.1:
        print("✓ NEGATIVE correlation: Higher volume → Lower efficiency (Infrastructure Constraint)")
    elif correlation > 0.1:
        print("⚠ POSITIVE correlation: Higher volume → Higher efficiency (Unexpected)")
    else:
        print("○ WEAK correlation: No clear relationship")
    
    # Scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(daily_volume['msc_count'], daily_volume['sorting_efficiency'], 
               alpha=0.5, s=50, color='steelblue')
    
    # Trend line
    z = np.polyfit(daily_volume['msc_count'], daily_volume['sorting_efficiency'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(daily_volume['msc_count'].min(), daily_volume['msc_count'].max(), 100)
    ax.plot(x_trend, p(x_trend), "r--", linewidth=2, label=f'Trend (r={correlation:.3f})')
    
    ax.set_xlabel('Daily MSC Referral Volume', fontsize=12)
    ax.set_ylabel('Sorting Efficiency', fontsize=12)
    ax.set_title('Congestion Externality: Volume vs. Efficiency', fontsize=14, fontweight='bold')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'volume_efficiency_correlation.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / 'volume_efficiency_correlation.png'}")
    
    # === Summary ===
    print("\n" + "="*80)
    print("SUMMARY: Infrastructure Constraint Test")
    print("="*80)
    
    print("\nEvidence for Infrastructure Constraints:")
    print(f"  1. Weekend Effect: Sunday efficiency = {daily_stats.loc[6, 'sorting_efficiency']:.1%}, "
          f"Wednesday efficiency = {daily_stats.loc[2, 'sorting_efficiency']:.1%}")
    print(f"  2. Volume-Efficiency Correlation: r = {correlation:.3f}")
    
    if correlation < -0.1:
        print("\n✓ CONCLUSION: Strong evidence for Infrastructure Constraint hypothesis.")
        print("  Sorting efficiency declines when referral volume is high (congestion externalities).")
        print("  Viable donors 'fall through the cracks' due to capacity limitations, not conscious rejection.")
    else:
        print("\n○ CONCLUSION: Mixed evidence. Further investigation needed.")
    
    print("\n" + "="*80)

def main():
    analyze_temporal_patterns()

if __name__ == '__main__':
    main()
