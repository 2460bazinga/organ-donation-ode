#!/usr/bin/env python3
"""
Time Window Analysis: Is Timing the DCD Bottleneck?
====================================================

Core Question: Why is DCD approach rate 10% while DBD is 94%?

Hypothesis: DCD requires faster action than OPOs can provide. The time window
between referral and withdrawal/death is too short for proactive cultivation.

Testable Predictions:
1. DCD referrals that ARE approached have longer time windows than those NOT approached
2. DBD referrals have longer time windows than DCD referrals (more forgiving timeline)
3. Time-to-approach is shorter for DCD than DBD (rushed)
4. DCD approach probability increases with window length (more time = more likely)

Data-Driven Approach:
- NO assumptions about "ideal" timelines
- Let the data show us what windows exist
- Compare approached vs. not-approached within pathways
- Compare DBD vs. DCD timelines

Author: Noah
Date: 2024-11-25
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
from pathlib import Path
from scipy import stats
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path.home() / 'physionet.org' / 'files' / 'orchid' / '2.1.1'
INPUT_FILE = DATA_DIR / 'orchid_with_msc_sensitivity.csv'
OUTPUT_DIR = Path.home() / 'results'
FIGURES_DIR = OUTPUT_DIR / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "="*80)
print("TIME WINDOW ANALYSIS: Is Timing the DCD Bottleneck?")
print("="*80)

# Load data
print("\nLoading data...")
df = pd.read_csv(INPUT_FILE)
mscs = df[df['is_msc_percentile_99'] == True].copy()

print(f"Total MSCs: {len(mscs):,}")

# Parse all temporal columns
temporal_cols = ['time_referred', 'time_approached', 'time_authorized', 'time_procured',
                'time_brain_death', 'time_asystole']

for col in temporal_cols:
    if col in mscs.columns:
        mscs[col] = pd.to_datetime(mscs[col], errors='coerce')

# Identify pathways
mscs['pathway'] = mscs['brain_death'].map({True: 'DBD', False: 'DCD'})

print(f"DBD MSCs: {(mscs['pathway'] == 'DBD').sum():,}")
print(f"DCD MSCs: {(mscs['pathway'] == 'DCD').sum():,}")

# ============================================================================
# PHASE 1: Data Availability Check
# ============================================================================

print("\n" + "="*80)
print("PHASE 1: Temporal Data Availability")
print("="*80)

print("\nAvailable temporal variables:")
for col in temporal_cols:
    if col in mscs.columns:
        non_null = mscs[col].notna().sum()
        pct = non_null / len(mscs) * 100
        print(f"  {col:20s}: {non_null:7,} ({pct:5.1f}%)")

# ============================================================================
# PHASE 2: Define Time Windows
# ============================================================================

print("\n" + "="*80)
print("PHASE 2: Calculate Time Windows")
print("="*80)

# For DBD: time_referred → time_brain_death (when donor becomes available)
mscs['window_to_brain_death'] = (mscs['time_brain_death'] - mscs['time_referred']).dt.total_seconds() / 3600

# For DCD: time_referred → time_asystole (when donor becomes available)
mscs['window_to_asystole'] = (mscs['time_asystole'] - mscs['time_referred']).dt.total_seconds() / 3600

# For approached cases: time_referred → time_approached
mscs['window_to_approach'] = (mscs['time_approached'] - mscs['time_referred']).dt.total_seconds() / 3600

# Create unified "window to outcome" variable
mscs['window_to_outcome'] = np.where(
    mscs['pathway'] == 'DBD',
    mscs['window_to_brain_death'],
    mscs['window_to_asystole']
)

# Filter out negative or extreme values (data errors)
mscs['window_to_outcome'] = mscs['window_to_outcome'].apply(
    lambda x: x if (0 < x < 1000) else np.nan
)
mscs['window_to_approach'] = mscs['window_to_approach'].apply(
    lambda x: x if (0 < x < 1000) else np.nan
)

print("\nTime window statistics (hours):")
print("\nDBD (Referral → Brain Death):")
dbd_windows = mscs[mscs['pathway'] == 'DBD']['window_to_brain_death'].dropna()
if len(dbd_windows) > 0:
    print(f"  n = {len(dbd_windows):,}")
    print(f"  Mean: {dbd_windows.mean():.1f} hours")
    print(f"  Median: {dbd_windows.median():.1f} hours")
    print(f"  25th percentile: {dbd_windows.quantile(0.25):.1f} hours")
    print(f"  75th percentile: {dbd_windows.quantile(0.75):.1f} hours")

print("\nDCD (Referral → Asystole):")
dcd_windows = mscs[mscs['pathway'] == 'DCD']['window_to_asystole'].dropna()
if len(dcd_windows) > 0:
    print(f"  n = {len(dcd_windows):,}")
    print(f"  Mean: {dcd_windows.mean():.1f} hours")
    print(f"  Median: {dcd_windows.median():.1f} hours")
    print(f"  25th percentile: {dcd_windows.quantile(0.25):.1f} hours")
    print(f"  75th percentile: {dcd_windows.quantile(0.75):.1f} hours")

# Statistical comparison
if len(dbd_windows) > 0 and len(dcd_windows) > 0:
    t_stat, p_value = stats.ttest_ind(dbd_windows, dcd_windows)
    print(f"\nDBD vs DCD window comparison:")
    print(f"  Difference in medians: {dbd_windows.median() - dcd_windows.median():.1f} hours")
    print(f"  T-test: t={t_stat:.2f}, p={p_value:.4f}")
    if p_value < 0.05:
        if dbd_windows.median() > dcd_windows.median():
            print("  ✓ DBD has significantly LONGER windows than DCD")
        else:
            print("  ✓ DCD has significantly LONGER windows than DBD")
    else:
        print("  ○ No significant difference in window lengths")

# ============================================================================
# PHASE 3: Approached vs Not-Approached (Within Pathway)
# ============================================================================

print("\n" + "="*80)
print("PHASE 3: Time Windows - Approached vs Not-Approached")
print("="*80)

# DBD Analysis
print("\n--- DBD Pathway ---")
dbd_mscs = mscs[mscs['pathway'] == 'DBD'].copy()
dbd_approached = dbd_mscs[dbd_mscs['approached'] == True]['window_to_brain_death'].dropna()
dbd_not_approached = dbd_mscs[dbd_mscs['approached'] == False]['window_to_brain_death'].dropna()

if len(dbd_approached) > 0 and len(dbd_not_approached) > 0:
    print(f"\nApproached (n={len(dbd_approached):,}):")
    print(f"  Median window: {dbd_approached.median():.1f} hours")
    print(f"  IQR: [{dbd_approached.quantile(0.25):.1f}, {dbd_approached.quantile(0.75):.1f}]")
    
    print(f"\nNot Approached (n={len(dbd_not_approached):,}):")
    print(f"  Median window: {dbd_not_approached.median():.1f} hours")
    print(f"  IQR: [{dbd_not_approached.quantile(0.25):.1f}, {dbd_not_approached.quantile(0.75):.1f}]")
    
    # Mann-Whitney U test (non-parametric)
    u_stat, p_value = stats.mannwhitneyu(dbd_approached, dbd_not_approached, alternative='two-sided')
    print(f"\nMann-Whitney U test: U={u_stat:.0f}, p={p_value:.4f}")
    
    if p_value < 0.05:
        if dbd_approached.median() > dbd_not_approached.median():
            print("✓ DBD approached cases have LONGER windows")
        else:
            print("✓ DBD approached cases have SHORTER windows")
    else:
        print("○ No significant difference in DBD windows")

# DCD Analysis
print("\n--- DCD Pathway ---")
dcd_mscs = mscs[mscs['pathway'] == 'DCD'].copy()
dcd_approached = dcd_mscs[dcd_mscs['approached'] == True]['window_to_asystole'].dropna()
dcd_not_approached = dcd_mscs[dcd_mscs['approached'] == False]['window_to_asystole'].dropna()

if len(dcd_approached) > 0 and len(dcd_not_approached) > 0:
    print(f"\nApproached (n={len(dcd_approached):,}):")
    print(f"  Median window: {dcd_approached.median():.1f} hours")
    print(f"  IQR: [{dcd_approached.quantile(0.25):.1f}, {dcd_approached.quantile(0.75):.1f}]")
    
    print(f"\nNot Approached (n={len(dcd_not_approached):,}):")
    print(f"  Median window: {dcd_not_approached.median():.1f} hours")
    print(f"  IQR: [{dcd_not_approached.quantile(0.25):.1f}, {dcd_not_approached.quantile(0.75):.1f}]")
    
    u_stat, p_value = stats.mannwhitneyu(dcd_approached, dcd_not_approached, alternative='two-sided')
    print(f"\nMann-Whitney U test: U={u_stat:.0f}, p={p_value:.4f}")
    
    if p_value < 0.05:
        if dcd_approached.median() > dcd_not_approached.median():
            print("✓ DCD approached cases have LONGER windows")
            print("  INTERPRETATION: Timing IS a bottleneck - longer windows enable approach")
        else:
            print("✓ DCD approached cases have SHORTER windows")
            print("  INTERPRETATION: Timing is NOT the bottleneck - something else matters")
    else:
        print("○ No significant difference in DCD windows")
        print("  INTERPRETATION: Timing is NOT the bottleneck")

# ============================================================================
# PHASE 4: Window Length vs Approach Probability
# ============================================================================

print("\n" + "="*80)
print("PHASE 4: Approach Probability by Window Length")
print("="*80)

# Bin DCD windows and calculate approach rates
dcd_with_windows = dcd_mscs[dcd_mscs['window_to_asystole'].notna()].copy()

if len(dcd_with_windows) > 0:
    # Create bins
    dcd_with_windows['window_bin'] = pd.cut(
        dcd_with_windows['window_to_asystole'],
        bins=[0, 12, 24, 48, 72, 1000],
        labels=['0-12hr', '12-24hr', '24-48hr', '48-72hr', '>72hr']
    )
    
    window_approach_rates = dcd_with_windows.groupby('window_bin').agg({
        'patient_id': 'count',
        'approached': ['sum', 'mean']
    }).round(3)
    
    window_approach_rates.columns = ['n_referrals', 'n_approached', 'approach_rate']
    
    print("\nDCD Approach Rates by Window Length:")
    print(window_approach_rates)
    
    # Test for trend
    from scipy.stats import spearmanr
    
    # Assign numeric values to bins
    bin_to_numeric = {'0-12hr': 6, '12-24hr': 18, '24-48hr': 36, '48-72hr': 60, '>72hr': 96}
    dcd_with_windows['window_numeric'] = dcd_with_windows['window_bin'].map(bin_to_numeric)
    
    valid_data = dcd_with_windows[['window_numeric', 'approached']].dropna()
    if len(valid_data) > 0:
        corr, p_value = spearmanr(valid_data['window_numeric'], valid_data['approached'])
        print(f"\nSpearman correlation (window length vs approach): ρ={corr:.3f}, p={p_value:.4f}")
        
        if p_value < 0.05:
            if corr > 0:
                print("✓ POSITIVE correlation: Longer windows → Higher approach rate")
                print("  INTERPRETATION: Timing IS the bottleneck")
            else:
                print("✓ NEGATIVE correlation: Longer windows → Lower approach rate")
                print("  INTERPRETATION: Timing is NOT the bottleneck (paradoxical)")
        else:
            print("○ No significant correlation")
            print("  INTERPRETATION: Window length doesn't predict approach likelihood")

# ============================================================================
# PHASE 5: Time-to-Approach (How Fast Do OPOs Act?)
# ============================================================================

print("\n" + "="*80)
print("PHASE 5: Time-to-Approach Analysis")
print("="*80)

# For approached cases, how long from referral to approach?
print("\nDBD Time-to-Approach:")
dbd_approach_times = dbd_mscs[dbd_mscs['approached'] == True]['window_to_approach'].dropna()
if len(dbd_approach_times) > 0:
    print(f"  n = {len(dbd_approach_times):,}")
    print(f"  Median: {dbd_approach_times.median():.1f} hours")
    print(f"  IQR: [{dbd_approach_times.quantile(0.25):.1f}, {dbd_approach_times.quantile(0.75):.1f}]")

print("\nDCD Time-to-Approach:")
dcd_approach_times = dcd_mscs[dcd_mscs['approached'] == True]['window_to_approach'].dropna()
if len(dcd_approach_times) > 0:
    print(f"  n = {len(dcd_approach_times):,}")
    print(f"  Median: {dcd_approach_times.median():.1f} hours")
    print(f"  IQR: [{dcd_approach_times.quantile(0.25):.1f}, {dcd_approach_times.quantile(0.75):.1f}]")

if len(dbd_approach_times) > 0 and len(dcd_approach_times) > 0:
    u_stat, p_value = stats.mannwhitneyu(dbd_approach_times, dcd_approach_times, alternative='two-sided')
    print(f"\nDBD vs DCD time-to-approach comparison:")
    print(f"  Difference: {dbd_approach_times.median() - dcd_approach_times.median():.1f} hours")
    print(f"  Mann-Whitney U: U={u_stat:.0f}, p={p_value:.4f}")
    
    if p_value < 0.05:
        if dcd_approach_times.median() < dbd_approach_times.median():
            print("✓ DCD is approached FASTER than DBD")
            print("  INTERPRETATION: OPOs recognize DCD urgency and act quickly")
        else:
            print("✓ DBD is approached FASTER than DCD")
            print("  INTERPRETATION: Unexpected - DBD is prioritized?")
    else:
        print("○ No significant difference in approach timing")

# ============================================================================
# PHASE 6: Timing Buffer Analysis
# ============================================================================

print("\n" + "="*80)
print("PHASE 6: Timing Buffer Analysis")
print("="*80)
print("\nFor approached cases: How much time BEFORE outcome did OPO approach?")

# Calculate buffer: window_to_outcome - window_to_approach
dbd_approached_with_both = dbd_mscs[
    (dbd_mscs['approached'] == True) &
    (dbd_mscs['window_to_brain_death'].notna()) &
    (dbd_mscs['window_to_approach'].notna())
].copy()

dcd_approached_with_both = dcd_mscs[
    (dcd_mscs['approached'] == True) &
    (dcd_mscs['window_to_asystole'].notna()) &
    (dcd_mscs['window_to_approach'].notna())
].copy()

if len(dbd_approached_with_both) > 0:
    dbd_approached_with_both['buffer'] = (
        dbd_approached_with_both['window_to_brain_death'] - 
        dbd_approached_with_both['window_to_approach']
    )
    
    print("\nDBD Buffer (hours before brain death that family was approached):")
    print(f"  n = {len(dbd_approached_with_both):,}")
    print(f"  Median: {dbd_approached_with_both['buffer'].median():.1f} hours")
    print(f"  IQR: [{dbd_approached_with_both['buffer'].quantile(0.25):.1f}, {dbd_approached_with_both['buffer'].quantile(0.75):.1f}]")
    
    negative_buffer = (dbd_approached_with_both['buffer'] < 0).sum()
    print(f"  Approached AFTER brain death: {negative_buffer} ({negative_buffer/len(dbd_approached_with_both):.1%})")

if len(dcd_approached_with_both) > 0:
    dcd_approached_with_both['buffer'] = (
        dcd_approached_with_both['window_to_asystole'] - 
        dcd_approached_with_both['window_to_approach']
    )
    
    print("\nDCD Buffer (hours before asystole that family was approached):")
    print(f"  n = {len(dcd_approached_with_both):,}")
    print(f"  Median: {dcd_approached_with_both['buffer'].median():.1f} hours")
    print(f"  IQR: [{dcd_approached_with_both['buffer'].quantile(0.25):.1f}, {dcd_approached_with_both['buffer'].quantile(0.75):.1f}]")
    
    negative_buffer = (dcd_approached_with_both['buffer'] < 0).sum()
    print(f"  Approached AFTER asystole: {negative_buffer} ({negative_buffer/len(dcd_approached_with_both):.1%})")

if len(dbd_approached_with_both) > 0 and len(dcd_approached_with_both) > 0:
    u_stat, p_value = stats.mannwhitneyu(
        dbd_approached_with_both['buffer'], 
        dcd_approached_with_both['buffer'], 
        alternative='two-sided'
    )
    
    print(f"\nDBD vs DCD buffer comparison:")
    print(f"  Difference: {dbd_approached_with_both['buffer'].median() - dcd_approached_with_both['buffer'].median():.1f} hours")
    print(f"  Mann-Whitney U: U={u_stat:.0f}, p={p_value:.4f}")
    
    if p_value < 0.05:
        if dcd_approached_with_both['buffer'].median() < dbd_approached_with_both['buffer'].median():
            print("✓ DCD has SHORTER buffer than DBD")
            print("  INTERPRETATION: DCD is approached closer to the deadline (rushed)")
        else:
            print("✓ DBD has SHORTER buffer than DCD")
            print("  INTERPRETATION: Unexpected pattern")
    else:
        print("○ No significant difference in buffers")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("Generating visualizations...")
print("="*80)

# Figure 1: Window distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# DBD windows
ax = axes[0, 0]
if len(dbd_approached) > 0 and len(dbd_not_approached) > 0:
    ax.hist([dbd_approached, dbd_not_approached], bins=30, alpha=0.7, 
           label=['Approached', 'Not Approached'], color=['green', 'red'])
    ax.axvline(dbd_approached.median(), color='green', linestyle='--', linewidth=2, 
              label=f'Approached median: {dbd_approached.median():.1f}hr')
    ax.axvline(dbd_not_approached.median(), color='red', linestyle='--', linewidth=2,
              label=f'Not approached median: {dbd_not_approached.median():.1f}hr')
    ax.set_xlabel('Hours from Referral to Brain Death', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('DBD: Window Length Distribution', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)

# DCD windows
ax = axes[0, 1]
if len(dcd_approached) > 0 and len(dcd_not_approached) > 0:
    ax.hist([dcd_approached, dcd_not_approached], bins=30, alpha=0.7,
           label=['Approached', 'Not Approached'], color=['green', 'red'])
    ax.axvline(dcd_approached.median(), color='green', linestyle='--', linewidth=2,
              label=f'Approached median: {dcd_approached.median():.1f}hr')
    ax.axvline(dcd_not_approached.median(), color='red', linestyle='--', linewidth=2,
              label=f'Not approached median: {dcd_not_approached.median():.1f}hr')
    ax.set_xlabel('Hours from Referral to Asystole', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('DCD: Window Length Distribution', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)

# Approach rates by window length
ax = axes[1, 0]
if 'window_approach_rates' in locals() and len(window_approach_rates) > 0:
    bars = ax.bar(range(len(window_approach_rates)), window_approach_rates['approach_rate'],
                 color=plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(window_approach_rates))),
                 alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Window Length', fontweight='bold')
    ax.set_ylabel('DCD Approach Rate', fontweight='bold')
    ax.set_title('DCD Approach Rate by Window Length', fontweight='bold', fontsize=12)
    ax.set_xticks(range(len(window_approach_rates)))
    ax.set_xticklabels(window_approach_rates.index, rotation=15, ha='right')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(axis='y', alpha=0.3)
    
    for i, (idx, row) in enumerate(window_approach_rates.iterrows()):
        ax.text(i, row['approach_rate'] + 0.01, f"n={row['n_referrals']:.0f}",
               ha='center', fontsize=9, color='gray')

# Buffer comparison
ax = axes[1, 1]
if len(dbd_approached_with_both) > 0 and len(dcd_approached_with_both) > 0:
    data_to_plot = [dbd_approached_with_both['buffer'], dcd_approached_with_both['buffer']]
    bp = ax.boxplot(data_to_plot, labels=['DBD', 'DCD'], patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('coral')
    ax.set_ylabel('Hours Before Outcome', fontweight='bold')
    ax.set_title('Timing Buffer: Approach Before Outcome', fontweight='bold', fontsize=12)
    ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Outcome time')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'time_window_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {FIGURES_DIR / 'time_window_analysis.png'}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("TIME WINDOW ANALYSIS: SUMMARY")
print("="*80)

print("\nCore Question: Is timing the DCD bottleneck?")

print("\nKey Findings:")

if len(dcd_approached) > 0 and len(dcd_not_approached) > 0:
    print(f"\n1. DCD Window Comparison:")
    print(f"   - Approached: {dcd_approached.median():.1f} hours (median)")
    print(f"   - Not approached: {dcd_not_approached.median():.1f} hours (median)")
    print(f"   - Difference: {dcd_approached.median() - dcd_not_approached.median():+.1f} hours")
    
    if dcd_approached.median() > dcd_not_approached.median():
        print("   → Approached cases have LONGER windows")
        print("   ✓ TIMING IS A BOTTLENECK")
    else:
        print("   → Approached cases have SHORTER windows")
        print("   ○ TIMING IS NOT THE BOTTLENECK")

if len(dbd_windows) > 0 and len(dcd_windows) > 0:
    print(f"\n2. DBD vs DCD Window Comparison:")
    print(f"   - DBD: {dbd_windows.median():.1f} hours (median)")
    print(f"   - DCD: {dcd_windows.median():.1f} hours (median)")
    print(f"   - Difference: {dbd_windows.median() - dcd_windows.median():+.1f} hours")

if len(dcd_approached_with_both) > 0:
    print(f"\n3. DCD Timing Buffer:")
    print(f"   - Median buffer: {dcd_approached_with_both['buffer'].median():.1f} hours")
    print(f"   - Families approached {dcd_approached_with_both['buffer'].median():.1f} hours before asystole")

print("\n" + "="*80)
print("CONCLUSION:")

# Determine conclusion based on findings
if len(dcd_approached) > 0 and len(dcd_not_approached) > 0:
    if dcd_approached.median() > dcd_not_approached.median() * 1.2:  # >20% longer
        print("✓ TIMING IS THE PRIMARY BOTTLENECK FOR DCD")
        print("  DCD donors with longer windows are significantly more likely to be approached.")
        print("  OPOs need more time to cultivate DCD donors proactively.")
    elif dcd_approached.median() < dcd_not_approached.median() * 0.8:  # >20% shorter
        print("○ TIMING IS NOT THE BOTTLENECK")
        print("  Approached DCD cases actually have SHORTER windows.")
        print("  Something other than timing drives sorting decisions.")
    else:
        print("◐ TIMING MAY BE A FACTOR, BUT NOT THE PRIMARY BOTTLENECK")
        print("  Window lengths are similar for approached vs. not-approached cases.")
        print("  Other factors (capacity, training, culture) likely matter more.")
else:
    print("⚠ INSUFFICIENT DATA")
    print("  Cannot determine if timing is the bottleneck.")

print("="*80)
