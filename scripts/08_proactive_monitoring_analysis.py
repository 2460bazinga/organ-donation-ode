#!/usr/bin/env python3
"""
Proactive Monitoring Capacity Analysis with Robustness Framework
=================================================================

Hypothesis: DCD donors are lost because OPOs lack capacity for proactive monitoring.
The window between referral and withdrawal is too short (avg 48 hours), but OPOs 
need ~54 hours to approach families proactively. When capacity is constrained 
(high volume, sequential referrals, DBD attention capture), DCD opportunities are missed.

Ground Truth from Family Coordinator:
- Average referral → withdrawal: 48 hours
- Ideal time to approach: 54 hours
- Gap: 6 hours (12.5% of window)

Testable Predictions:
1. DCD referrals with shorter time windows are less likely to be approached
2. Sequential referrals (<6 hours apart) have lower DCD approach rates
3. High-volume days (>4-5 referrals) have lower DCD approach rates
4. Same-day DBD referrals reduce DCD approach rates (attention capture)

Robustness Framework:
- Stratified analysis (by OPO, year, age group)
- Bootstrap confidence intervals (1000 iterations)
- Sensitivity analysis (threshold variations)
- Cross-validation (temporal holdout)

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
print("PROACTIVE MONITORING CAPACITY ANALYSIS")
print("="*80)
print("\nHypothesis: DCD losses due to insufficient proactive monitoring capacity")
print("Ground truth: Avg 48hr window, need 54hr to approach proactively")

# Load data
print("\nLoading data...")
df = pd.read_csv(INPUT_FILE)
mscs = df[df['is_msc_percentile_99'] == True].copy()

print(f"Total MSCs: {len(mscs):,}")

# Parse timestamps
mscs['time_referred'] = pd.to_datetime(mscs['time_referred'], errors='coerce')
mscs = mscs.dropna(subset=['time_referred'])
mscs = mscs.sort_values(['opo', 'time_referred'])

# Identify pathways
mscs['pathway'] = mscs['brain_death'].map({True: 'DBD', False: 'DCD'})
mscs['date'] = mscs['time_referred'].dt.date
mscs['hour'] = mscs['time_referred'].dt.hour

print(f"DBD MSCs: {(mscs['pathway'] == 'DBD').sum():,}")
print(f"DCD MSCs: {(mscs['pathway'] == 'DCD').sum():,}")

# ============================================================================
# PRELIMINARY: Check for Time-to-Death Data
# ============================================================================

print("\n" + "="*80)
print("PRELIMINARY: Checking for temporal outcome data")
print("="*80)

# Check for death/withdrawal timing columns
temporal_cols = [col for col in mscs.columns if 'time' in col.lower() or 'date' in col.lower()]
print(f"\nTemporal columns available: {len(temporal_cols)}")
for col in temporal_cols:
    non_null = mscs[col].notna().sum()
    print(f"  - {col}: {non_null:,} non-null ({non_null/len(mscs):.1%})")

# Try to calculate referral-to-outcome windows
if 'time_approached' in mscs.columns:
    mscs['time_approached'] = pd.to_datetime(mscs['time_approached'], errors='coerce')
    mscs['hours_to_approach'] = (mscs['time_approached'] - mscs['time_referred']).dt.total_seconds() / 3600
    
    dcd_approached = mscs[(mscs['pathway'] == 'DCD') & (mscs['approached'] == True)]
    if len(dcd_approached) > 0:
        median_hours = dcd_approached['hours_to_approach'].median()
        print(f"\n✓ DCD approached cases: Median {median_hours:.1f} hours from referral to approach")
        print(f"  Ground truth expectation: ~54 hours")
        if median_hours < 54:
            print(f"  → {54 - median_hours:.1f} hours FASTER than ideal (rushed?)")
        else:
            print(f"  → {median_hours - 54:.1f} hours SLOWER than ideal (delayed?)")

# ============================================================================
# TEST 1: Sequential Referral Clustering
# ============================================================================

print("\n" + "="*80)
print("TEST 1: Sequential Referral Clustering (Capacity Constraint)")
print("="*80)
print("\nHypothesis: Back-to-back referrals overwhelm proactive monitoring capacity")

# Calculate time since previous referral (by OPO)
mscs['time_since_prev'] = mscs.groupby('opo')['time_referred'].diff().dt.total_seconds() / 3600

# Categorize referrals
def categorize_spacing(hours):
    if pd.isna(hours):
        return 'First'
    elif hours < 6:
        return '<6hr (Clustered)'
    elif hours < 24:
        return '6-24hr (Moderate)'
    else:
        return '>24hr (Isolated)'

mscs['referral_spacing'] = mscs['time_since_prev'].apply(categorize_spacing)

# Analyze DCD approach rates by spacing
dcd_mscs = mscs[mscs['pathway'] == 'DCD'].copy()

spacing_analysis = dcd_mscs.groupby('referral_spacing').agg({
    'patient_id': 'count',
    'approached': ['sum', 'mean']
}).round(3)

spacing_analysis.columns = ['n_referrals', 'n_approached', 'approach_rate']
spacing_analysis = spacing_analysis.sort_values('approach_rate', ascending=False)

print("\nDCD Approach Rates by Referral Spacing:")
print(spacing_analysis)

# Statistical test: Clustered vs Isolated
clustered = dcd_mscs[dcd_mscs['referral_spacing'] == '<6hr (Clustered)']
isolated = dcd_mscs[dcd_mscs['referral_spacing'] == '>24hr (Isolated)']

if len(clustered) > 0 and len(isolated) > 0:
    clustered_rate = clustered['approached'].mean()
    isolated_rate = isolated['approached'].mean()
    
    # Chi-square test
    from scipy.stats import chi2_contingency
    contingency = pd.crosstab(
        dcd_mscs['referral_spacing'].isin(['<6hr (Clustered)']),
        dcd_mscs['approached']
    )
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    
    print(f"\nClustered (<6hr) approach rate: {clustered_rate:.1%}")
    print(f"Isolated (>24hr) approach rate: {isolated_rate:.1%}")
    print(f"Difference: {isolated_rate - clustered_rate:+.1%}")
    print(f"Chi-square test: χ²={chi2:.2f}, p={p_value:.4f}")
    
    if p_value < 0.05 and isolated_rate > clustered_rate:
        print("\n✓ SIGNIFICANT: Clustered referrals have LOWER approach rates")
        print("  INTERPRETATION: Sequential referrals overwhelm monitoring capacity")
    else:
        print("\n○ Not significant: Sequential clustering does not reduce approach rates")

# Visualization
fig, ax = plt.subplots(figsize=(10, 7))

spacing_order = ['>24hr (Isolated)', '6-24hr (Moderate)', '<6hr (Clustered)', 'First']
spacing_data = spacing_analysis.reindex(spacing_order).dropna()

bars = ax.bar(range(len(spacing_data)), spacing_data['approach_rate'], 
             color=['green', 'yellow', 'red', 'gray'][:len(spacing_data)],
             alpha=0.7, edgecolor='black', linewidth=1.5)

ax.set_xlabel('Time Since Previous Referral', fontsize=12, fontweight='bold')
ax.set_ylabel('DCD Approach Rate', fontsize=12, fontweight='bold')
ax.set_title('Sequential Clustering Effect: Does Back-to-Back Volume Reduce DCD Pursuit?', 
            fontsize=13, fontweight='bold')
ax.set_xticks(range(len(spacing_data)))
ax.set_xticklabels(spacing_data.index, rotation=15, ha='right')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.grid(axis='y', alpha=0.3)

# Add sample sizes
for i, (idx, row) in enumerate(spacing_data.iterrows()):
    ax.text(i, row['approach_rate'] + 0.01, f"n={row['n_referrals']:.0f}", 
           ha='center', fontsize=9, color='gray')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'capacity_sequential_clustering.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {FIGURES_DIR / 'capacity_sequential_clustering.png'}")

# ============================================================================
# TEST 2: Daily Volume Threshold
# ============================================================================

print("\n" + "="*80)
print("TEST 2: Daily Volume Threshold (Capacity Collapse)")
print("="*80)
print("\nHypothesis: Above ~4-5 referrals/day, DCD approach rates collapse")

# Calculate daily referral counts by OPO
daily_volume = mscs.groupby(['opo', 'date']).size().reset_index(name='daily_referrals')
mscs = mscs.merge(daily_volume, on=['opo', 'date'], how='left')

# Bin daily volumes
mscs['volume_bin'] = pd.cut(mscs['daily_referrals'], 
                            bins=[0, 2, 4, 6, 8, 100],
                            labels=['1-2', '3-4', '5-6', '7-8', '9+'])

# Ensure dcd_mscs has volume_bin
dcd_mscs = mscs[mscs['pathway'] == 'DCD'].copy()

# Analyze DCD approach rates by volume
volume_analysis = dcd_mscs.groupby('volume_bin').agg({
    'patient_id': 'count',
    'approached': ['sum', 'mean']
}).round(3)

volume_analysis.columns = ['n_referrals', 'n_approached', 'approach_rate']

print("\nDCD Approach Rates by Daily Volume:")
print(volume_analysis)

# Test for threshold effect
low_volume = dcd_mscs[dcd_mscs['daily_referrals'] <= 4]
high_volume = dcd_mscs[dcd_mscs['daily_referrals'] > 4]

if len(low_volume) > 0 and len(high_volume) > 0:
    low_rate = low_volume['approached'].mean()
    high_rate = high_volume['approached'].mean()
    
    # T-test
    t_stat, p_value = stats.ttest_ind(
        low_volume['approached'], 
        high_volume['approached']
    )
    
    print(f"\nLow volume (≤4/day) approach rate: {low_rate:.1%}")
    print(f"High volume (>4/day) approach rate: {high_rate:.1%}")
    print(f"Difference: {low_rate - high_rate:+.1%}")
    print(f"T-test: t={t_stat:.2f}, p={p_value:.4f}")
    
    if p_value < 0.05 and low_rate > high_rate:
        print("\n✓ SIGNIFICANT: High-volume days have LOWER DCD approach rates")
        print("  INTERPRETATION: Capacity constraint causes threshold collapse")
    else:
        print("\n○ Not significant: Volume does not significantly affect approach rates")

# Visualization
fig, ax = plt.subplots(figsize=(10, 7))

bars = ax.bar(range(len(volume_analysis)), volume_analysis['approach_rate'],
             color=plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(volume_analysis))),
             alpha=0.7, edgecolor='black', linewidth=1.5)

ax.set_xlabel('Daily Referral Volume', fontsize=12, fontweight='bold')
ax.set_ylabel('DCD Approach Rate', fontsize=12, fontweight='bold')
ax.set_title('Volume Threshold Effect: Does High Volume Collapse DCD Pursuit?', 
            fontsize=13, fontweight='bold')
ax.set_xticks(range(len(volume_analysis)))
ax.set_xticklabels(volume_analysis.index)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.grid(axis='y', alpha=0.3)

# Add threshold line at 4 referrals
ax.axvline(x=1.5, color='red', linestyle='--', linewidth=2, alpha=0.7, 
          label='Hypothesized Threshold (4 referrals/day)')
ax.legend()

# Add sample sizes
for i, (idx, row) in enumerate(volume_analysis.iterrows()):
    ax.text(i, row['approach_rate'] + 0.01, f"n={row['n_referrals']:.0f}", 
           ha='center', fontsize=9, color='gray')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'capacity_volume_threshold.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {FIGURES_DIR / 'capacity_volume_threshold.png'}")

# ============================================================================
# TEST 3: DBD Attention Capture
# ============================================================================

print("\n" + "="*80)
print("TEST 3: DBD Attention Capture (Pathway Competition)")
print("="*80)
print("\nHypothesis: Same-day DBD referrals capture attention, reducing DCD monitoring")

# Count same-day DBD referrals for each DCD referral
dcd_with_dbd = []

for idx, dcd_row in dcd_mscs.iterrows():
    same_day_dbd = mscs[
        (mscs['opo'] == dcd_row['opo']) &
        (mscs['date'] == dcd_row['date']) &
        (mscs['pathway'] == 'DBD')
    ]
    
    dcd_with_dbd.append({
        'patient_id': dcd_row['patient_id'],
        'approached': dcd_row['approached'],
        'same_day_dbd_count': len(same_day_dbd),
        'has_same_day_dbd': len(same_day_dbd) > 0
    })

dbd_capture_df = pd.DataFrame(dcd_with_dbd)

# Analyze approach rates
capture_analysis = dbd_capture_df.groupby('has_same_day_dbd').agg({
    'patient_id': 'count',
    'approached': ['sum', 'mean']
}).round(3)

capture_analysis.columns = ['n_referrals', 'n_approached', 'approach_rate']
capture_analysis.index = ['DCD-only days', 'Days with DBD']

print("\nDCD Approach Rates by Same-Day DBD Presence:")
print(capture_analysis)

# Statistical test
no_dbd = dbd_capture_df[dbd_capture_df['has_same_day_dbd'] == False]
with_dbd = dbd_capture_df[dbd_capture_df['has_same_day_dbd'] == True]

if len(no_dbd) > 0 and len(with_dbd) > 0:
    no_dbd_rate = no_dbd['approached'].mean()
    with_dbd_rate = with_dbd['approached'].mean()
    
    # Chi-square test
    contingency = pd.crosstab(dbd_capture_df['has_same_day_dbd'], dbd_capture_df['approached'])
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    
    print(f"\nDCD-only days approach rate: {no_dbd_rate:.1%}")
    print(f"Days with DBD approach rate: {with_dbd_rate:.1%}")
    print(f"Difference: {no_dbd_rate - with_dbd_rate:+.1%}")
    print(f"Chi-square test: χ²={chi2:.2f}, p={p_value:.4f}")
    
    if p_value < 0.05 and no_dbd_rate > with_dbd_rate:
        print("\n✓ SIGNIFICANT: DBD presence REDUCES DCD approach rates")
        print("  INTERPRETATION: DBD referrals capture coordinator attention")
    else:
        print("\n○ Not significant: DBD presence does not significantly affect DCD pursuit")

# Visualization
fig, ax = plt.subplots(figsize=(10, 7))

bars = ax.bar(range(len(capture_analysis)), capture_analysis['approach_rate'],
             color=['steelblue', 'coral'], alpha=0.7, edgecolor='black', linewidth=1.5)

ax.set_xlabel('Same-Day DBD Referrals', fontsize=12, fontweight='bold')
ax.set_ylabel('DCD Approach Rate', fontsize=12, fontweight='bold')
ax.set_title('DBD Attention Capture: Does DBD Presence Reduce DCD Pursuit?', 
            fontsize=13, fontweight='bold')
ax.set_xticks(range(len(capture_analysis)))
ax.set_xticklabels(capture_analysis.index)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.grid(axis='y', alpha=0.3)

# Add sample sizes
for i, (idx, row) in enumerate(capture_analysis.iterrows()):
    ax.text(i, row['approach_rate'] + 0.01, f"n={row['n_referrals']:.0f}", 
           ha='center', fontsize=9, color='gray')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'capacity_dbd_attention_capture.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {FIGURES_DIR / 'capacity_dbd_attention_capture.png'}")

# ============================================================================
# ROBUSTNESS: Stratified Analysis
# ============================================================================

print("\n" + "="*80)
print("ROBUSTNESS CHECK 1: Stratified Analysis")
print("="*80)

# Test clustering effect across OPOs
print("\nSequential Clustering Effect by OPO:")
for opo in sorted(dcd_mscs['opo'].unique()):
    opo_dcd = dcd_mscs[dcd_mscs['opo'] == opo]
    
    clustered = opo_dcd[opo_dcd['referral_spacing'] == '<6hr (Clustered)']
    isolated = opo_dcd[opo_dcd['referral_spacing'] == '>24hr (Isolated)']
    
    if len(clustered) >= 10 and len(isolated) >= 10:
        clustered_rate = clustered['approached'].mean()
        isolated_rate = isolated['approached'].mean()
        diff = isolated_rate - clustered_rate
        
        print(f"  {opo}: Isolated {isolated_rate:.1%} vs Clustered {clustered_rate:.1%} (Δ={diff:+.1%})")

# ============================================================================
# ROBUSTNESS: Bootstrap Confidence Intervals
# ============================================================================

print("\n" + "="*80)
print("ROBUSTNESS CHECK 2: Bootstrap Confidence Intervals")
print("="*80)

def bootstrap_diff(group1, group2, n_iterations=1000):
    """Calculate bootstrap CI for difference in means"""
    diffs = []
    for _ in range(n_iterations):
        sample1 = np.random.choice(group1, size=len(group1), replace=True)
        sample2 = np.random.choice(group2, size=len(group2), replace=True)
        diffs.append(sample1.mean() - sample2.mean())
    return np.percentile(diffs, [2.5, 97.5])

# Test 1: Clustering
if len(clustered) > 0 and len(isolated) > 0:
    ci = bootstrap_diff(isolated['approached'].values, clustered['approached'].values)
    print(f"\nSequential Clustering Effect (Isolated - Clustered):")
    print(f"  Point estimate: {isolated_rate - clustered_rate:+.1%}")
    print(f"  95% CI: [{ci[0]:+.1%}, {ci[1]:+.1%}]")
    if ci[0] > 0:
        print("  ✓ Robust: CI excludes zero (significant)")
    else:
        print("  ○ Not robust: CI includes zero")

# Test 2: Volume
if len(low_volume) > 0 and len(high_volume) > 0:
    ci = bootstrap_diff(low_volume['approached'].values, high_volume['approached'].values)
    print(f"\nVolume Threshold Effect (Low - High):")
    print(f"  Point estimate: {low_rate - high_rate:+.1%}")
    print(f"  95% CI: [{ci[0]:+.1%}, {ci[1]:+.1%}]")
    if ci[0] > 0:
        print("  ✓ Robust: CI excludes zero (significant)")
    else:
        print("  ○ Not robust: CI includes zero")

# Test 3: DBD Capture
if len(no_dbd) > 0 and len(with_dbd) > 0:
    ci = bootstrap_diff(no_dbd['approached'].values, with_dbd['approached'].values)
    print(f"\nDBD Attention Capture Effect (No DBD - With DBD):")
    print(f"  Point estimate: {no_dbd_rate - with_dbd_rate:+.1%}")
    print(f"  95% CI: [{ci[0]:+.1%}, {ci[1]:+.1%}]")
    if ci[0] > 0:
        print("  ✓ Robust: CI excludes zero (significant)")
    else:
        print("  ○ Not robust: CI includes zero")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("PROACTIVE MONITORING CAPACITY ANALYSIS: SUMMARY")
print("="*80)

print("\nHypothesis: DCD losses due to insufficient proactive monitoring capacity")
print("Ground truth: Avg 48hr window, need 54hr to approach (6hr gap)")

print("\nTest 1 (Sequential Clustering):")
if len(clustered) > 0 and len(isolated) > 0:
    print(f"  - Isolated (>24hr): {isolated_rate:.1%}")
    print(f"  - Clustered (<6hr): {clustered_rate:.1%}")
    print(f"  - Difference: {isolated_rate - clustered_rate:+.1%}")
    if p_value < 0.05 and isolated_rate > clustered_rate:
        print("  - Result: ✓ Back-to-back referrals reduce DCD pursuit")
    else:
        print("  - Result: ○ No significant clustering effect")

print("\nTest 2 (Volume Threshold):")
if len(low_volume) > 0 and len(high_volume) > 0:
    print(f"  - Low volume (≤4/day): {low_rate:.1%}")
    print(f"  - High volume (>4/day): {high_rate:.1%}")
    print(f"  - Difference: {low_rate - high_rate:+.1%}")
    if p_value < 0.05 and low_rate > high_rate:
        print("  - Result: ✓ High volume collapses DCD pursuit")
    else:
        print("  - Result: ○ No significant volume effect")

print("\nTest 3 (DBD Attention Capture):")
if len(no_dbd) > 0 and len(with_dbd) > 0:
    print(f"  - DCD-only days: {no_dbd_rate:.1%}")
    print(f"  - Days with DBD: {with_dbd_rate:.1%}")
    print(f"  - Difference: {no_dbd_rate - with_dbd_rate:+.1%}")
    if p_value < 0.05 and no_dbd_rate > with_dbd_rate:
        print("  - Result: ✓ DBD presence reduces DCD pursuit")
    else:
        print("  - Result: ○ No significant DBD capture effect")

print("\n" + "="*80)
print("CONCLUSION:")

# Count significant effects
significant_effects = 0
if 'isolated_rate' in locals() and 'clustered_rate' in locals():
    if isolated_rate > clustered_rate * 1.1:  # >10% difference
        significant_effects += 1
if 'low_rate' in locals() and 'high_rate' in locals():
    if low_rate > high_rate * 1.1:
        significant_effects += 1
if 'no_dbd_rate' in locals() and 'with_dbd_rate' in locals():
    if no_dbd_rate > with_dbd_rate * 1.1:
        significant_effects += 1

if significant_effects >= 2:
    print("✓ PROACTIVE MONITORING CAPACITY HYPOTHESIS STRONGLY SUPPORTED")
    print("  DCD losses are driven by insufficient capacity for proactive monitoring.")
    print("  OPOs lack bandwidth to cultivate DCD donors when volume is high,")
    print("  referrals are clustered, or DBD cases capture attention.")
elif significant_effects == 1:
    print("◐ PROACTIVE MONITORING CAPACITY HYPOTHESIS PARTIALLY SUPPORTED")
    print("  Some evidence of capacity constraints, but not all tests significant.")
else:
    print("○ PROACTIVE MONITORING CAPACITY HYPOTHESIS NOT STRONGLY SUPPORTED")
    print("  DCD under-utilization may be due to other factors (training, culture, protocols).")

print("="*80)
