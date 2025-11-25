#!/usr/bin/env python3
"""
Pathway Competition Analysis: Testing the DBD-DCD Trade-off Hypothesis
=======================================================================

Hypothesis: OPOs cannot pursue DCD donors because they're waiting to see if
DBD donors materialize. This creates a sequential dependency where DCD opportunities
are lost due to pathway switching costs.

Testable Predictions:
1. Lower DCD approach rates in OPOs with higher DBD volume (resource competition)
2. Higher DCD approach rates on weekends (lower DBD volume, more capacity)
3. Temporal clustering of missed DCD cases around DBD referrals
4. DCD losses concentrated in time periods with high DBD activity

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

# Paths
DATA_DIR = Path.home() / 'physionet.org' / 'files' / 'orchid' / '2.1.1'
INPUT_FILE = DATA_DIR / 'orchid_with_msc_sensitivity.csv'
OUTPUT_DIR = Path.home() / 'results'
FIGURES_DIR = OUTPUT_DIR / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "="*80)
print("PATHWAY COMPETITION ANALYSIS: DBD-DCD Trade-off Hypothesis")
print("="*80)

# Load data
print("\nLoading data...")
df = pd.read_csv(INPUT_FILE)
mscs = df[df['is_msc_percentile_99'] == True].copy()

print(f"Total MSCs: {len(mscs):,}")

# Parse timestamps
mscs['time_referred'] = pd.to_datetime(mscs['time_referred'], errors='coerce')
mscs = mscs.dropna(subset=['time_referred'])
mscs['date'] = mscs['time_referred'].dt.date
mscs['hour'] = mscs['time_referred'].dt.hour
mscs['day_of_week'] = mscs['time_referred'].dt.dayofweek
mscs['is_weekend'] = mscs['day_of_week'].isin([5, 6])

# Identify pathways
mscs['pathway'] = mscs['brain_death'].map({True: 'DBD', False: 'DCD'})

print(f"DBD MSCs: {(mscs['pathway'] == 'DBD').sum():,} ({(mscs['pathway'] == 'DBD').mean():.1%})")
print(f"DCD MSCs: {(mscs['pathway'] == 'DCD').sum():,} ({(mscs['pathway'] == 'DCD').mean():.1%})")

# ============================================================================
# TEST 1: Resource Competition (OPO-Level)
# ============================================================================

print("\n" + "="*80)
print("TEST 1: Do OPOs with higher DBD volume have lower DCD approach rates?")
print("="*80)

opo_pathway_stats = []

for opo in sorted(mscs['opo'].unique()):
    opo_df = mscs[mscs['opo'] == opo]
    
    # DBD metrics
    dbd_df = opo_df[opo_df['pathway'] == 'DBD']
    dbd_volume = len(dbd_df)
    dbd_approach_rate = dbd_df['approached'].mean() if len(dbd_df) > 0 else 0
    
    # DCD metrics
    dcd_df = opo_df[opo_df['pathway'] == 'DCD']
    dcd_volume = len(dcd_df)
    dcd_approach_rate = dcd_df['approached'].mean() if len(dcd_df) > 0 else 0
    
    # Total
    total_volume = len(opo_df)
    dbd_fraction = dbd_volume / total_volume if total_volume > 0 else 0
    
    opo_pathway_stats.append({
        'opo': opo,
        'total_volume': total_volume,
        'dbd_volume': dbd_volume,
        'dcd_volume': dcd_volume,
        'dbd_fraction': dbd_fraction,
        'dbd_approach_rate': dbd_approach_rate,
        'dcd_approach_rate': dcd_approach_rate
    })

opo_df = pd.DataFrame(opo_pathway_stats)

print("\nOPO Pathway Statistics:")
print(opo_df.to_string(index=False))

# Correlation: DBD volume vs DCD approach rate
correlation = opo_df['dbd_fraction'].corr(opo_df['dcd_approach_rate'])
p_value = stats.pearsonr(opo_df['dbd_fraction'], opo_df['dcd_approach_rate'])[1]

print(f"\nCorrelation (DBD fraction vs DCD approach rate): r = {correlation:.3f} (p = {p_value:.3f})")

if correlation < -0.3 and p_value < 0.05:
    print("✓ SIGNIFICANT NEGATIVE correlation: Higher DBD volume → Lower DCD approach rate")
    print("  INTERPRETATION: Evidence of resource competition between pathways")
else:
    print("○ No significant correlation: Resource competition hypothesis not supported")

# Visualization
fig, ax = plt.subplots(figsize=(10, 7))

scatter = ax.scatter(opo_df['dbd_fraction'], opo_df['dcd_approach_rate'],
                    s=opo_df['total_volume']/10, alpha=0.7, 
                    c=opo_df['dbd_approach_rate'], cmap='RdYlGn',
                    edgecolors='black', linewidth=1.5)

# Labels
for idx, row in opo_df.iterrows():
    ax.annotate(row['opo'], 
               (row['dbd_fraction'], row['dcd_approach_rate']),
               xytext=(5, 5), textcoords='offset points',
               fontsize=10, fontweight='bold')

# Trend line
z = np.polyfit(opo_df['dbd_fraction'], opo_df['dcd_approach_rate'], 1)
p = np.poly1d(z)
x_trend = np.linspace(opo_df['dbd_fraction'].min(), opo_df['dbd_fraction'].max(), 100)
ax.plot(x_trend, p(x_trend), "r--", linewidth=2, alpha=0.7, 
        label=f'Trend (r={correlation:.3f}, p={p_value:.3f})')

ax.set_xlabel('DBD Fraction of MSCs', fontsize=12, fontweight='bold')
ax.set_ylabel('DCD Approach Rate', fontsize=12, fontweight='bold')
ax.set_title('Pathway Competition: DBD Volume vs DCD Approach Rate', 
            fontsize=14, fontweight='bold')
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.grid(alpha=0.3)
ax.legend()

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('DBD Approach Rate', fontsize=10)
cbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'pathway_competition_opo.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {FIGURES_DIR / 'pathway_competition_opo.png'}")

# ============================================================================
# TEST 2: Temporal Competition (Weekend Effect by Pathway)
# ============================================================================

print("\n" + "="*80)
print("TEST 2: Are DCD approach rates higher on weekends (lower DBD volume)?")
print("="*80)

# Weekend vs weekday by pathway
pathway_temporal = mscs.groupby(['pathway', 'is_weekend']).agg({
    'patient_id': 'count',
    'approached': 'sum'
}).rename(columns={'patient_id': 'volume', 'approached': 'approached_count'})

pathway_temporal['approach_rate'] = pathway_temporal['approached_count'] / pathway_temporal['volume']

print("\nApproach Rates by Pathway and Weekend:")
print(pathway_temporal)

# Calculate weekend effect for each pathway
dbd_weekday = pathway_temporal.loc[('DBD', False), 'approach_rate']
dbd_weekend = pathway_temporal.loc[('DBD', True), 'approach_rate']
dcd_weekday = pathway_temporal.loc[('DCD', False), 'approach_rate']
dcd_weekend = pathway_temporal.loc[('DCD', True), 'approach_rate']

dbd_weekend_effect = (dbd_weekend - dbd_weekday) / dbd_weekday
dcd_weekend_effect = (dcd_weekend - dcd_weekday) / dcd_weekday

print(f"\nDBD Weekend Effect: {dbd_weekend_effect:+.1%}")
print(f"DCD Weekend Effect: {dcd_weekend_effect:+.1%}")

if dcd_weekend_effect > dbd_weekend_effect:
    print("\n✓ DCD weekend effect is STRONGER than DBD")
    print("  INTERPRETATION: DCD benefits more from reduced competition on weekends")
else:
    print("\n○ DCD weekend effect is not stronger than DBD")
    print("  INTERPRETATION: Weekend effect is pathway-independent")

# Visualization
fig, ax = plt.subplots(figsize=(10, 7))

x = np.arange(2)
width = 0.35

weekday_rates = [dbd_weekday, dcd_weekday]
weekend_rates = [dbd_weekend, dcd_weekend]

bars1 = ax.bar(x - width/2, weekday_rates, width, label='Weekday', color='steelblue', alpha=0.8)
bars2 = ax.bar(x + width/2, weekend_rates, width, label='Weekend', color='orange', alpha=0.8)

ax.set_ylabel('Approach Rate', fontsize=12, fontweight='bold')
ax.set_title('Weekend Effect by Pathway: Evidence of Competition?', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['DBD', 'DCD'], fontsize=12)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add percentage change labels
for i, (wd, we) in enumerate(zip(weekday_rates, weekend_rates)):
    change = (we - wd) / wd
    ax.text(i, max(wd, we) + 0.02, f'{change:+.1%}', 
           ha='center', fontsize=10, fontweight='bold',
           color='green' if change > 0 else 'red')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'pathway_weekend_effect.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {FIGURES_DIR / 'pathway_weekend_effect.png'}")

# ============================================================================
# TEST 3: Daily Competition (Same-Day DBD Volume vs DCD Approach Rate)
# ============================================================================

print("\n" + "="*80)
print("TEST 3: Within-day competition (DBD volume vs DCD approach rate)")
print("="*80)

# Calculate daily metrics by OPO
daily_competition = []

for opo in mscs['opo'].unique():
    opo_df = mscs[mscs['opo'] == opo]
    
    for date in opo_df['date'].unique():
        day_df = opo_df[opo_df['date'] == date]
        
        dbd_day = day_df[day_df['pathway'] == 'DBD']
        dcd_day = day_df[day_df['pathway'] == 'DCD']
        
        if len(dcd_day) > 0:  # Only include days with DCD referrals
            daily_competition.append({
                'opo': opo,
                'date': date,
                'dbd_volume': len(dbd_day),
                'dcd_volume': len(dcd_day),
                'dcd_approached': dcd_day['approached'].sum(),
                'dcd_approach_rate': dcd_day['approached'].mean()
            })

daily_df = pd.DataFrame(daily_competition)

print(f"\nDays with DCD referrals: {len(daily_df):,}")

# Correlation: Same-day DBD volume vs DCD approach rate
correlation = daily_df['dbd_volume'].corr(daily_df['dcd_approach_rate'])
p_value = stats.pearsonr(daily_df['dbd_volume'], daily_df['dcd_approach_rate'])[1]

print(f"\nCorrelation (Same-day DBD volume vs DCD approach rate): r = {correlation:.3f} (p = {p_value:.3f})")

if correlation < -0.1 and p_value < 0.05:
    print("✓ SIGNIFICANT NEGATIVE correlation: Higher same-day DBD volume → Lower DCD approach rate")
    print("  INTERPRETATION: Strong evidence of within-day resource competition")
else:
    print("○ No significant correlation: Within-day competition hypothesis not supported")

# Visualization
fig, ax = plt.subplots(figsize=(10, 7))

# Bin DBD volume for clearer visualization
daily_df['dbd_volume_bin'] = pd.cut(daily_df['dbd_volume'], bins=[0, 1, 2, 3, 5, 100], 
                                     labels=['0-1', '2', '3', '4-5', '6+'])

binned_stats = daily_df.groupby('dbd_volume_bin').agg({
    'dcd_approach_rate': 'mean',
    'date': 'count'
}).rename(columns={'date': 'n_days'})

ax.bar(range(len(binned_stats)), binned_stats['dcd_approach_rate'], 
       color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)

ax.set_xlabel('Same-Day DBD Volume', fontsize=12, fontweight='bold')
ax.set_ylabel('DCD Approach Rate', fontsize=12, fontweight='bold')
ax.set_title('Within-Day Competition: DBD Volume vs DCD Approach Rate', 
            fontsize=14, fontweight='bold')
ax.set_xticks(range(len(binned_stats)))
ax.set_xticklabels(binned_stats.index)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.grid(axis='y', alpha=0.3)

# Add sample size labels
for i, (idx, row) in enumerate(binned_stats.iterrows()):
    ax.text(i, row['dcd_approach_rate'] + 0.005, f"n={row['n_days']:.0f}", 
           ha='center', fontsize=9, color='gray')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'pathway_daily_competition.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {FIGURES_DIR / 'pathway_daily_competition.png'}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("PATHWAY COMPETITION ANALYSIS: SUMMARY")
print("="*80)

print("\nHypothesis: OPOs trade off DBD vs DCD pursuit due to resource constraints")

print("\nTest 1 (OPO-Level Competition):")
print(f"  - Correlation: r = {opo_df['dbd_fraction'].corr(opo_df['dcd_approach_rate']):.3f}")
if opo_df['dbd_fraction'].corr(opo_df['dcd_approach_rate']) < -0.3:
    print("  - Result: ✓ Evidence of resource competition")
else:
    print("  - Result: ○ No strong evidence")

print("\nTest 2 (Weekend Effect by Pathway):")
print(f"  - DBD weekend effect: {dbd_weekend_effect:+.1%}")
print(f"  - DCD weekend effect: {dcd_weekend_effect:+.1%}")
if dcd_weekend_effect > dbd_weekend_effect:
    print("  - Result: ✓ DCD benefits more from reduced competition")
else:
    print("  - Result: ○ Weekend effect is pathway-independent")

print("\nTest 3 (Within-Day Competition):")
print(f"  - Correlation: r = {correlation:.3f}")
if correlation < -0.1 and p_value < 0.05:
    print("  - Result: ✓ Strong evidence of within-day competition")
else:
    print("  - Result: ○ No strong evidence")

print("\n" + "="*80)
print("CONCLUSION:")
if (opo_df['dbd_fraction'].corr(opo_df['dcd_approach_rate']) < -0.3 or 
    dcd_weekend_effect > dbd_weekend_effect or 
    (correlation < -0.1 and p_value < 0.05)):
    print("✓ PATHWAY COMPETITION HYPOTHESIS SUPPORTED")
    print("  OPOs face resource constraints that force trade-offs between DBD and DCD pursuit.")
    print("  This explains the systematic DCD under-utilization.")
else:
    print("○ PATHWAY COMPETITION HYPOTHESIS NOT STRONGLY SUPPORTED")
    print("  DCD under-utilization may be due to other factors (training, protocols, culture).")
print("="*80)
