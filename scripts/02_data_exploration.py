#!/usr/bin/env python3
"""
ORCHID Dataset Exploration Script
==================================
Deep dive into the referral data to verify we can construct the Loss Waterfall.

This script explores:
1. Pipeline stage variables (approached, authorized, procured, transplanted)
2. Value distributions and data types
3. Funnel analysis: how many referrals make it through each stage
4. Missing data patterns
5. Organ-specific outcomes
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# ANSI colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
RESET = '\033[0m'

def print_header(text):
    print(f"\n{BLUE}{'='*80}{RESET}")
    print(f"{BLUE}{text:^80}{RESET}")
    print(f"{BLUE}{'='*80}{RESET}\n")

def print_subheader(text):
    print(f"\n{CYAN}{text}{RESET}")
    print(f"{CYAN}{'-'*len(text)}{RESET}")

def explore_orchid_data():
    # Load data
    data_dir = Path.home() / 'physionet.org' / 'files' / 'orchid' / '2.1.1'
    referral_file = data_dir / 'OPOReferrals.csv'
    
    print_header("ORCHID DATASET DEEP EXPLORATION")
    
    print("Loading OPOReferrals.csv...")
    df = pd.read_csv(referral_file, low_memory=False)
    print(f"{GREEN}✓ Loaded {len(df):,} referral records with {len(df.columns)} columns{RESET}\n")
    
    # ============================================================================
    # SECTION 1: ALL COLUMNS
    # ============================================================================
    print_header("SECTION 1: COMPLETE COLUMN LIST")
    
    print("All 38 columns in dataset:\n")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        non_null = df[col].notna().sum()
        null_pct = 100 * (1 - non_null / len(df))
        print(f"  {i:2d}. {col:30s} | Type: {str(dtype):10s} | Missing: {null_pct:5.1f}%")
    
    # ============================================================================
    # SECTION 2: PIPELINE STAGE VARIABLES
    # ============================================================================
    print_header("SECTION 2: PIPELINE STAGE VARIABLES")
    
    pipeline_vars = ['approached', 'authorized', 'procured', 'transplanted']
    
    print("Examining critical pipeline variables:\n")
    
    for var in pipeline_vars:
        if var in df.columns:
            print_subheader(f"Variable: {var}")
            
            # Basic stats
            print(f"Data type: {df[var].dtype}")
            print(f"Non-null count: {df[var].notna().sum():,} ({100*df[var].notna().sum()/len(df):.1f}%)")
            print(f"Null count: {df[var].isna().sum():,} ({100*df[var].isna().sum()/len(df):.1f}%)")
            
            # Unique values
            unique_vals = df[var].dropna().unique()
            print(f"Unique values: {len(unique_vals)}")
            
            # Value distribution
            value_counts = df[var].value_counts(dropna=False)
            print("\nValue distribution:")
            for val, count in value_counts.items():
                pct = 100 * count / len(df)
                print(f"  {val}: {count:,} ({pct:.1f}%)")
            
            print()
        else:
            print(f"{RED}✗ {var} not found in dataset{RESET}\n")
    
    # ============================================================================
    # SECTION 3: FUNNEL ANALYSIS
    # ============================================================================
    print_header("SECTION 3: REFERRAL FUNNEL ANALYSIS")
    
    print("Calculating stage-by-stage conversion rates:\n")
    
    # Total referrals
    total_referrals = len(df)
    print(f"1. Total Referrals: {total_referrals:,}")
    
    # Check if variables are binary or need interpretation
    if 'approached' in df.columns:
        # Try to interpret as binary
        approached_count = df['approached'].sum() if df['approached'].dtype in ['int64', 'float64'] else df[df['approached'] == 1].shape[0]
        approached_rate = approached_count / total_referrals
        print(f"2. Approached: {approached_count:,} ({100*approached_rate:.1f}%)")
        print(f"   → Sorting Loss: {total_referrals - approached_count:,} ({100*(1-approached_rate):.1f}%)")
    else:
        approached_count = None
        print(f"{YELLOW}⚠ Cannot calculate 'approached' - variable not found or unclear{RESET}")
    
    if 'authorized' in df.columns and approached_count:
        authorized_count = df['authorized'].sum() if df['authorized'].dtype in ['int64', 'float64'] else df[df['authorized'] == 1].shape[0]
        authorized_rate = authorized_count / approached_count if approached_count > 0 else 0
        print(f"3. Authorized: {authorized_count:,} ({100*authorized_rate:.1f}% of approached)")
        print(f"   → Authorization Loss: {approached_count - authorized_count:,} ({100*(1-authorized_rate):.1f}%)")
    else:
        authorized_count = None
        print(f"{YELLOW}⚠ Cannot calculate 'authorized'{RESET}")
    
    if 'procured' in df.columns and authorized_count:
        procured_count = df['procured'].sum() if df['procured'].dtype in ['int64', 'float64'] else df[df['procured'] == 1].shape[0]
        procured_rate = procured_count / authorized_count if authorized_count > 0 else 0
        print(f"4. Procured: {procured_count:,} ({100*procured_rate:.1f}% of authorized)")
        print(f"   → Procurement Loss: {authorized_count - procured_count:,} ({100*(1-procured_rate):.1f}%)")
    else:
        procured_count = None
        print(f"{YELLOW}⚠ Cannot calculate 'procured'{RESET}")
    
    if 'transplanted' in df.columns and procured_count:
        transplanted_count = df['transplanted'].sum() if df['transplanted'].dtype in ['int64', 'float64'] else df[df['transplanted'] == 1].shape[0]
        transplanted_rate = transplanted_count / procured_count if procured_count > 0 else 0
        print(f"5. Transplanted: {transplanted_count:,} ({100*transplanted_rate:.1f}% of procured)")
        print(f"   → Placement Loss: {procured_count - transplanted_count:,} ({100*(1-transplanted_rate):.1f}%)")
    else:
        transplanted_count = None
        print(f"{YELLOW}⚠ Cannot calculate 'transplanted'{RESET}")
    
    # Overall conversion
    if transplanted_count:
        overall_rate = transplanted_count / total_referrals
        print(f"\n{GREEN}Overall Conversion: {transplanted_count:,} / {total_referrals:,} = {100*overall_rate:.1f}%{RESET}")
        print(f"{RED}Total System Loss: {total_referrals - transplanted_count:,} ({100*(1-overall_rate):.1f}%){RESET}")
    
    # ============================================================================
    # SECTION 4: ORGAN-SPECIFIC OUTCOMES
    # ============================================================================
    print_header("SECTION 4: ORGAN-SPECIFIC OUTCOMES")
    
    organ_outcome_cols = [col for col in df.columns if col.startswith('outcome_')]
    
    if organ_outcome_cols:
        print(f"Found {len(organ_outcome_cols)} organ-specific outcome columns:\n")
        
        for col in organ_outcome_cols:
            organ = col.replace('outcome_', '').title()
            print_subheader(f"Organ: {organ}")
            
            outcome_counts = df[col].value_counts()
            print("Outcomes:")
            for outcome, count in outcome_counts.head(10).items():
                pct = 100 * count / len(df)
                print(f"  {outcome}: {count:,} ({pct:.1f}%)")
            
            # Count transplanted organs
            transplanted = df[df[col] == 'Transplanted'].shape[0]
            print(f"\n{GREEN}Transplanted: {transplanted:,} ({100*transplanted/len(df):.1f}%){RESET}\n")
    else:
        print(f"{YELLOW}No organ-specific outcome columns found{RESET}")
    
    # ============================================================================
    # SECTION 5: CROSS-TABULATION
    # ============================================================================
    print_header("SECTION 5: STAGE CROSS-TABULATION")
    
    if all(var in df.columns for var in ['approached', 'authorized', 'procured']):
        print("Cross-tab: Approached × Authorized × Procured\n")
        
        # Create contingency table
        crosstab = pd.crosstab(
            [df['approached'], df['authorized']], 
            df['procured'], 
            margins=True
        )
        print(crosstab)
        print()
    else:
        print(f"{YELLOW}Cannot create cross-tabulation - missing variables{RESET}")
    
    # ============================================================================
    # SECTION 6: MISSING DATA PATTERNS
    # ============================================================================
    print_header("SECTION 6: MISSING DATA PATTERNS")
    
    print("Missing data for pipeline variables:\n")
    
    for var in pipeline_vars:
        if var in df.columns:
            missing_count = df[var].isna().sum()
            missing_pct = 100 * missing_count / len(df)
            
            if missing_pct > 0:
                print(f"{YELLOW}{var}: {missing_count:,} missing ({missing_pct:.1f}%){RESET}")
            else:
                print(f"{GREEN}{var}: No missing data{RESET}")
    
    # ============================================================================
    # SECTION 7: OPO-LEVEL ANALYSIS
    # ============================================================================
    print_header("SECTION 7: OPO-LEVEL CONVERSION RATES")
    
    if 'opo' in df.columns and 'transplanted' in df.columns:
        print("Conversion rates by OPO:\n")
        
        opo_stats = df.groupby('opo').agg({
            'patient_id': 'count',
            'approached': 'sum' if 'approached' in df.columns else 'count',
            'authorized': 'sum' if 'authorized' in df.columns else 'count',
            'procured': 'sum' if 'procured' in df.columns else 'count',
            'transplanted': 'sum' if 'transplanted' in df.columns else 'count'
        }).rename(columns={'patient_id': 'total_referrals'})
        
        # Calculate rates
        opo_stats['conversion_rate'] = opo_stats['transplanted'] / opo_stats['total_referrals']
        opo_stats = opo_stats.sort_values('conversion_rate', ascending=False)
        
        print(opo_stats.to_string())
        print()
        
        # Variance analysis
        print(f"Conversion rate variance across OPOs:")
        print(f"  Mean: {opo_stats['conversion_rate'].mean():.3f}")
        print(f"  Std:  {opo_stats['conversion_rate'].std():.3f}")
        print(f"  Min:  {opo_stats['conversion_rate'].min():.3f}")
        print(f"  Max:  {opo_stats['conversion_rate'].max():.3f}")
        print(f"  Range: {opo_stats['conversion_rate'].max() - opo_stats['conversion_rate'].min():.3f}")
    
    # ============================================================================
    # SECTION 8: RECOMMENDATIONS
    # ============================================================================
    print_header("SECTION 8: ANALYSIS RECOMMENDATIONS")
    
    print("Based on this exploration:\n")
    
    # Check if we can do Loss Waterfall
    can_do_waterfall = all(var in df.columns for var in pipeline_vars)
    
    if can_do_waterfall:
        print(f"{GREEN}✓ Loss Waterfall Decomposition: FEASIBLE{RESET}")
        print("  All required variables present (approached, authorized, procured, transplanted)")
    else:
        print(f"{RED}✗ Loss Waterfall Decomposition: NOT FEASIBLE{RESET}")
        missing = [var for var in pipeline_vars if var not in df.columns]
        print(f"  Missing variables: {', '.join(missing)}")
    
    print()
    
    # Check organ-level analysis
    if organ_outcome_cols:
        print(f"{GREEN}✓ Organ-Level Analysis: FEASIBLE{RESET}")
        print(f"  {len(organ_outcome_cols)} organ-specific outcome columns available")
    else:
        print(f"{YELLOW}⚠ Organ-Level Analysis: LIMITED{RESET}")
        print("  No organ-specific outcome columns found")
    
    print()
    
    # Check OPO comparison
    if 'opo' in df.columns:
        print(f"{GREEN}✓ OPO Comparison: FEASIBLE{RESET}")
        print(f"  6 OPOs available for comparative analysis")
    else:
        print(f"{RED}✗ OPO Comparison: NOT FEASIBLE{RESET}")
    
    print()
    
    # Temporal analysis
    date_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
    if date_cols:
        print(f"{GREEN}✓ Temporal Analysis: FEASIBLE{RESET}")
        print(f"  {len(date_cols)} date/time columns available")
    else:
        print(f"{YELLOW}⚠ Temporal Analysis: LIMITED{RESET}")
    
    # ============================================================================
    # SAVE SUMMARY
    # ============================================================================
    print_header("SAVING EXPLORATION SUMMARY")
    
    summary = {
        'total_referrals': int(total_referrals),
        'total_columns': len(df.columns),
        'pipeline_variables_present': [var for var in pipeline_vars if var in df.columns],
        'pipeline_variables_missing': [var for var in pipeline_vars if var not in df.columns],
        'organ_outcome_columns': organ_outcome_cols,
        'can_do_loss_waterfall': can_do_waterfall,
        'opo_count': df['opo'].nunique() if 'opo' in df.columns else 0,
        'date_columns': date_cols
    }
    
    output_file = data_dir / 'exploration_summary.json'
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {output_file}")
    
    print(f"\n{GREEN}{'='*80}{RESET}")
    print(f"{GREEN}EXPLORATION COMPLETE{RESET}")
    print(f"{GREEN}{'='*80}{RESET}\n")

if __name__ == '__main__':
    explore_orchid_data()
