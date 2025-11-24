#!/usr/bin/env python3
"""
Medically Suitable Candidate (MSC) Identification - Sensitivity Analysis
========================================================================

Implements THREE approaches to Layer 3 (empirical ranges) and compares results:

Approach A: ABSOLUTE MAXIMUM (Most Liberal)
  - "If it has been done successfully once, it is an MSC"
  - Uses max observed age/BMI from successful cases
  - Maximizes denominator, captures all risk-aversion

Approach B: 99TH PERCENTILE (Moderate)
  - "If top 1% can do it, it's viable"
  - Uses 99th percentile of successful cases
  - Balances outliers vs. best practice

Approach C: BEST-PERFORMING OPO (Benchmark)
  - "If the best OPO can do it, others should too"
  - Uses empirical ranges from highest-performing OPO
  - Measures underperformance relative to achievable standard

All three share:
- Layer 1: Absolute contraindications (clinical guidelines)
- Layer 2: DBD vs DCD pathway separation
- Layer 3: Varies by approach

Author: Noah
Date: 2024-11-24
Version: 3.0 (Sensitivity Analysis)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ANSI colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
MAGENTA = '\033[95m'
RESET = '\033[0m'

def print_header(text):
    print(f"\n{BLUE}{'='*80}{RESET}")
    print(f"{BLUE}{text:^80}{RESET}")
    print(f"{BLUE}{'='*80}{RESET}\n")

def print_subheader(text):
    print(f"\n{CYAN}{text}{RESET}")
    print(f"{CYAN}{'-'*len(text)}{RESET}")

def print_success(text):
    print(f"{GREEN}✓ {text}{RESET}")

def print_warning(text):
    print(f"{YELLOW}⚠ {text}{RESET}")

def print_error(text):
    print(f"{RED}✗ {text}{RESET}")

# ============================================================================
# CLINICAL GUIDELINES (Layer 1: Absolute Contraindications)
# ============================================================================

ABSOLUTE_CONTRAINDICATIONS = {
    'cause_of_death': [
        'Malignant Neoplasm',
        'Cancer',
        'Carcinoma',
        'HIV',
        'AIDS',
        'Rabies',
        'Creutzfeldt-Jakob',
        'CJD',
        'Active Tuberculosis',
        'TB'
    ]
}

# Biological plausibility bounds (prevent absurd values)
BIOLOGICAL_LIMITS = {
    'age': {'min': 0, 'max': 100},
    'bmi': {'min': 10, 'max': 60}
}

# Organ-specific biological age ceilings (DBD)
BIOLOGICAL_AGE_CEILING_DBD = {
    'liver': 100,
    'kidney': 90,
    'heart': 70,
    'lung': 75,
    'pancreas': 60,
    'intestine': 65
}

# Organ-specific biological age ceilings (DCD)
BIOLOGICAL_AGE_CEILING_DCD = {
    'liver': 70,
    'kidney': 75,
    'heart': 55,
    'lung': 60,
    'pancreas': 50,
    'intestine': 55
}

class SensitivityMSCAnalyzer:
    def __init__(self, data_path, random_state=42):
        self.data_path = Path(data_path)
        self.random_state = random_state
        
        self.organs = ['heart', 'liver', 'kidney_left', 'kidney_right', 
                       'lung_left', 'lung_right', 'pancreas', 'intestine']
        
        self.organ_families = {
            'heart': 'heart',
            'liver': 'liver',
            'kidney_left': 'kidney',
            'kidney_right': 'kidney',
            'lung_left': 'lung',
            'lung_right': 'lung',
            'pancreas': 'pancreas',
            'intestine': 'intestine'
        }
        
        # Storage for three approaches
        self.approaches = {
            'absolute_max': {},
            'percentile_99': {},
            'best_opo': {}
        }
        
        self.results = {}
        
    def load_data(self):
        """Load ORCHID referral data"""
        print_header("LOADING DATA")
        
        referral_file = self.data_path / 'OPOReferrals.csv'
        print(f"Loading {referral_file}...")
        
        self.df = pd.read_csv(referral_file, low_memory=False)
        print_success(f"Loaded {len(self.df):,} referral records")
        
        # Calculate BMI
        self.df['bmi'] = self.df['weight_kg'] / ((self.df['height_in'] * 0.0254) ** 2)
        print_success("Calculated BMI from height and weight")
        
        return self.df
    
    def check_absolute_contraindications(self, row):
        """Check Layer 1: Absolute contraindications"""
        if pd.notna(row.get('cause_of_death_unos')):
            cod = str(row['cause_of_death_unos']).upper()
            for contraindication in ABSOLUTE_CONTRAINDICATIONS['cause_of_death']:
                if contraindication.upper() in cod:
                    return True
        
        if pd.notna(row.get('cause_of_death_opo')):
            cod = str(row['cause_of_death_opo']).upper()
            for contraindication in ABSOLUTE_CONTRAINDICATIONS['cause_of_death']:
                if contraindication.upper() in cod:
                    return True
        
        return False
    
    def get_successful_cases(self, df, organs):
        """Get cases where at least one organ in family was procured"""
        successful = pd.DataFrame()
        
        for organ in organs:
            outcome_col = f'outcome_{organ}'
            if outcome_col in df.columns:
                organ_procured = df[df[outcome_col].notna()]
                successful = pd.concat([successful, organ_procured])
        
        successful = successful.drop_duplicates(subset=['patient_id'])
        return successful
    
    def extract_ranges_approach_a(self):
        """Approach A: Absolute Maximum"""
        print_header("APPROACH A: ABSOLUTE MAXIMUM")
        print("Logic: 'If it has been done successfully once, it is an MSC'\n")
        
        dbd_cases = self.df[self.df['brain_death'] == True]
        dcd_cases = self.df[self.df['brain_death'] == False]
        
        for organ_family in set(self.organ_families.values()):
            family_organs = [org for org, fam in self.organ_families.items() if fam == organ_family]
            
            # DBD
            dbd_successful = self.get_successful_cases(dbd_cases, family_organs)
            if len(dbd_successful) >= 10:
                age_valid = dbd_successful['age'].dropna()
                bmi_valid = dbd_successful['bmi'].dropna()
                
                if len(age_valid) >= 10:
                    age_max = age_valid.max()
                    age_max = min(age_max, BIOLOGICAL_AGE_CEILING_DBD.get(organ_family, 70))
                    
                    self.approaches['absolute_max'][f'{organ_family}_dbd'] = {
                        'age': {'min': 0, 'max': age_max},
                        'n': len(dbd_successful)
                    }
                    
                    print(f"{organ_family.upper()} DBD: Age 0-{age_max:.0f} (n={len(dbd_successful)})")
            
            # DCD
            dcd_successful = self.get_successful_cases(dcd_cases, family_organs)
            if len(dcd_successful) >= 10:
                age_valid = dcd_successful['age'].dropna()
                
                if len(age_valid) >= 10:
                    age_max = age_valid.max()
                    age_max = min(age_max, BIOLOGICAL_AGE_CEILING_DCD.get(organ_family, 60))
                    
                    self.approaches['absolute_max'][f'{organ_family}_dcd'] = {
                        'age': {'min': 0, 'max': age_max},
                        'n': len(dcd_successful)
                    }
                    
                    print(f"{organ_family.upper()} DCD: Age 0-{age_max:.0f} (n={len(dcd_successful)})")
    
    def extract_ranges_approach_b(self):
        """Approach B: 99th Percentile"""
        print_header("APPROACH B: 99TH PERCENTILE")
        print("Logic: 'If top 1% can do it, it's viable'\n")
        
        dbd_cases = self.df[self.df['brain_death'] == True]
        dcd_cases = self.df[self.df['brain_death'] == False]
        
        for organ_family in set(self.organ_families.values()):
            family_organs = [org for org, fam in self.organ_families.items() if fam == organ_family]
            
            # DBD
            dbd_successful = self.get_successful_cases(dbd_cases, family_organs)
            if len(dbd_successful) >= 30:
                age_valid = dbd_successful['age'].dropna()
                
                if len(age_valid) >= 30:
                    age_99 = np.percentile(age_valid, 99)
                    age_99 = min(age_99, BIOLOGICAL_AGE_CEILING_DBD.get(organ_family, 70))
                    
                    self.approaches['percentile_99'][f'{organ_family}_dbd'] = {
                        'age': {'min': 0, 'max': age_99},
                        'n': len(dbd_successful)
                    }
                    
                    print(f"{organ_family.upper()} DBD: Age 0-{age_99:.0f} (n={len(dbd_successful)})")
            
            # DCD
            dcd_successful = self.get_successful_cases(dcd_cases, family_organs)
            if len(dcd_successful) >= 30:
                age_valid = dcd_successful['age'].dropna()
                
                if len(age_valid) >= 30:
                    age_99 = np.percentile(age_valid, 99)
                    age_99 = min(age_99, BIOLOGICAL_AGE_CEILING_DCD.get(organ_family, 60))
                    
                    self.approaches['percentile_99'][f'{organ_family}_dcd'] = {
                        'age': {'min': 0, 'max': age_99},
                        'n': len(dcd_successful)
                    }
                    
                    print(f"{organ_family.upper()} DCD: Age 0-{age_99:.0f} (n={len(dcd_successful)})")
    
    def extract_ranges_approach_c(self):
        """Approach C: Best-Performing OPO"""
        print_header("APPROACH C: BEST-PERFORMING OPO")
        print("Logic: 'If the best OPO can do it, others should too'\n")
        
        # First, identify best-performing OPO (highest overall conversion rate)
        opo_conversion = {}
        for opo in self.df['opo'].unique():
            opo_df = self.df[self.df['opo'] == opo]
            approached = opo_df['approached'].sum()
            transplanted = opo_df['transplanted'].sum()
            if approached > 0:
                opo_conversion[opo] = transplanted / len(opo_df)
        
        best_opo = max(opo_conversion, key=opo_conversion.get)
        print(f"Best-performing OPO: {best_opo} (conversion rate: {100*opo_conversion[best_opo]:.1f}%)\n")
        
        best_opo_df = self.df[self.df['opo'] == best_opo]
        dbd_cases = best_opo_df[best_opo_df['brain_death'] == True]
        dcd_cases = best_opo_df[best_opo_df['brain_death'] == False]
        
        for organ_family in set(self.organ_families.values()):
            family_organs = [org for org, fam in self.organ_families.items() if fam == organ_family]
            
            # DBD
            dbd_successful = self.get_successful_cases(dbd_cases, family_organs)
            if len(dbd_successful) >= 10:
                age_valid = dbd_successful['age'].dropna()
                
                if len(age_valid) >= 10:
                    age_95 = np.percentile(age_valid, 95)
                    age_95 = min(age_95, BIOLOGICAL_AGE_CEILING_DBD.get(organ_family, 70))
                    
                    self.approaches['best_opo'][f'{organ_family}_dbd'] = {
                        'age': {'min': 0, 'max': age_95},
                        'n': len(dbd_successful),
                        'opo': best_opo
                    }
                    
                    print(f"{organ_family.upper()} DBD: Age 0-{age_95:.0f} (n={len(dbd_successful)})")
            
            # DCD
            dcd_successful = self.get_successful_cases(dcd_cases, family_organs)
            if len(dcd_successful) >= 10:
                age_valid = dcd_successful['age'].dropna()
                
                if len(age_valid) >= 10:
                    age_95 = np.percentile(age_valid, 95)
                    age_95 = min(age_95, BIOLOGICAL_AGE_CEILING_DCD.get(organ_family, 60))
                    
                    self.approaches['best_opo'][f'{organ_family}_dcd'] = {
                        'age': {'min': 0, 'max': age_95},
                        'n': len(dcd_successful),
                        'opo': best_opo
                    }
                    
                    print(f"{organ_family.upper()} DCD: Age 0-{age_95:.0f} (n={len(dcd_successful)})")
    
    def check_msc_status(self, row, approach_name):
        """Check if referral is MSC under given approach"""
        # Layer 1: Absolute contraindications
        if self.check_absolute_contraindications(row):
            return {
                'is_msc': False,
                'viable_organs': [],
                'reason': 'absolute_contraindication'
            }
        
        # Layer 2: Determine donation type
        donation_type = 'dbd' if row['brain_death'] else 'dcd'
        
        # Layer 3: Check viability using approach-specific ranges
        ranges_dict = self.approaches[approach_name]
        viable_organs = []
        
        for organ_family in set(self.organ_families.values()):
            ranges_key = f'{organ_family}_{donation_type}'
            ranges = ranges_dict.get(ranges_key)
            
            if ranges is None:
                continue
            
            # Check age
            if 'age' in ranges:
                if pd.isna(row['age']):
                    continue
                if not (ranges['age']['min'] <= row['age'] <= ranges['age']['max']):
                    continue
            
            viable_organs.append(organ_family)
        
        is_msc = len(viable_organs) > 0
        
        return {
            'is_msc': is_msc,
            'viable_organs': viable_organs,
            'donation_type': donation_type,
            'reason': 'viable' if is_msc else 'outside_range'
        }
    
    def identify_mscs_all_approaches(self):
        """Identify MSCs under all three approaches"""
        print_header("IDENTIFYING MSCs UNDER ALL THREE APPROACHES")
        
        for approach_name in ['absolute_max', 'percentile_99', 'best_opo']:
            print_subheader(f"Approach: {approach_name.replace('_', ' ').title()}")
            
            msc_results = self.df.apply(
                lambda row: self.check_msc_status(row, approach_name), 
                axis=1, 
                result_type='expand'
            )
            
            self.df[f'is_msc_{approach_name}'] = msc_results['is_msc']
            self.df[f'viable_organs_{approach_name}'] = msc_results['viable_organs']
            
            msc_count = self.df[f'is_msc_{approach_name}'].sum()
            msc_rate = msc_count / len(self.df)
            
            print(f"MSCs: {msc_count:,} / {len(self.df):,} ({100*msc_rate:.1f}%)")
    
    def calculate_loss_waterfall(self, approach_name, df=None, label="Full Dataset"):
        """Calculate Loss Waterfall for given approach"""
        if df is None:
            df = self.df
        
        msc_col = f'is_msc_{approach_name}'
        
        # MSCs
        mscs = df[df[msc_col]]
        msc_count = len(mscs)
        
        if msc_count == 0:
            return None
        
        # Stages
        n_approached = mscs['approached'].sum()
        r_s = n_approached / msc_count
        
        msc_approached = mscs[mscs['approached']]
        n_authorized = msc_approached['authorized'].sum() if len(msc_approached) > 0 else 0
        r_a = n_authorized / n_approached if n_approached > 0 else 0
        
        msc_authorized = msc_approached[msc_approached['authorized']]
        n_procured = msc_authorized['procured'].sum() if len(msc_authorized) > 0 else 0
        r_proc = n_procured / n_authorized if n_authorized > 0 else 0
        
        msc_procured = msc_authorized[msc_authorized['procured']]
        n_transplanted = msc_procured['transplanted'].sum() if len(msc_procured) > 0 else 0
        r_p = n_transplanted / n_procured if n_procured > 0 else 0
        
        overall_conversion = n_transplanted / msc_count if msc_count > 0 else 0
        
        # Counterfactual loss decomposition
        if n_transplanted > 0:
            total_loss_count = msc_count - n_transplanted
            
            ds_if_approached = n_transplanted / n_approached if n_approached > 0 else 0
            ds_if_authorized = n_transplanted / n_authorized if n_authorized > 0 else 0
            ds_if_procured = n_transplanted / n_procured if n_procured > 0 else 0
            
            sorting_loss_cf = (msc_count - n_approached) * ds_if_approached
            auth_loss_cf = (n_approached - n_authorized) * ds_if_authorized
            proc_loss_cf = (n_authorized - n_procured) * ds_if_procured
            place_loss_cf = n_procured - n_transplanted
            
            sorting_pct = sorting_loss_cf / total_loss_count if total_loss_count > 0 else 0
            auth_pct = auth_loss_cf / total_loss_count if total_loss_count > 0 else 0
            proc_pct = proc_loss_cf / total_loss_count if total_loss_count > 0 else 0
            place_pct = place_loss_cf / total_loss_count if total_loss_count > 0 else 0
        else:
            sorting_pct = auth_pct = proc_pct = place_pct = 0
        
        return {
            'msc_count': msc_count,
            'msc_rate': msc_count / len(df),
            'stages': {
                'sorting': {'rate': r_s, 'n_success': n_approached},
                'authorization': {'rate': r_a, 'n_success': n_authorized},
                'procurement': {'rate': r_proc, 'n_success': n_procured},
                'placement': {'rate': r_p, 'n_success': n_transplanted}
            },
            'overall_conversion': overall_conversion,
            'loss_decomposition': {
                'sorting': sorting_pct,
                'authorization': auth_pct,
                'procurement': proc_pct,
                'placement': place_pct
            }
        }
    
    def compare_approaches(self):
        """Compare all three approaches"""
        print_header("SENSITIVITY ANALYSIS: COMPARING ALL THREE APPROACHES")
        
        comparison = {}
        
        for approach_name in ['absolute_max', 'percentile_99', 'best_opo']:
            results = self.calculate_loss_waterfall(approach_name)
            if results:
                comparison[approach_name] = results
        
        # Print comparison table
        print(f"\n{'Approach':<20} {'MSCs':>10} {'MSC%':>8} {'Sort%':>8} {'Auth%':>8} {'Overall%':>10} {'Sort Loss%':>12}")
        print(f"{'-'*20} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*12}")
        
        for approach_name, res in comparison.items():
            approach_label = approach_name.replace('_', ' ').title()
            print(f"{approach_label:<20} {res['msc_count']:>10,} {100*res['msc_rate']:>7.1f}% "
                  f"{100*res['stages']['sorting']['rate']:>7.1f}% "
                  f"{100*res['stages']['authorization']['rate']:>7.1f}% "
                  f"{100*res['overall_conversion']:>9.1f}% "
                  f"{100*res['loss_decomposition']['sorting']:>11.1f}%")
        
        print(f"\n{MAGENTA}KEY INSIGHT:{RESET}")
        print(f"The range of MSC counts across approaches shows sensitivity to Layer 3 definition.")
        print(f"Sorting Loss % shows how much inefficiency is attributed to the sorting stage.")
        
        return comparison
    
    def run_full_analysis(self):
        """Run complete sensitivity analysis"""
        print(f"\n{BLUE}{'='*80}{RESET}")
        print(f"{BLUE}MSC SENSITIVITY ANALYSIS{RESET}")
        print(f"{BLUE}Comparing Three Approaches to Layer 3 (Empirical Ranges){RESET}")
        print(f"{BLUE}{'='*80}{RESET}")
        
        # Load data
        self.load_data()
        
        # Extract ranges for all three approaches
        self.extract_ranges_approach_a()
        self.extract_ranges_approach_b()
        self.extract_ranges_approach_c()
        
        # Identify MSCs under all approaches
        self.identify_mscs_all_approaches()
        
        # Compare approaches
        comparison = self.compare_approaches()
        
        # Save results
        output_file = self.data_path / 'msc_sensitivity_results.json'
        with open(output_file, 'w') as f:
            json.dump({
                'approaches': self.approaches,
                'comparison': comparison
            }, f, indent=2, default=str)
        
        print_success(f"\nResults saved to: {output_file}")
        
        # Save labeled dataset
        output_csv = self.data_path / 'orchid_with_msc_sensitivity.csv'
        self.df.to_csv(output_csv, index=False)
        print_success(f"Labeled dataset saved to: {output_csv}")
        
        print(f"\n{GREEN}{'='*80}{RESET}")
        print(f"{GREEN}SENSITIVITY ANALYSIS COMPLETE{RESET}")
        print(f"{GREEN}{'='*80}{RESET}\n")
        
        return comparison

def main():
    data_dir = Path.home() / 'physionet.org' / 'files' / 'orchid' / '2.1.1'
    
    analyzer = SensitivityMSCAnalyzer(data_dir)
    comparison = analyzer.run_full_analysis()

if __name__ == '__main__':
    main()
