#!/usr/bin/env python3
"""
ORCHID Dataset Validation Script
=================================
Pre-check validation to ensure all necessary data exists and is complete
before running the Loss Waterfall Decomposition analysis.

This script validates:
1. File existence and integrity (checksums)
2. Data completeness (row counts, schemas)
3. Referral pipeline coverage (sorting, authorization, placement stages)
4. Temporal coverage (2015-2021)
5. OPO coverage (all 6 OPOs)
6. Critical variables for Loss Waterfall analysis
"""

import os
import sys
import hashlib
import pandas as pd
from pathlib import Path
from datetime import datetime
import json

# ANSI color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

class ORCHIDValidator:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.results = {
            'file_checks': {},
            'schema_checks': {},
            'coverage_checks': {},
            'critical_variables': {},
            'summary': {}
        }
        self.errors = []
        self.warnings = []
        
    def print_header(self, text):
        print(f"\n{BLUE}{'='*80}{RESET}")
        print(f"{BLUE}{text:^80}{RESET}")
        print(f"{BLUE}{'='*80}{RESET}\n")
        
    def print_success(self, text):
        print(f"{GREEN}✓ {text}{RESET}")
        
    def print_error(self, text):
        print(f"{RED}✗ {text}{RESET}")
        self.errors.append(text)
        
    def print_warning(self, text):
        print(f"{YELLOW}⚠ {text}{RESET}")
        self.warnings.append(text)
        
    def print_info(self, text):
        print(f"  {text}")
        
    def validate_file_existence(self):
        """Check that all expected files exist"""
        self.print_header("PHASE 1: FILE EXISTENCE CHECK")
        
        expected_files = {
            'OPOReferrals.csv': 'Main referral data (CRITICAL)',
            'ABGEvents.csv': 'Arterial blood gas measurements',
            'CBCEvents.csv': 'Complete blood count data',
            'ChemistryEvents.csv': 'Lab chemistry values',
            'HemoEvents.csv': 'Hemodynamic measurements',
            'FluidBalanceEvents.csv': 'Fluid balance data',
            'SerologyEvents.csv': 'Serology test results',
            'CultureEvents.csv': 'Culture test results',
            'CalcDeaths.csv': 'Calculated death records',
            'DataDescription.csv': 'Data dictionary',
            'LICENSE.txt': 'License information',
            'SHA256SUMS.txt': 'Checksum file'
        }
        
        for filename, description in expected_files.items():
            filepath = self.data_dir / filename
            if filepath.exists():
                size_mb = filepath.stat().st_size / (1024 * 1024)
                self.print_success(f"{filename} exists ({size_mb:.1f} MB) - {description}")
                self.results['file_checks'][filename] = {
                    'exists': True,
                    'size_mb': round(size_mb, 2),
                    'description': description
                }
            else:
                self.print_error(f"{filename} NOT FOUND - {description}")
                self.results['file_checks'][filename] = {
                    'exists': False,
                    'description': description
                }
                
    def validate_checksums(self):
        """Validate file integrity using SHA256 checksums"""
        self.print_header("PHASE 2: FILE INTEGRITY CHECK")
        
        checksum_file = self.data_dir / 'SHA256SUMS.txt'
        if not checksum_file.exists():
            self.print_warning("SHA256SUMS.txt not found - skipping integrity check")
            return
            
        print("Reading checksums...")
        with open(checksum_file, 'r') as f:
            expected_checksums = {}
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) == 2:
                        expected_checksums[parts[1]] = parts[0]
        
        print(f"Found {len(expected_checksums)} checksums to validate\n")
        
        # Only validate critical files to save time
        critical_files = ['OPOReferrals.csv', 'DataDescription.csv']
        
        for filename in critical_files:
            if filename in expected_checksums:
                filepath = self.data_dir / filename
                if filepath.exists():
                    print(f"Validating {filename}...", end=' ')
                    actual_hash = self.compute_sha256(filepath)
                    expected_hash = expected_checksums[filename]
                    
                    if actual_hash == expected_hash:
                        self.print_success(f"{filename} integrity verified")
                    else:
                        self.print_error(f"{filename} checksum mismatch!")
                        self.print_info(f"Expected: {expected_hash}")
                        self.print_info(f"Actual:   {actual_hash}")
                        
    def compute_sha256(self, filepath):
        """Compute SHA256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
        
    def validate_referral_schema(self):
        """Validate OPOReferrals.csv schema and critical fields"""
        self.print_header("PHASE 3: REFERRAL DATA SCHEMA VALIDATION")
        
        referral_file = self.data_dir / 'OPOReferrals.csv'
        if not referral_file.exists():
            self.print_error("OPOReferrals.csv not found - cannot proceed")
            return None
            
        print("Loading OPOReferrals.csv (this may take a moment)...")
        try:
            df = pd.read_csv(referral_file, low_memory=False)
            self.print_success(f"Loaded {len(df):,} referral records")
            self.results['schema_checks']['total_referrals'] = len(df)
            
            # Check critical columns for Loss Waterfall analysis
            critical_columns = {
                'referral_id': 'Unique referral identifier',
                'opo_id': 'OPO identifier',
                'referral_date': 'Date of referral',
                'referral_outcome': 'Final outcome (donated, not authorized, etc.)',
                'age': 'Donor age',
                'cause_of_death': 'Cause of death',
                'authorization_status': 'Family authorization status',
                'organs_recovered': 'Number of organs recovered',
                'organs_transplanted': 'Number of organs transplanted'
            }
            
            print("\nChecking critical columns:")
            available_cols = df.columns.tolist()
            self.print_info(f"Total columns in dataset: {len(available_cols)}")
            
            # Print first 20 actual column names to help identify what's available
            print("\nFirst 20 columns in dataset:")
            for i, col in enumerate(available_cols[:20], 1):
                self.print_info(f"  {i}. {col}")
            
            if len(available_cols) > 20:
                self.print_info(f"  ... and {len(available_cols) - 20} more columns")
            
            # Check for critical columns (flexible matching)
            print("\nSearching for critical variables:")
            found_columns = {}
            for critical_col, description in critical_columns.items():
                # Try exact match first
                if critical_col in available_cols:
                    self.print_success(f"{critical_col} - {description}")
                    found_columns[critical_col] = critical_col
                else:
                    # Try partial match
                    matches = [col for col in available_cols if critical_col.lower() in col.lower()]
                    if matches:
                        self.print_warning(f"{critical_col} not found, but found similar: {matches[0]}")
                        found_columns[critical_col] = matches[0]
                    else:
                        self.print_warning(f"{critical_col} not found - {description}")
                        
            self.results['schema_checks']['critical_columns'] = found_columns
            
            return df
            
        except Exception as e:
            self.print_error(f"Failed to load OPOReferrals.csv: {str(e)}")
            return None
            
    def validate_temporal_coverage(self, df):
        """Validate temporal coverage (2015-2021)"""
        self.print_header("PHASE 4: TEMPORAL COVERAGE")
        
        if df is None:
            self.print_error("No referral data available")
            return
            
        # Find date columns
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        
        if not date_columns:
            self.print_warning("No date columns found")
            return
            
        self.print_info(f"Found {len(date_columns)} date columns: {', '.join(date_columns[:5])}")
        
        # Try to parse the first date column
        date_col = date_columns[0]
        print(f"\nAnalyzing temporal coverage using '{date_col}':")
        
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            valid_dates = df[date_col].dropna()
            
            if len(valid_dates) > 0:
                min_date = valid_dates.min()
                max_date = valid_dates.max()
                
                self.print_success(f"Date range: {min_date.date()} to {max_date.date()}")
                self.print_info(f"Total span: {(max_date - min_date).days} days")
                
                # Check coverage by year
                df['year'] = valid_dates.dt.year
                year_counts = df['year'].value_counts().sort_index()
                
                print("\nReferrals by year:")
                for year, count in year_counts.items():
                    self.print_info(f"  {year}: {count:,} referrals")
                    
                self.results['coverage_checks']['temporal'] = {
                    'min_date': str(min_date.date()),
                    'max_date': str(max_date.date()),
                    'years': year_counts.to_dict()
                }
                
                # Check if we have 2015-2021 coverage
                expected_years = set(range(2015, 2022))
                actual_years = set(year_counts.index)
                missing_years = expected_years - actual_years
                
                if missing_years:
                    self.print_warning(f"Missing data for years: {sorted(missing_years)}")
                else:
                    self.print_success("Complete 2015-2021 coverage confirmed")
                    
            else:
                self.print_warning("No valid dates found")
                
        except Exception as e:
            self.print_warning(f"Could not parse dates: {str(e)}")
            
    def validate_opo_coverage(self, df):
        """Validate OPO coverage (should have 6 OPOs)"""
        self.print_header("PHASE 5: OPO COVERAGE")
        
        if df is None:
            self.print_error("No referral data available")
            return
            
        # Find OPO identifier column
        opo_columns = [col for col in df.columns if 'opo' in col.lower()]
        
        if not opo_columns:
            self.print_warning("No OPO identifier column found")
            return
            
        opo_col = opo_columns[0]
        self.print_info(f"Using OPO column: '{opo_col}'")
        
        opo_counts = df[opo_col].value_counts()
        
        print(f"\nFound {len(opo_counts)} OPOs:")
        for opo_id, count in opo_counts.items():
            pct = 100 * count / len(df)
            self.print_info(f"  OPO {opo_id}: {count:,} referrals ({pct:.1f}%)")
            
        self.results['coverage_checks']['opos'] = opo_counts.to_dict()
        
        if len(opo_counts) == 6:
            self.print_success("All 6 OPOs present in dataset")
        else:
            self.print_warning(f"Expected 6 OPOs, found {len(opo_counts)}")
            
    def validate_pipeline_stages(self, df):
        """Validate that we can identify sorting, authorization, and placement stages"""
        self.print_header("PHASE 6: REFERRAL PIPELINE STAGES")
        
        if df is None:
            self.print_error("No referral data available")
            return
            
        print("Searching for pipeline stage indicators...\n")
        
        # Look for outcome/status columns
        outcome_columns = [col for col in df.columns if any(
            keyword in col.lower() for keyword in 
            ['outcome', 'status', 'result', 'disposition', 'authorization', 'consent']
        )]
        
        if outcome_columns:
            self.print_success(f"Found {len(outcome_columns)} outcome/status columns:")
            for col in outcome_columns[:10]:
                self.print_info(f"  - {col}")
                
            # Analyze the first outcome column
            outcome_col = outcome_columns[0]
            print(f"\nAnalyzing '{outcome_col}' distribution:")
            
            outcome_counts = df[outcome_col].value_counts()
            for outcome, count in outcome_counts.head(10).items():
                pct = 100 * count / len(df)
                self.print_info(f"  {outcome}: {count:,} ({pct:.1f}%)")
                
            self.results['coverage_checks']['pipeline_stages'] = {
                'outcome_column': outcome_col,
                'outcomes': outcome_counts.head(20).to_dict()
            }
        else:
            self.print_warning("No clear outcome/status columns found")
            
        # Look for organ recovery/transplant data
        organ_columns = [col for col in df.columns if 'organ' in col.lower()]
        
        if organ_columns:
            print(f"\nFound {len(organ_columns)} organ-related columns:")
            for col in organ_columns[:10]:
                self.print_info(f"  - {col}")
        else:
            self.print_warning("No organ-related columns found")
            
    def validate_critical_variables(self, df):
        """Validate variables needed for Loss Waterfall Decomposition"""
        self.print_header("PHASE 7: CRITICAL VARIABLES FOR LOSS WATERFALL")
        
        if df is None:
            self.print_error("No referral data available")
            return
            
        print("Variables needed for Loss Waterfall Decomposition:\n")
        
        required_data = {
            'Sorting stage': ['referral screening', 'medical suitability', 'donor evaluation'],
            'Authorization stage': ['family consent', 'authorization', 'next of kin'],
            'Placement stage': ['organ recovery', 'organ placement', 'transplant']
        }
        
        for stage, keywords in required_data.items():
            print(f"{stage}:")
            found = []
            for col in df.columns:
                if any(kw in col.lower() for kw in keywords):
                    found.append(col)
                    
            if found:
                self.print_success(f"Found {len(found)} relevant columns")
                for col in found[:5]:
                    self.print_info(f"  - {col}")
            else:
                self.print_warning(f"No clear columns found for {stage}")
                
            print()
            
    def generate_summary(self):
        """Generate validation summary"""
        self.print_header("VALIDATION SUMMARY")
        
        total_checks = len(self.results['file_checks'])
        files_exist = sum(1 for v in self.results['file_checks'].values() if v.get('exists', False))
        
        print(f"Files checked: {files_exist}/{total_checks} exist")
        print(f"Errors: {len(self.errors)}")
        print(f"Warnings: {len(self.warnings)}")
        
        if self.errors:
            print(f"\n{RED}ERRORS:{RESET}")
            for error in self.errors:
                print(f"  - {error}")
                
        if self.warnings:
            print(f"\n{YELLOW}WARNINGS:{RESET}")
            for warning in self.warnings[:10]:
                print(f"  - {warning}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more warnings")
                
        # Overall status
        print()
        if not self.errors:
            self.print_success("✓ VALIDATION PASSED - Dataset ready for analysis")
            return True
        else:
            self.print_error("✗ VALIDATION FAILED - Critical issues found")
            return False
            
    def save_results(self, output_file='validation_results.json'):
        """Save validation results to JSON"""
        output_path = self.data_dir / output_file
        
        self.results['summary'] = {
            'timestamp': datetime.now().isoformat(),
            'total_errors': len(self.errors),
            'total_warnings': len(self.warnings),
            'errors': self.errors,
            'warnings': self.warnings[:20]  # Limit warnings in output
        }
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
            
        print(f"\nValidation results saved to: {output_path}")
        
    def run_full_validation(self):
        """Run all validation checks"""
        print(f"\n{BLUE}{'='*80}{RESET}")
        print(f"{BLUE}ORCHID DATASET VALIDATION{RESET}")
        print(f"{BLUE}Data Directory: {self.data_dir}{RESET}")
        print(f"{BLUE}{'='*80}{RESET}")
        
        # Phase 1: File existence
        self.validate_file_existence()
        
        # Phase 2: Checksums (critical files only)
        self.validate_checksums()
        
        # Phase 3-7: Data validation
        df = self.validate_referral_schema()
        if df is not None:
            self.validate_temporal_coverage(df)
            self.validate_opo_coverage(df)
            self.validate_pipeline_stages(df)
            self.validate_critical_variables(df)
        
        # Summary
        success = self.generate_summary()
        
        # Save results
        self.save_results()
        
        return success

def main():
    # Default data directory
    data_dir = Path.home() / 'physionet.org' / 'files' / 'orchid' / '2.1.1'
    
    # Allow override from command line
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
        
    if not data_dir.exists():
        print(f"{RED}Error: Data directory not found: {data_dir}{RESET}")
        print(f"\nUsage: python {sys.argv[0]} [data_directory]")
        sys.exit(1)
        
    # Run validation
    validator = ORCHIDValidator(data_dir)
    success = validator.run_full_validation()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
