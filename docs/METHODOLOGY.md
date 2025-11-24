# Methodology: MSC Identification and Loss Waterfall Decomposition

## Overview

This document provides a detailed technical explanation of the methodology used to identify Medically Suitable Candidates (MSCs) and decompose organ losses across the donation pipeline.

---

## The Survivor Bias Problem

### Why We Can't Just Learn from Successful Cases

A naive approach to identifying viable donors would be:

1. Look at all successfully transplanted cases
2. Extract their characteristics (age, BMI, cause of death, etc.)
3. Define "viable" as anyone matching these characteristics

**The Problem**: This creates a **tautology**. If the system systematically rejects 70-year-old donors due to risk aversion (not medical unsuitability), then:
- No 70-year-olds will appear in the "successful" dataset
- The model learns "70-year-olds are not viable"
- When applied to the full dataset, it labels all 70-year-olds as "not MSC"
- **Result**: We codify the very inefficiency we're trying to measure

This is **survivor bias**: we only observe the outcomes of decisions the system made, not the counterfactual outcomes of decisions it didn't make.

---

## Our Solution: Three-Layer Hybrid Approach

We combine **clinical guidelines** (external ground truth) with **empirical data** (reflects actual practice) to avoid survivor bias while remaining grounded in reality.

### Layer 1: Absolute Contraindications (Clinical Guidelines)

**Purpose**: Remove referrals that are medically impossible to donate, regardless of system behavior.

**Sources**:
- OPTN/UNOS Policies (Deceased Donor Organ Procurement)
- ASTS Guidelines for Controlled DCD (2009)
- CDC contraindication lists

**Criteria**:
1. **Active malignancy** (most cancers, except some skin/brain)
2. **HIV+** (with temporal adjustment for HOPE Act, effective 2015)
3. **Rabies**
4. **Creutzfeldt-Jakob disease** (prion disease)
5. **Active sepsis** (uncontrolled systemic infection)
6. **Untreated tuberculosis**

**Implementation**:
```python
def check_absolute_contraindications(row):
    """
    Returns True if contraindicated (NOT suitable)
    """
    contraindications = [
        'Malignant Neoplasm', 'Cancer', 'Carcinoma',
        'HIV', 'AIDS', 'Rabies', 'Creutzfeldt-Jakob', 'CJD',
        'Active Tuberculosis', 'TB'
    ]
    
    cod = str(row.get('cause_of_death_unos', '')).upper()
    for c in contraindications:
        if c.upper() in cod:
            return True
    
    return False
```

**Key Point**: These rules are **time-invariant** (mostly) and **biologically grounded**, not learned from system behavior.

---

### Layer 2: Donation Type Pathways (DBD vs DCD)

**Purpose**: Separate criteria based on donation mechanism, as age limits differ significantly.

**Brain Death Donation (DBD)**:
- Patient is declared brain dead while on mechanical ventilation
- Organs remain perfused with oxygenated blood until procurement
- **No warm ischemia** → more permissive age limits

**Donation after Circulatory Death (DCD)**:
- Patient is not brain dead; life support is withdrawn
- Organs experience **warm ischemia** (time between cardiac arrest and cold perfusion)
- Older organs tolerate warm ischemia poorly → stricter age limits

**Age Ceilings by Organ and Donation Type**:

| Organ     | DBD Max Age | DCD Max Age | Notes                                    |
|-----------|-------------|-------------|------------------------------------------|
| Liver     | 100         | 70          | DBD: No effective ceiling (user insight) |
| Kidney    | 90          | 75          | DCD ceiling expanding over time          |
| Heart     | 70          | 55          | DCD heart newest (started ~2019)         |
| Lung      | 75          | 60          | Moderate DCD restriction                 |
| Pancreas  | 60          | 50          | Strict due to endocrine function         |
| Intestine | 65          | 55          | Rare, conservative                       |

**Implementation**:
```python
def get_donation_type(row):
    """
    Determine DBD vs DCD potential
    """
    if row['brain_death'] == True:
        return 'dbd'
    else:
        return 'dcd'  # Conservative: treat all non-brain-death as DCD potential
```

**Biological Plausibility Bounds**:
- Age: [0, 100] (prevent absurd values)
- BMI: [10, 60] (below 10 incompatible with life, above 60 prohibitive surgical risk)

---

### Layer 3: Empirical Ranges (Sensitivity Analysis)

**Purpose**: Define age/BMI ranges for viability using empirical data, but in a way that avoids codifying system inefficiency.

We implement **three approaches** to test robustness:

#### **Approach A: Absolute Maximum** (Most Liberal)

**Logic**: "If it has been done successfully once, it is an MSC"

**Method**:
```python
age_max = successful_cases['age'].max()
age_max = min(age_max, BIOLOGICAL_AGE_CEILING[organ][donation_type])
```

**Rationale**:
- If one OPO successfully transplanted an 80-year-old liver, then all 80-year-old livers are theoretically viable
- Captures the **tail events** that prove biological feasibility
- Maximizes the denominator (MSC count)

**Risk**: May treat exceptional outliers as standard practice

**Use Case**: Upper bound on sorting loss

---

#### **Approach B: 99th Percentile** (Moderate - PRIMARY)

**Logic**: "If the top 1% can do it, it's viable"

**Method**:
```python
age_99 = np.percentile(successful_cases['age'], 99)
age_99 = min(age_99, BIOLOGICAL_AGE_CEILING[organ][donation_type])
```

**Rationale**:
- Represents **best practice**, not exceptional circumstances
- Avoids being dominated by single outliers
- Still captures tail events (top 1% = ~100 cases for kidneys)

**Risk**: Balanced approach

**Use Case**: Primary analysis (recommended)

---

#### **Approach C: Best-Performing OPO** (Benchmark)

**Logic**: "If the best OPO can do it, others should too"

**Method**:
1. Identify OPO with highest overall conversion rate
2. Extract age ranges from that OPO's successful cases (95th percentile)
3. Apply those ranges to all OPOs

**Rationale**:
- Measures **underperformance relative to achievable standard**
- Policy-relevant: "This is what's possible in practice"
- Accounts for system-level constraints (not just biological limits)

**Risk**: Assumes best OPO is not itself inefficient

**Use Case**: Benchmark for policy recommendations

---

### Combining the Three Layers

**Algorithm**:

```python
def check_msc_status(row, approach):
    # Layer 1: Absolute contraindications
    if check_absolute_contraindications(row):
        return {'is_msc': False, 'reason': 'contraindicated'}
    
    # Layer 2: Donation type
    donation_type = get_donation_type(row)
    
    # Layer 3: Empirical ranges (approach-specific)
    ranges = get_empirical_ranges(approach, donation_type)
    
    viable_organs = []
    for organ_family in ['heart', 'liver', 'kidney', 'lung', 'pancreas', 'intestine']:
        organ_ranges = ranges.get(f'{organ_family}_{donation_type}')
        
        if organ_ranges is None:
            continue
        
        # Check age
        if pd.isna(row['age']):
            continue
        if not (organ_ranges['age']['min'] <= row['age'] <= organ_ranges['age']['max']):
            continue
        
        # If we get here, this organ is viable
        viable_organs.append(organ_family)
    
    # MSC if viable for ANY organ (union rule)
    is_msc = len(viable_organs) > 0
    
    return {
        'is_msc': is_msc,
        'viable_organs': viable_organs,
        'donation_type': donation_type
    }
```

**Union Rule**: A referral is an MSC if viable for **at least one organ**. This prevents OPOs from saying "the liver was bad so we walked away" when the lungs were viable.

---

## Loss Waterfall Decomposition

### Traditional Approach (WRONG)

Traditional loss decomposition simply counts losses at each stage:

```
Sorting Loss = MSCs not approached
Authorization Loss = Approached not authorized
Placement Loss = Procured not transplanted
```

**Problem**: This treats all losses equally, ignoring their position in the sequential process.

---

### Counterfactual Value Method (CORRECT)

We ask: **"How many transplants would we have gained if we fixed this stage?"**

**Formula**:

For each stage $s$:

$$\text{Counterfactual Loss}_s = N_{\text{lost at } s} \times P(\text{success} \mid \text{pass stage } s)$$

Where:
- $N_{\text{lost at } s}$ = Number of candidates lost at stage $s$
- $P(\text{success} \mid \text{pass stage } s)$ = **Downstream success rate** conditional on passing stage $s$

**Downstream Success Rates**:

$$DS_{\text{sorting}} = \frac{N_{\text{transplanted}}}{N_{\text{approached}}}$$

$$DS_{\text{authorization}} = \frac{N_{\text{transplanted}}}{N_{\text{authorized}}}$$

$$DS_{\text{procurement}} = \frac{N_{\text{transplanted}}}{N_{\text{procured}}}$$

**Example Calculation**:

Suppose:
- 40,000 MSCs
- 8,000 approached (20%)
- 5,000 authorized (62.5% of approached)
- 4,000 procured (80% of authorized)
- 3,800 transplanted (95% of procured)

**Downstream success rates**:
- $DS_{\text{sorting}} = 3,800 / 8,000 = 47.5\%$
- $DS_{\text{authorization}} = 3,800 / 5,000 = 76\%$
- $DS_{\text{procurement}} = 3,800 / 4,000 = 95\%$

**Counterfactual losses**:
- **Sorting Loss** = $(40,000 - 8,000) \times 0.475 = 15,200$ transplants
- **Authorization Loss** = $(8,000 - 5,000) \times 0.76 = 2,280$ transplants
- **Procurement Loss** = $(5,000 - 4,000) \times 0.95 = 950$ transplants
- **Placement Loss** = $4,000 - 3,800 = 200$ transplants (actual, not counterfactual)

**Total Loss** = $40,000 - 3,800 = 36,200$ transplants

**Loss Decomposition (%)**:
- Sorting: $15,200 / 36,200 = 42.0\%$
- Authorization: $2,280 / 36,200 = 6.3\%$
- Procurement: $950 / 36,200 = 2.6\%$
- Placement: $200 / 36,200 = 0.6\%$

**Wait, that only adds to 51.5%?**

No! The remaining 48.5% is the **interaction effect** between stages. This is why we use counterfactual values: fixing one stage doesn't give you the full downstream success rate because other stages also have losses.

**Correct Interpretation**: 
- If we **only** fixed sorting (approached all MSCs), we'd gain 15,200 transplants
- If we **only** fixed authorization (all approached authorized), we'd gain 2,280 transplants
- Fixing **both** would gain more than the sum (due to multiplicative effects)

---

## Robustness Analysis

### Layer 1: Stratified Analysis

Break down by:
- **Year** (2015-2021): Check temporal stability
- **OPO** (1-6): Measure variance (coordination failure signature)
- **Donation Type** (DBD vs DCD): Separate pathways
- **Age Groups** (<18, 18-40, 41-60, 61+): Detect age bias
- **Cause of Death**: Check if COD affects sorting

**Implementation**:
```python
for year in range(2015, 2022):
    year_df = df[df['year'] == year]
    results[year] = calculate_loss_waterfall(year_df)
```

---

### Layer 2: Bootstrap Confidence Intervals

Resample dataset 1,000 times to get 95% CIs on:
- MSC count
- Sorting Loss %
- Authorization Loss %
- Overall conversion rate

**Implementation**:
```python
from sklearn.utils import resample

bootstrap_results = []
for i in range(1000):
    boot_df = resample(df, replace=True, n_samples=len(df))
    results = calculate_loss_waterfall(boot_df)
    bootstrap_results.append(results)

ci_low = np.percentile(bootstrap_results, 2.5, axis=0)
ci_high = np.percentile(bootstrap_results, 97.5, axis=0)
```

---

### Layer 3: Cross-Validation

Split data into train/test:
- **Train (70%)**: Build MSC profiles
- **Test (30%)**: Apply profiles and calculate Loss Waterfall

Check if results are stable across splits.

---

### Layer 4: Sensitivity to Exclusions

Test robustness to:
- Excluding OPO2 (shorter time period)
- Excluding 2015 (incomplete year for some OPOs)
- Excluding rare organs (intestine, pancreas)
- Excluding DCD (focus on DBD only)

---

## Validation Checks

### Check 1: Are Approached Donors Mostly MSCs?

**Expected**: >85% of approached donors should be MSCs

**Interpretation**:
- If <70%: MSC criteria too liberal (false positives)
- If >95%: MSC criteria too conservative (false negatives)

---

### Check 2: Do MSCs Have Higher Authorization Rates?

**Expected**: MSC authorization rate > non-MSC authorization rate

**Rationale**: Families are more likely to authorize when the patient is a good candidate

---

### Check 3: Do MSCs Have Higher Transplant Rates?

**Expected**: MSC transplant rate (among procured) > non-MSC transplant rate

**Rationale**: Organs from MSCs should be more likely to be accepted by transplant centers

---

## National Benchmark Comparison

**National Estimate**: 35,000-42,000 potential donors annually (from literature)

**Scaling for Dataset**:
```
Expected MSC = (35,000 to 42,000) × (6 OPOs / 58 total OPOs) × 7 years
             = 25,345 to 30,414 MSCs
```

**Interpretation**:
- If actual MSC count is within this range: Well-calibrated
- If below: MSC criteria too strict
- If above: MSC criteria too liberal OR national estimates are outdated

**Important**: Don't force MSC count to match national estimates. Use as validation check only.

---

## Software Implementation

All scripts are in `scripts/`:

1. `01_data_validation.py` - Pre-check dataset
2. `02_data_exploration.py` - EDA
3. `03_msc_sensitivity.py` - MSC identification (3 approaches)
4. `04_loss_waterfall.py` - Loss Waterfall decomposition
5. `05_robustness_analysis.py` - Bootstrap, stratification, CV
6. `06_shapley_decomposition.py` - Shapley values
7. `07_visualization.py` - Generate figures

---

## References

1. OPTN/UNOS Policies: https://optn.transplant.hrsa.gov/policies-bylaws/policies/
2. ASTS DCD Guidelines: Reich et al. (2009), *American Journal of Transplantation*
3. ORCHID Dataset: Adam et al. (2025), PhysioNet
