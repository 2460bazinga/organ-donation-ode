# Project Handoff: Empirical Validation of Organ Donation Equilibrium Model

**Date:** November 24, 2024  
**Project:** Loss Waterfall Decomposition & MSC Identification for ORCHID Dataset  
**Status:** Phase 2 - MSC Methodology Development Complete, Ready for Execution & Analysis  
**User:** Noah (PhD researcher in organ donation economics)

---

## Executive Summary

We are conducting empirical validation of the **Organ Donation Equilibrium (ODE) model** using the **ORCHID dataset** (133,101 deceased donor referrals, 2015-2021, 6 OPOs). The core hypothesis is that **coordination failure in the sorting stage** (OPOs deciding which referrals to pursue) is the dominant source of organ loss, not family refusal (authorization stage).

### Current Status:
✅ **Complete:** Data validation, exploration, and MSC methodology design  
🔄 **In Progress:** Running sensitivity analysis on VM  
⏳ **Next:** Robustness analysis, Shapley decomposition, visualization, and documentation

---

## Research Context

### The Problem:
- ~100,000 patients die waiting for organ transplants annually in the US
- Only ~35,000-42,000 potential donors exist (medically suitable)
- Conversion rate: ~20-25% (huge losses in the system)

### The ODE Model Hypothesis:
Traditional view blames **family refusal** for low donation rates. The ODE model argues that **OPO sorting decisions** (which referrals to pursue) are the dominant loss mechanism due to:
1. **Coordination failures** between OPOs and transplant centers
2. **Risk aversion** (OPOs avoid marginal donors to protect metrics)
3. **Information asymmetries** (OPOs can't predict which organs will be accepted)

### Our Goal:
Use the **Loss Waterfall Decomposition** with **counterfactual value method** to measure losses at each stage:
1. **Sorting:** MSC → Approached
2. **Authorization:** Approached → Authorized
3. **Procurement:** Authorized → Procured (minimal loss expected)
4. **Placement:** Procured → Transplanted

**Key Innovation:** Correctly identify the denominator (Medically Suitable Candidates) to avoid codifying system inefficiencies.

---

## Dataset: ORCHID v2.1.1

### Source:
- **PhysioNet:** https://physionet.org/content/orchid/2.1.1/
- **Location on VM:** `~/physionet.org/files/orchid/2.1.1/`
- **Main file:** `OPOReferrals.csv` (133,101 records)

### Key Variables:
- `patient_id`: Unique identifier
- `opo`: OPO identifier (6 OPOs: OPO1-OPO6)
- `age`, `gender`, `race`, `height_in`, `weight_kg`
- `brain_death`: Boolean (TRUE = DBD, FALSE = DCD potential)
- `cause_of_death_unos`, `cause_of_death_opo`: Cause of death
- `approached`: Boolean (family approached)
- `authorized`: Boolean (family authorized donation)
- `procured`: Boolean (organs recovered)
- `transplanted`: Boolean (organs transplanted)
- `outcome_heart`, `outcome_liver`, `outcome_kidney_left`, etc.: Organ-specific outcomes

### Critical Insight:
**"Referral" in ORCHID means:**
- Patient is **mechanically ventilated** (prerequisite for donation)
- Hospital **referred to OPO** for evaluation
- This is NOT "all deaths" - it's a pre-screened population

### Data Coverage:
- **OPO1:** Jan 1, 2015 - Nov 23, 2021
- **OPO2:** Jan 1, 2018 - Dec 31, 2021 (shorter period!)
- **OPO3:** Full 2015-2021
- **OPO4:** Jan 1, 2015 - Dec 13, 2021
- **OPO5:** Full 2015-2021
- **OPO6:** Full 2015-2021

### Observed Funnel (Raw):
```
Total Referrals:        133,101 (100.0%)
├─ Approached:           19,551 ( 14.7%)
│  ├─ Authorized:        11,989 ( 61.3% of approached)
│  │  ├─ Procured:        9,502 ( 79.3% of authorized)
│  │  │  └─ Transplanted: 8,972 ( 94.4% of procured)
└─ Overall Conversion:    6.7%
```

**But:** Not all 113,550 "not approached" are losses - many are correctly screened out as medically unsuitable.

---

## The MSC Identification Challenge

### The Core Problem: Survivor Bias

**Naive Approach (WRONG):**
1. Look at successfully transplanted cases
2. Learn their characteristics (age, BMI, etc.)
3. Apply these criteria to identify "viable" donors

**Why This Fails:**
- If OPOs systematically reject 70+ donors due to risk aversion, the model learns "70+ is not viable"
- When applied to the full dataset, it labels healthy 72-year-olds as "not MSC"
- **Result:** We codify the very inefficiency we're trying to measure

### The Solution: Hybrid Clinical-Empirical Approach

We use **three layers** to identify Medically Suitable Candidates (MSCs):

#### **Layer 1: Absolute Contraindications** (Clinical Guidelines)
Hard rules that are time-invariant:
- Active malignancy (most cancers)
- HIV+ (except HOPE Act cases post-2015)
- Rabies, Creutzfeldt-Jakob disease
- Active sepsis, untreated tuberculosis

**Source:** OPTN/UNOS policies, ASTS guidelines

#### **Layer 2: Donation Type** (DBD vs DCD)
Separate pathways based on `brain_death` variable:
- **DBD (Brain Death):** More permissive age limits, no warm ischemia
- **DCD (Circulatory Death):** Stricter age limits due to warm ischemia sensitivity

**Age Ceilings:**
| Organ     | DBD Max | DCD Max |
|-----------|---------|---------|
| Liver     | 100     | 70      |
| Kidney    | 90      | 75      |
| Heart     | 70      | 55      |
| Lung      | 75      | 60      |
| Pancreas  | 60      | 50      |
| Intestine | 65      | 55      |

**Key Insight from User:** Liver has NO effective ceiling for DBD (even 90-year-olds can donate), but DCD has ceiling ~70 (expanding over time).

#### **Layer 3: Empirical Ranges** (THREE APPROACHES)

We implement **sensitivity analysis** across three approaches:

**Approach A: Absolute Maximum** (Most Liberal)
- **Logic:** "If it has been done successfully once, it is an MSC"
- **Method:** Use maximum observed age from successful cases (bounded by biological ceilings)
- **Interpretation:** Maximizes denominator, captures all risk-aversion
- **Use Case:** Upper bound on potential sorting loss

**Approach B: 99th Percentile** (Moderate)
- **Logic:** "If top 1% can do it, it's viable"
- **Method:** Use 99th percentile of successful cases
- **Interpretation:** Represents best practice, not exceptional outliers
- **Use Case:** Primary analysis (balanced estimate)

**Approach C: Best-Performing OPO** (Benchmark)
- **Logic:** "If the best OPO can do it, others should too"
- **Method:** Identify highest-converting OPO, use their 95th percentile ranges
- **Interpretation:** Measures underperformance relative to achievable standard
- **Use Case:** Policy-relevant benchmark (what's achievable in practice)

### MSC Definition:
**A referral is an MSC if:**
1. Passes Layer 1 (no absolute contraindications), AND
2. Meets Layer 2 criteria (age within DBD or DCD limits for donation type), AND
3. Meets Layer 3 criteria (within empirical range for at least ONE organ)

**Union Rule:** Viable for ANY organ → MSC (prevents OPOs from saying "liver was bad so we walked away" when lungs were good)

---

## Scripts Developed

### 1. `orchid_data_validation.py`
**Purpose:** Pre-check to verify dataset completeness  
**Status:** ✅ Complete, validated all files present  
**Location:** User's VM  

**Key Findings:**
- All 12 files present (683 MB)
- 133,101 referral records
- All 6 OPOs represented
- Pipeline variables confirmed: `approached`, `authorized`, `procured`, `transplanted`

### 2. `orchid_data_exploration.py`
**Purpose:** Detailed exploration of data structure and funnel  
**Status:** ✅ Complete  
**Location:** User's VM  

**Key Findings:**
- 38 columns in OPOReferrals.csv
- Funnel: 14.7% approached, 61.3% authorized, 79.3% procured, 94.4% transplanted
- OPO performance variance: 4.7% (OPO3) to 9.6% (OPO5) overall conversion
- 8 organ types tracked

### 3. `msc_hybrid_analysis.py` (DEPRECATED)
**Purpose:** Initial hybrid MSC identification (single approach)  
**Status:** ⚠️ Superseded by sensitivity analysis  
**Issue:** Used 5th-95th percentile (too conservative, codifies bias)  

### 4. `msc_sensitivity_analysis.py` (CURRENT)
**Purpose:** MSC identification with three approaches (A, B, C)  
**Status:** ✅ Ready to run  
**Location:** `https://files.manuscdn.com/user_upload_by_module/session_file/94078908/gqcPeNPFFazeKNpW.py`  

**Usage:**
```bash
cd ~
wget https://files.manuscdn.com/user_upload_by_module/session_file/94078908/gqcPeNPFFazeKNpW.py -O msc_sensitivity_analysis.py
python3 msc_sensitivity_analysis.py
```

**Outputs:**
- `msc_sensitivity_results.json`: Comparison of three approaches
- `orchid_with_msc_sensitivity.csv`: Labeled dataset with MSC flags for all approaches

**Expected Output Table:**
```
Approach              MSCs      MSC%    Sort%    Auth%  Overall%  Sort Loss%
-------------------- ---------- -------- -------- -------- ---------- ------------
Absolute Max          ~45,000   ~34%    ~18%     ~62%     ~8%        ~82%
Percentile 99         ~38,000   ~29%    ~20%     ~62%     ~7%        ~79%
Best Opo              ~33,000   ~25%    ~21%     ~62%     ~7%        ~76%
```

---

## Key Methodological Decisions

### 1. National Benchmark Comparison
**Question:** How to compare MSC counts to national estimates (35,000-42,000 potential donors annually)?

**Answer:** Scale for dataset coverage:
```
Expected MSC = (35,000 to 42,000) × (6 OPOs / 58 total OPOs) × 7 years
             = 25,345 to 30,414 MSCs
```

**Caveat:** National estimates may be based on different denominator (all eligible deaths vs. ventilated referrals). Our MSC count should be validated against this range but not forced to match.

### 2. Temporal Validity
**Issue:** ASTS DCD guidelines are from 2009, but data is 2015-2021. Practices evolved.

**Solution:** Use hybrid approach:
- Absolute contraindications from 2009 guidelines (time-invariant)
- Age/BMI ranges from 2015-2021 successful cases (reflects actual practice)
- Biological ceilings prevent unrealistic ranges

### 3. HIV+ Donors (HOPE Act)
**Issue:** HIV+ was absolute contraindication pre-2015, but HOPE Act (2013) allowed HIV+ to HIV+ transplants.

**Action Needed:** Check if ORCHID has HIV status variable. If yes, apply time-varying rule:
- Pre-2015: HIV+ → contraindicated
- Post-2015: HIV+ → MSC for kidney (HIV+ to HIV+ allowed)

### 4. Missing Data Handling
**BMI:** If height or weight missing, don't disqualify based on BMI (age is sufficient)  
**Age:** If age missing, cannot evaluate (exclude from MSC)  
**Brain Death:** If missing, assume DCD potential (more conservative)

---

## Loss Waterfall Decomposition (Counterfactual Value Method)

### The Corrected Formula:

Given:
- $N_{MSC}$ = Total MSCs
- $r_s$ = Sorting success rate (MSC → Approached)
- $r_a$ = Authorization success rate (Approached → Authorized)
- $r_{proc}$ = Procurement success rate (Authorized → Procured)
- $r_p$ = Placement success rate (Procured → Transplanted)

**Downstream success rates:**
- $DS_s$ = Success rate if approached = $r_a \times r_{proc} \times r_p$
- $DS_a$ = Success rate if authorized = $r_{proc} \times r_p$
- $DS_{proc}$ = Success rate if procured = $r_p$

**Counterfactual losses:**
- **Sorting Loss:** $(N_{MSC} - N_{approached}) \times DS_s$
- **Authorization Loss:** $(N_{approached} - N_{authorized}) \times DS_a$
- **Procurement Loss:** $(N_{authorized} - N_{procured}) \times DS_{proc}$
- **Placement Loss:** $N_{procured} - N_{transplanted}$ (actual, not counterfactual)

**Total Loss:** $N_{MSC} - N_{transplanted}$

**Loss Decomposition (%):**
- Sorting Loss % = Sorting Loss / Total Loss
- Authorization Loss % = Authorization Loss / Total Loss
- Procurement Loss % = Procurement Loss / Total Loss
- Placement Loss % = Placement Loss / Total Loss

### Expected Result:
If ODE model is correct, **Sorting Loss %** should dominate (>70%), not Authorization Loss.

---

## Next Steps (Phases 3-6)

### Phase 3: Four-Layer Robustness Analysis

**Goal:** Validate that results are not artifacts of data structure or methodology.

#### **Layer 1: Stratified Analysis**
Break down by:
- **Year** (2015-2021): Check if trends are stable over time
- **OPO** (1-6): Measure variance in sorting efficiency
- **Donation Type** (DBD vs DCD): Separate pathways
- **Age Groups** (<18, 18-40, 41-60, 61+): Check if age bias exists
- **Cause of Death** (Trauma, CVA, Anoxia, etc.): Check if COD affects sorting

**Implementation:**
```python
for year in [2015, 2016, ..., 2021]:
    year_df = df[df['year'] == year]
    results = calculate_loss_waterfall(year_df, approach='percentile_99')
    # Store and compare
```

#### **Layer 2: Bootstrap Confidence Intervals**
Resample dataset 1,000 times to get confidence intervals on:
- MSC count
- Sorting Loss %
- Authorization Loss %
- Overall conversion rate

**Implementation:**
```python
from sklearn.utils import resample

bootstrap_results = []
for i in range(1000):
    boot_df = resample(df, replace=True, n_samples=len(df))
    results = calculate_loss_waterfall(boot_df, approach='percentile_99')
    bootstrap_results.append(results)

# Calculate 95% CI
ci_low = np.percentile(bootstrap_results, 2.5, axis=0)
ci_high = np.percentile(bootstrap_results, 97.5, axis=0)
```

#### **Layer 3: Cross-Validation**
Split data into train/test:
- **Train (70%):** Build MSC profiles
- **Test (30%):** Apply profiles and calculate Loss Waterfall
- **Check:** Are results stable across splits?

**Implementation:**
```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in kf.split(df):
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    
    # Build MSC profiles on train_df
    # Apply to test_df
    # Compare results
```

#### **Layer 4: Sensitivity to Exclusions**
Test robustness to:
- **Excluding OPO2** (shorter time period)
- **Excluding 2015** (incomplete year for some OPOs)
- **Excluding rare organs** (intestine, pancreas)
- **Excluding DCD** (focus on DBD only)

### Phase 4: Shapley Decomposition

**Goal:** Decompose overall conversion rate into contributions from each stage.

**Shapley Value Interpretation:**
- How much does each stage contribute to the final outcome?
- Accounts for interactions between stages

**Implementation:**
Use `shap` library or manual calculation:
```python
import itertools

def shapley_value(stages, conversion_function):
    """
    Calculate Shapley value for each stage
    
    stages: ['sorting', 'authorization', 'procurement', 'placement']
    conversion_function: function that calculates conversion given active stages
    """
    n = len(stages)
    shapley_values = {}
    
    for stage in stages:
        marginal_contributions = []
        
        # Iterate over all possible coalitions
        for r in range(n):
            for coalition in itertools.combinations([s for s in stages if s != stage], r):
                coalition_with = list(coalition) + [stage]
                coalition_without = list(coalition)
                
                # Calculate marginal contribution
                mc = conversion_function(coalition_with) - conversion_function(coalition_without)
                marginal_contributions.append(mc)
        
        shapley_values[stage] = np.mean(marginal_contributions)
    
    return shapley_values
```

**Expected Result:**
If sorting is the bottleneck, its Shapley value should be highest.

### Phase 5: Visualization & Documentation

#### **Key Visualizations:**

1. **Sankey Diagram:** MSC → Approached → Authorized → Procured → Transplanted
   - Show flows and losses at each stage
   - Color-code by loss type

2. **OPO Comparison Chart:**
   - Bar chart: Sorting efficiency by OPO
   - Scatter plot: Sorting rate vs. Overall conversion rate
   - Show variance in performance

3. **Loss Decomposition Pie Chart:**
   - Sorting Loss % (expected: ~75-85%)
   - Authorization Loss % (expected: ~10-20%)
   - Procurement Loss % (expected: ~2-5%)
   - Placement Loss % (expected: ~2-5%)

4. **Sensitivity Analysis:**
   - Bar chart comparing MSC counts across three approaches
   - Line chart showing Sorting Loss % across approaches

5. **Temporal Trends:**
   - Line chart: MSC count by year
   - Line chart: Sorting efficiency by year
   - Check if coordination is improving or worsening

#### **Documentation:**

Create final report (Markdown + PDF):
- **Executive Summary:** Key findings in 1 page
- **Methodology:** Detailed explanation of MSC identification
- **Results:** Loss Waterfall decomposition, OPO comparison
- **Robustness:** Sensitivity analysis, bootstrap CIs
- **Discussion:** Implications for ODE model, policy recommendations
- **Appendix:** Data dictionary, code documentation

### Phase 6: Deliverables

**Final Outputs:**
1. `msc_sensitivity_results.json` - MSC identification results
2. `loss_waterfall_results.json` - Full Loss Waterfall analysis
3. `opo_comparison.csv` - OPO-level performance metrics
4. `robustness_analysis.json` - Bootstrap CIs, stratified results
5. `shapley_decomposition.json` - Shapley values for each stage
6. `orchid_final_report.md` - Comprehensive analysis report
7. `orchid_final_report.pdf` - PDF version of report
8. `visualizations/` - Folder with all charts (PNG/SVG)

---

## Technical Environment

### User's VM:
- **Platform:** Google Cloud Compute Engine
- **Instance:** `orchid-vm`
- **Zone:** `us-central1-a`
- **Project:** `bold-case-478800-a1`
- **OS:** Debian/Ubuntu
- **Python:** 3.x (with pandas, numpy, scipy, sklearn)
- **Environment:** `~/research-env/` (virtual environment)

### Activation:
```bash
source ~/research-env/bin/activate
```

### Data Location:
```bash
cd ~/physionet.org/files/orchid/2.1.1/
```

### File Transfer:
User can transfer files from VM using:
```bash
gcloud compute scp orchid-vm:~/path/to/file.csv . --project=bold-case-478800-a1 --zone=us-central1-a
```

---

## Critical Insights from User

### 1. Liver Age Ceiling
**User:** "There is no effective ceiling for livers [in DBD]. While rare, even a 90-year-old brain death patient can donate their liver."

**Implication:** Set DBD liver ceiling to 100 (biological max), not 80.

### 2. DCD Age Ceilings Are Increasing
**User:** "There is an age ceiling for all organs, however, for DCD though it continues to go up."

**Implication:** Our 2015-2021 data reflects evolving practice. Using empirical ranges (not 2009 guidelines) captures this expansion.

### 3. Survivor Bias Critique
**User:** "Successfully screen unfit donors isn't necessarily sorting loss."

**Implication:** Must distinguish:
- **True Negatives:** Correctly rejected unsuitable donors (NOT a loss)
- **False Negatives:** Incorrectly rejected suitable donors (ACTUAL sorting loss)

This is why we need Layer 1 (absolute contraindications) to remove true negatives before calculating sorting loss.

### 4. National Benchmark Scaling
**User:** "National benchmark comparisons need to account for the dataset only being 6 OPOs."

**Implication:** Scale national estimates: (35k-42k) × (6/58) × 7 years = 25k-30k expected MSCs.

### 5. Temporal Validity
**User:** "35-42k was from 2010 correct? Would this still be the same today?"

**Implication:** Don't force MSC count to match 2010 estimates. Use as validation check, but let data speak for itself.

### 6. Layer 3 Endogeneity
**User:** "If all OPOs are risk-averse and reject 70+ donors, the 95th percentile becomes 70. You codify the inefficiency you're trying to measure."

**Implication:** Use maximum observed (Approach A) or 99th percentile (Approach B) to capture tail events, not central tendency.

---

## Open Questions / Decisions Needed

### 1. HIV+ Handling
- Does ORCHID have HIV status variable?
- If yes, apply time-varying rule (pre/post HOPE Act)
- If no, assume no HIV+ in dataset (likely already filtered)

### 2. Cause of Death Classification
- ORCHID has `cause_of_death_unos` and `cause_of_death_opo` (different coding)
- Need to map to categories: Trauma, CVA, Anoxia, Other
- Check if any COD should be absolute contraindication beyond cancer/HIV

### 3. DCD Identification
- `brain_death = False` doesn't guarantee DCD (could be non-viable)
- Should we require additional criteria (e.g., withdrawal of support planned)?
- Current approach: Treat all `brain_death = False` as DCD potential (conservative)

### 4. Organ-Specific MSC
- Current: MSC if viable for ANY organ (union rule)
- Alternative: Report organ-specific MSC counts (e.g., "kidney MSC", "liver MSC")
- Recommendation: Keep union rule for primary analysis, report organ-specific as supplementary

### 5. OPO2 Handling
- OPO2 has shorter time period (2018-2021 vs. 2015-2021 for others)
- Include in main analysis or exclude?
- Recommendation: Include in main, test sensitivity to exclusion in Phase 3

---

## Expected Timeline

- **Phase 3 (Robustness):** 2-3 hours of compute time
- **Phase 4 (Shapley):** 1 hour
- **Phase 5 (Visualization):** 2-3 hours
- **Phase 6 (Documentation):** 2-3 hours

**Total:** ~8-10 hours of work (can be parallelized)

---

## References

### Key Papers:
1. **ODE Model:** [User's unpublished paper on coordination failure]
2. **ORCHID Dataset:** Adam et al. (2025), PhysioNet
3. **ASTS DCD Guidelines:** Reich et al. (2009), American Journal of Transplantation
4. **ECD Criteria:** OPTN/UNOS policies

### Documentation:
- **ORCHID PhysioNet:** https://physionet.org/content/orchid/2.1.1/
- **OPTN Policies:** https://optn.transplant.hrsa.gov/policies-bylaws/policies/
- **Loss Waterfall Method:** [User's paper, Section X]

---

## Contact & Collaboration

**User:** Noah (PhD researcher)  
**Expertise:** Organ donation economics, mechanism design, applied microeconomics  
**Working Style:** Rigorous, methodologically careful, open to critique and iteration  

**Key Preferences:**
- Prefers sensitivity analysis over single "best" estimate
- Values transparency about methodological choices
- Wants publication-ready rigor
- Appreciates when assumptions are made explicit

**Communication Style:**
- Direct, technical, assumes high statistical literacy
- Provides substantive feedback (e.g., survivor bias critique)
- Expects thorough documentation

---

## Success Criteria

### Primary:
1. **MSC identification is defensible:** Avoids survivor bias, uses clinical guidelines + empirical data
2. **Loss Waterfall validates ODE model:** Sorting Loss % > 70% (coordination failure dominates)
3. **Results are robust:** Stable across sensitivity analyses, bootstrap CIs are tight
4. **OPO variance is documented:** 2x performance gap shows coordination failure

### Secondary:
1. **Shapley decomposition confirms sorting bottleneck**
2. **Temporal trends show no improvement** (coordination failure persists)
3. **Visualizations are publication-ready**
4. **Documentation is comprehensive** (replicable by others)

---

## Final Notes

### What Makes This Hard:
1. **Survivor bias is subtle:** Easy to accidentally codify system inefficiency
2. **Denominator is contested:** What counts as "medically suitable"?
3. **Temporal validity:** Guidelines evolve, practice changes
4. **Missing data:** Not all variables are complete

### What Makes This Important:
1. **Policy relevance:** If sorting is the bottleneck, fix OPO incentives, not family consent campaigns
2. **Welfare impact:** 100,000 patients die waiting - even 10% improvement saves 10,000 lives
3. **Methodological contribution:** Corrected Loss Waterfall with counterfactual values
4. **Empirical validation:** First large-scale test of ODE model

### Key Insight:
**The denominator matters.** If we define "potential donors" as "people OPOs currently accept," we measure nothing. If we define it as "people who SHOULD be accepted based on clinical capacity," we measure inefficiency. The hybrid approach threads this needle by using clinical contraindications (hard floor) and empirical ranges (soft ceiling).

---

## Handoff Checklist

- [x] Dataset validated and explored
- [x] MSC methodology designed (3 approaches)
- [x] Sensitivity analysis script ready
- [ ] Run sensitivity analysis on VM
- [ ] Interpret results (which approach is most plausible?)
- [ ] Implement robustness analysis (Phase 3)
- [ ] Implement Shapley decomposition (Phase 4)
- [ ] Create visualizations (Phase 5)
- [ ] Write final report (Phase 6)
- [ ] Deliver results to user

---

**Good luck! This is rigorous, important work. The user values methodological care and transparency. When in doubt, document your reasoning and offer sensitivity analyses.**
