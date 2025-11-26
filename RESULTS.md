# Key Findings: Organ Donation Equilibrium Analysis

**Dataset:** ORCHID v1.0.0 (133,101 referrals, 2015-2021)  
**Analysis Date:** November 25, 2025  
**Author:** Noah Parrish

---

## Executive Summary

The organ shortage is primarily a **coordination failure at the sorting stage**, not a problem of biological scarcity, family refusal, or inadequate technology. Using Shapley value decomposition on the ORCHID dataset, we find that:

- **58.6% of welfare loss** occurs at sorting (before families are approached)
- **27.7% of welfare loss** occurs at authorization (family consent)
- **13.7% of welfare loss** occurs at procurement (surgical recovery)

Yet **95.8% of private investment (2015-2021)** targeted procurement, addressing only 13.7% of the problem. This represents a **7.0× misalignment** between investment and welfare impact.

---

## Complete Results Table

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Total Referrals** | 133,101 | All deceased donor referrals (2015-2021) |
| **Medically Suitable** | 60,668 | Age < 70, BMI 15-45, known cause |
| **Approached** | 19,551 | 32.2% of suitable referrals |
| **Authorized** | 11,989 | 61.3% of approached |
| **Procured** | 9,502 | 79.3% of authorized |
| **Total Loss** | 51,166 | Suitable referrals not procured |
| **Sorting Loss (Shapley)** | 29,983 (58.6%) | Before approach |
| **Authorization Loss (Shapley)** | 14,173 (27.7%) | Family declined |
| **Procurement Loss (Shapley)** | 7,010 (13.7%) | Surgical/medical failure |

---

## Finding 1: Sorting Dominates Welfare Loss

### Shapley Value Decomposition

| Stage | Shapley Value | % of Total Loss | Lost Donors |
|-------|---------------|-----------------|-------------|
| **Sorting** | **0.586** | **58.6%** | **29,983** |
| Authorization | 0.277 | 27.7% | 14,173 |
| Procurement | 0.137 | 13.7% | 7,010 |
| **Total** | **1.000** | **100.0%** | **51,166** |

### Proportional Attribution

| Stage | Proportional Loss | % of Total Loss | Lost Donors |
|-------|-------------------|-----------------|-------------|
| **Sorting** | **0.804** | **80.4%** | **41,117** |
| Authorization | 0.148 | 14.8% | 7,562 |
| Procurement | 0.049 | 4.9% | 2,487 |
| **Total** | **1.000** | **100.0%** | **51,166** |

**Interpretation:** Even accounting for stage interactions (Shapley method), sorting accounts for the majority of welfare loss. The field's focus on downstream optimization (perfusion, allocation, surgical techniques) addresses only 5-14% of the problem.

---

## Finding 2: Medical Suitability Is Not the Constraint

Of 60,668 medically suitable referrals, only 19,551 (32.2%) were approached. **67.8% of suitable referrals** were never approached for authorization.

**Medical Suitability Criteria (Conservative):**
- Age < 70 years
- BMI 15-45 kg/m²
- Known cause of death
- No absolute contraindications

---

## Finding 3: Age Bias Is the Dominant Factor

**Multivariate regression (controlling for all factors):**
- Age coefficient: -0.0424 (p < 0.0001)
- **Odds Ratio: 0.96 per year**
- A 60-year-old has 0.96^60 = **0.09× the odds** of a newborn (91% less likely)

**Age effect by decade:**
- 0-17 years: 40.0% approach rate
- 80-89 years: 0.9% approach rate
- **Disparity: 42.2×**

---

## Finding 4: OPO Performance Varies 2.28×

| OPO | Approach Rate |
|-----|---------------|
| Best (OPO 5) | 21.2% |
| Worst (OPO 2) | 9.3% |
| **Variance** | **2.28×** |

---

## Finding 5: Procurement Is Efficient (79.3%)

Only 2,487 losses at procurement (4.9% of total loss, or 13.7% by Shapley).

---

## Finding 6: Investment Is Misaligned 7.0×

| Stage | Investment | % | Shapley Loss | Ratio |
|-------|------------|---|--------------|-------|
| **Procurement** | **$294M** | **95.8%** | **13.7%** | **7.0×** |
| Sorting | $13M | 4.2% | 58.6% | 0.07× |
| Authorization | $0M | 0.0% | 27.7% | 0.0× |

---

## Policy Implications

1. **Rebalance investments** from procurement (5-14% of problem) to sorting (59-80%)
2. **Address age discrimination** (60-year-olds are 91% less likely to be approached)
3. **Expand OPO capacity** (2.28× variance shows organizational capacity matters)
4. **Deploy decision support tools** to reduce age bias and improve coordination

---

**For detailed methodology, see:** [README.md](README.md)  
**For code and replication:** [scripts/](scripts/)
