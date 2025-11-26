# Organ Donation Equilibrium (ODE): An Empirical Analysis of Coordination Failure

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/DOI-10.13026%2Feytj--4f29-blue)](https://doi.org/10.13026/eytj-4f29)

**A game-theoretic framework revealing that the organ shortage is primarily a coordination failure, not a scarcity problem.**

*By Noah Parrish*

---

## Abstract

This paper introduces the **Organ Donation Equilibrium (ODE)**, a game-theoretic framework that reframes the U.S. organ shortage as a multi-stakeholder coordination failure rather than a problem of biological scarcity or family refusal. Using the ORCHID dataset (133,101 deceased donor referrals, 2015-2021), we apply Shapley value decomposition to attribute welfare loss across three stages: sorting (OPO identification and monitoring), authorization (family consent), and procurement (surgical recovery). 

**Key findings:**
- **Sorting accounts for 58.6%** of welfare loss (Shapley decomposition)
- **67.8% of medically suitable referrals** are never approached for authorization
- **Age bias dominates**: 60-year-olds are 91% less likely to be approached than newborns
- **Procurement is efficient**: 79.3% success rate, accounting for only 13.7% of welfare loss
- **Investment misalignment**: $307M invested 2015-2021, with 95.8% targeting procurement (13.7% of problem) vs. 4.2% targeting sorting (58.6% of problem)

The organ shortage is not primarily a problem of inadequate technology, family refusal, or biological scarcity. It is a coordination failure in a fragmented system where viable donors fall through the cracks before families are ever approached.

---

## Table of Contents

- [The Problem](#the-problem)
- [The ODE Model](#the-ode-model)
- [Key Findings](#key-findings)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## The Problem

Over 100,000 Americans wait for organ transplants. Thousands die waiting each year. The conventional narrative attributes this shortage to:
1. **Biological scarcity** - not enough suitable donors
2. **Family refusal** - families declining to authorize donation
3. **Technical limitations** - organs damaged during procurement or preservation

This paper demonstrates that all three narratives are largely incorrect.

### The Real Problem: Coordination Failure

The organ donation system is a twelve-player dynamic game involving hospitals, organ procurement organizations (OPOs), transplant centers, and families. Each player operates under uncertainty, time constraints, and incomplete information about others' actions. The result is a **low-level equilibrium** where:

- Hospitals under-refer marginal cases
- OPOs under-monitor marginal referrals
- Transplant centers under-prepare for marginal organs
- Families are never asked (because donors exit the system upstream)

No single player is failing. The system is failing to coordinate.

---

## The ODE Model

The **Organ Donation Equilibrium** is a game-theoretic framework that models organ donation as a three-stage sequential process:

### Stage 1: Sorting
**Players:** Hospitals, OPOs  
**Action:** Identify, refer, and monitor potential donors  
**Outcome:** 32.2% of medically suitable referrals are approached; 67.8% exit the system

### Stage 2: Authorization
**Players:** OPOs, Families  
**Action:** Request and obtain family consent  
**Outcome:** 61.3% of approached families authorize donation

### Stage 3: Procurement
**Players:** OPOs, Surgeons  
**Action:** Surgically recover organs  
**Outcome:** 79.3% of authorized donors yield procured organs

### The Coordination Failure

The model predicts that **sorting losses dominate** because:
1. **Information asymmetry**: OPOs lack real-time visibility into donor evolution
2. **Capacity constraints**: OPOs cannot monitor all marginal referrals
3. **Strategic complementarity**: Each player's optimal effort depends on others' actions
4. **Lack of infrastructure**: No coordinated monitoring or decision support systems

The system settles into a stable but inefficient equilibrium where viable donors are lost not through active rejection, but through **passive attrition**.

---

## Key Findings

### 1. Sorting Dominates Welfare Loss

**Shapley Value Decomposition:**
- **Sorting: 58.6%** of welfare loss (29,983 lost donors)
- **Authorization: 27.7%** (14,173 lost donors)
- **Procurement: 13.7%** (7,010 lost donors)

**Proportional Attribution:**
- **Sorting: 80.4%** (41,117 lost donors)
- Authorization: 14.8% (7,562 lost donors)
- Procurement: 4.9% (2,487 lost donors)

### 2. Medical Suitability Is Not the Constraint

Of 133,101 total referrals:
- **60,668 were medically suitable** (age < 70, BMI 15-45, known cause of death)
- Only **19,551 were approached** (32.2% approach rate)
- **41,117 suitable referrals** were never approached (67.8%)

**Conclusion:** The bottleneck is not biological scarcity—it's the failure to identify and pursue suitable donors.

### 3. Age Bias Is the Dominant Factor

**Multivariate regression (controlling for all factors):**
- Age coefficient: -0.0424 (p < 0.0001)
- **Odds Ratio: 0.96 per year** (each additional year reduces odds of approach by 4%)
- A 60-year-old has 0.96^60 = **0.09× the odds** of a newborn (91% less likely to be approached)

**Age effect by decade:**
- 0-17 years: 40.0% approach rate
- 18-39 years: 24.5% approach rate
- 40-59 years: 13.8% approach rate
- 60-79 years: 6.2% approach rate
- 80-89 years: 0.9% approach rate

**Disparity: 42.2× between youngest and oldest groups**

### 4. OPO Performance Varies Substantially

**Approach rates across 6 OPOs:**
- Best OPO: 21.2%
- Worst OPO: 9.3%
- **Variance: 2.28×**

**Interpretation:** Organizational capacity and practices matter. The 2.28× variance suggests that some OPOs have developed coordination capabilities that others lack.

### 5. Procurement Is Not the Bottleneck

**Procurement success rate: 79.3%**
- Of 11,989 authorized donors, 9,502 were successfully procured
- Only 2,487 losses at procurement (4.9% of total loss)

**Implication:** High-tech interventions (normothermic perfusion, organ preservation devices, surgical innovations) address less than 5% of the problem.

### 6. Investment Is Severely Misaligned

**Total private investment (2015-2021): $307M**

**By stage:**
- **Procurement: $294M (95.8%)** → addresses 13.7% of welfare loss
- **Sorting: $13M (4.2%)** → addresses 58.6% of welfare loss
- **Authorization: $0M (0.0%)** → addresses 27.7% of welfare loss

**Misalignment ratio: 7.0×** overinvestment in procurement relative to its welfare impact

**Major investments:**
- TransMedics (Organ Care System): $150M
- OrganOx (liver perfusion): $12M (later acquired for $1.5B in 2025)
- Paragonix (organ transport): $5M
- Other organ preservation startups: $80M
- NIH grants (procurement-focused): ~$32M/year
- NIH grants (coordination/sorting): ~$8M/year

---

## Dataset

### ORCHID v1.0.0

**Source:** [PhysioNet](https://doi.org/10.13026/eytj-4f29)

**Citation:** Adam, H., Suriyakumar, V., Pollard, T., Moody, B., Erickson, J., Segal, G., ... & Ghassemi, M. (2023). Organ Retrieval and Collection of Health Information for Donation (ORCHID) (version 1.0.0). *PhysioNet*.

**Description:** The ORCHID dataset is a multi-center, de-identified dataset containing granular information on deceased donor referrals across six U.S. Organ Procurement Organizations.

**Coverage:**
- **133,101 referral records** (2015-2021)
- **6 OPOs** across 13 states
- **9,502 procured donors**
- **Longitudinal clinical data** (labs, vitals, serology)

**Key Variables:**
- Demographics: age, gender, race, height, weight, BMI
- Clinical: brain_death status, cause_of_death, comorbidities, laboratory values
- Process: approached (binary), authorized (binary), procured (binary)
- Temporal: referral_date, approach_date, authorization_date, procurement_date

**Data Access:**

The ORCHID dataset requires credentialed access through PhysioNet:
1. Complete PhysioNet credentialing: https://physionet.org/login/
2. Sign the data use agreement for ORCHID v1.0.0
3. Download the dataset to `data/orchid/`

**Note:** Raw data is not included in this repository due to PhysioNet's data use agreement.

---

## Methodology

### Shapley Value Decomposition

We apply **Shapley value decomposition** to attribute welfare loss across the three stages. The Shapley value is a game-theoretic solution concept that allocates total value (or loss) to players based on their marginal contributions across all possible orderings.

**Formula:**

For each stage $s \in \{\text{sorting}, \text{authorization}, \text{procurement}\}$:

$$\phi_s = \sum_{S \subseteq N \setminus \{s\}} \frac{|S|! \cdot (|N| - |S| - 1)!}{|N|!} \cdot [v(S \cup \{s\}) - v(S)]$$

Where:
- $N$ = set of all stages
- $S$ = subset of stages
- $v(S)$ = value function (total procured donors when stages in $S$ succeed)
- $\phi_s$ = Shapley value for stage $s$

**Interpretation:** The Shapley value accounts for **interactive effects** between stages. It answers: "How much does each stage contribute to total welfare loss, accounting for dependencies?"

### Medical Suitability Criteria

We define **medically suitable referrals** using conservative, time-invariant criteria to avoid survivor bias:

**Inclusion criteria:**
- Age < 70 years
- BMI 15-45 kg/m²
- Known cause of death (not "Unknown")
- No absolute contraindications (active malignancy, HIV+, rabies, CJD)

**Rationale:** These criteria are deliberately conservative. Many donors outside these ranges are successfully transplanted, but using broader criteria would risk codifying existing inefficiencies.

### Multiple Imputation

We use **multiple imputation** (MICE algorithm) to handle missing data:
- 100% sample retention (no case deletion)
- 10 imputed datasets
- Pooled estimates using Rubin's rules

**Missing data rates:**
- Age: 0.2%
- BMI: 3.1%
- Cause of death: 8.4%
- Race: 12.7%

### Multivariate Regression

We estimate a logistic regression model to control for confounding:

**Model:**
```
Pr(approached = 1) = logit^{-1}(β₀ + β₁·age + β₂·BMI + β₃·race + β₄·gender + β₅·OPO + β₆·year + ε)
```

**Controls:**
- Demographics: age, BMI, race, gender
- Temporal: year fixed effects (2015-2021)
- Organizational: OPO fixed effects (6 OPOs)

**Sample:** 133,101 referrals  
**Pseudo R²:** 0.1177  
**Convergence:** Successful

---

## Repository Structure

```
organ-donation-ode/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── data/                              # Data directory (not tracked)
│   └── orchid/                        # ORCHID dataset (user must download)
├── scripts/                           # Analysis scripts
│   ├── 01_data_validation.py          # Data cleaning and validation
│   ├── 02_shapley_decomposition.py    # Shapley value computation
│   ├── 03_multivariate_regression.py  # Logistic regression analysis
│   ├── 04_opo_performance.py          # OPO variance analysis
│   ├── 05_age_effect.py               # Age bias analysis
│   └── 06_investment_analysis.py      # Investment misalignment analysis
├── results/                           # Output directory
│   ├── shapley_values.csv             # Shapley decomposition results
│   ├── regression_results.csv         # Multivariate regression output
│   └── figures/                       # Visualizations
└── docs/                              # Documentation
    ├── methodology.md                 # Detailed methodology
    └── data_dictionary.md             # ORCHID variable descriptions
```

---

## Installation

### Prerequisites

- Python 3.8+
- PhysioNet credentialed access to ORCHID dataset

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/2460bazinga/organ-donation-ode.git
cd organ-donation-ode
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download ORCHID dataset:**
   - Obtain credentialed access from PhysioNet
   - Download ORCHID v1.0.0
   - Place files in `data/orchid/`

---

## Usage

### Run Complete Analysis

```bash
# 1. Validate and clean data
python scripts/01_data_validation.py

# 2. Compute Shapley values
python scripts/02_shapley_decomposition.py

# 3. Run multivariate regression
python scripts/03_multivariate_regression.py

# 4. Analyze OPO performance
python scripts/04_opo_performance.py

# 5. Analyze age effects
python scripts/05_age_effect.py

# 6. Analyze investment misalignment
python scripts/06_investment_analysis.py
```

### Generate Figures

All figures from the paper can be reproduced:

```bash
python scripts/generate_figures.py
```

Output will be saved to `results/figures/`.

---

## Citation

If you use this code or findings in your research, please cite:

**Paper:**
```bibtex
@article{parrish2025ode,
  title={Organ Donation Equilibrium: An Empirical Analysis of Coordination Failure in the U.S. Organ Donation System},
  author={Parrish, Noah},
  journal={[Journal Name]},
  year={2025},
  note={In preparation}
}
```

**Dataset:**
```bibtex
@misc{adam2023orchid,
  title={Organ Retrieval and Collection of Health Information for Donation (ORCHID)},
  author={Adam, Hana and Suriyakumar, Vinith and Pollard, Tom and Moody, Benjamin and Erickson, Jared and Segal, Gabriel and Wiens, Jenna and Ghassemi, Marzyeh},
  year={2023},
  publisher={PhysioNet},
  doi={10.13026/eytj-4f29}
}
```

---

## Acknowledgments

The author acknowledges the use of large language models (Manus 1.5, Gemini 2.5 Flash, and Claude Opus 4.5) for assistance with literature review, data analysis, and computational implementation. All AI-generated content was reviewed, verified, and edited by the author. The author takes full responsibility for the accuracy and integrity of the work.

Special thanks to the ORCHID team at MIT and the Federation of American Scientists for making this dataset publicly available.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

**Noah Parrish**  
Email: [Your email]  
GitHub: [@2460bazinga](https://github.com/2460bazinga)

For questions about the ORCHID dataset, please contact the dataset authors through PhysioNet.

---

**Last Updated:** November 25, 2025
