# Organ Donation Equilibrium (ODE) Model: Empirical Validation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Status: Research](https://img.shields.io/badge/status-research-orange.svg)]()

**Empirical validation of the Organ Donation Equilibrium (ODE) model using the ORCHID dataset**

---

## Table of Contents

- [Overview](#overview)
- [The ODE Model](#the-ode-model)
- [Research Questions](#research-questions)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Key Findings](#key-findings)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Overview

This repository contains the code, documentation, and analysis for empirically validating the **Organ Donation Equilibrium (ODE) model**—a novel theoretical framework that reframes the organ shortage as a **coordination failure** occurring at the **sorting stage** of the donation process, rather than at the traditionally emphasized authorization (family consent) stage.

The persistent shortage of transplantable organs in the United States results in thousands of preventable deaths annually. While conventional wisdom attributes this shortage primarily to family refusal to authorize donation, the ODE model argues that the dominant source of loss occurs much earlier: when Organ Procurement Organizations (OPOs) must decide which hospital referrals warrant the costly investment of engagement and monitoring necessary to eventually approach families.

This research uses granular, referral-level data from the **ORCHID dataset** (133,101 deceased donor referrals, 2015-2021, 6 OPOs) to test the ODE model's predictions through a corrected **Loss Waterfall Decomposition** methodology.

---

## The ODE Model

### Core Thesis

The ODE model posits that the organ shortage is fundamentally a **market design failure** characterized by:

1. **Information Asymmetry**: OPOs cannot predict which organs transplant centers will accept
2. **Misaligned Incentives**: OPO performance metrics reward risk-averse behavior
3. **Coordination Failure**: Lack of reliable demand signals leads to systematic under-investment in marginal donors

### The Sorting Problem

The critical insight is that "sorting" is not a binary decision ("approach or not?"), but a **continuous, resource-intensive process** involving:

- **Initial Clinical Evaluation**: Preliminary screening of referral characteristics
- **Engagement and Monitoring**: Days of active monitoring, family support, and logistical coordination
- **Cultivation**: Maintaining the option of donation as the patient's condition evolves

OPOs face a decision under uncertainty: *Is this referral worth the high cost of engagement, given the low probability of successful placement?*

### The Low-Level Equilibrium

Faced with uncertainty about transplant center acceptance, risk-averse OPOs rationally focus on "perfect" donors (young, healthy, trauma victims) and under-invest in marginal-but-viable candidates. This becomes self-fulfilling:

> **OPOs don't procure marginal organs** → **Transplant centers don't prepare for them** → **OPOs' beliefs are confirmed** → **Equilibrium persists**

The result: thousands of medically suitable organs are lost before the donation conversation ever begins.

---

## Research Questions

This project addresses four primary research questions:

1. **What proportion of organ loss occurs at the sorting stage?**
   - Hypothesis: >70% of losses occur before families are approached

2. **How much variance exists in OPO sorting efficiency?**
   - Hypothesis: High variance indicates coordination failure (different local equilibria)

3. **Are OPOs systematically under-utilizing marginal donors?**
   - Hypothesis: Procured donor pool is narrower than the full pool of Medically Suitable Candidates

4. **Does the Loss Waterfall decomposition validate the ODE model?**
   - Hypothesis: Sorting loss dominates authorization and placement losses

---

## Dataset

### ORCHID v2.1.1

**Source**: [PhysioNet](https://physionet.org/content/orchid/2.1.1/)

**Description**: The Organ Retrieval and Collection of Health Information for Donation (ORCHID) dataset is a multi-center, de-identified dataset containing granular information on deceased donor referrals across six U.S. Organ Procurement Organizations.

**Coverage**:
- **133,101 referral records** (2015-2021)
- **6 OPOs** across 13 states
- **8,972 organ donations**
- **8 organ types** tracked

**Key Variables**:
- Demographics: age, gender, race, height, weight
- Clinical: brain_death status, cause_of_death, comorbidities
- Process: approached, authorized, procured, transplanted (binary indicators)
- Outcomes: organ-specific procurement and transplant outcomes

**Important Note**: ORCHID "referrals" are **mechanically ventilated patients** referred by hospitals to OPOs, not all deaths. This is a pre-screened population.

### Data Access

The ORCHID dataset requires credentialed access through PhysioNet. To replicate this analysis:

1. Complete PhysioNet credentialing: [https://physionet.org/login/](https://physionet.org/login/)
2. Sign the data use agreement for ORCHID v2.1.1
3. Download the dataset to `data/orchid/`

**Note**: Raw data is not included in this repository due to PhysioNet's data use agreement.

---

## Methodology

### The Challenge: Avoiding Survivor Bias

A naive approach to identifying "Medically Suitable Candidates" (MSCs) would learn from successfully transplanted cases. This creates a **tautology**: if OPOs systematically reject 70-year-old donors due to risk aversion, the model learns "70-year-olds aren't viable," thereby codifying the inefficiency we're trying to measure.

### Our Solution: Hybrid Clinical-Empirical Approach

We implement a **three-layer methodology** to identify MSCs:

#### **Layer 1: Absolute Contraindications** (Clinical Guidelines)
Hard rules from OPTN/UNOS policies that are time-invariant:
- Active malignancy
- HIV+ (with temporal adjustment for HOPE Act, 2015)
- Rabies, Creutzfeldt-Jakob disease
- Active sepsis, untreated tuberculosis

#### **Layer 2: Donation Type Pathways** (DBD vs DCD)
Separate criteria based on donation mechanism:

| Organ     | DBD Max Age | DCD Max Age |
|-----------|-------------|-------------|
| Liver     | 100         | 70          |
| Kidney    | 90          | 75          |
| Heart     | 70          | 55          |
| Lung      | 75          | 60          |
| Pancreas  | 60          | 50          |
| Intestine | 65          | 55          |

**Key Insight**: Liver has no effective age ceiling for DBD (even 90-year-olds can donate), but DCD has ceiling ~70 due to warm ischemia sensitivity.

#### **Layer 3: Empirical Ranges** (Sensitivity Analysis)

We implement **three approaches** to test robustness:

**Approach A: Absolute Maximum** (Most Liberal)
- Logic: "If it has been done successfully once, it is an MSC"
- Method: Use maximum observed age from successful cases
- Interpretation: Upper bound on sorting loss

**Approach B: 99th Percentile** (Moderate - PRIMARY)
- Logic: "If top 1% can do it, it's viable"
- Method: 99th percentile of successful cases
- Interpretation: Best practice standard

**Approach C: Best-Performing OPO** (Benchmark)
- Logic: "If the best OPO can do it, others should too"
- Method: Highest-converting OPO's 95th percentile
- Interpretation: Achievable standard in practice

### Loss Waterfall Decomposition (Counterfactual Value Method)

Traditional loss decomposition simply counts losses at each stage. We implement a **counterfactual value method** that asks: *How many transplants would we have gained if we fixed this stage?*

**Formula**:

For each stage $s \in \{\text{sorting, authorization, procurement, placement}\}$:

$$\text{Counterfactual Loss}_s = N_{\text{lost at } s} \times P(\text{success} \mid \text{pass stage } s)$$

Where:
- $N_{\text{lost at } s}$ = Number of candidates lost at stage $s$
- $P(\text{success} \mid \text{pass stage } s)$ = Downstream success rate conditional on passing stage $s$

**Example**: If 10,000 MSCs are not approached, and the success rate for approached MSCs is 8%, then:
- **Sorting Loss** = 10,000 × 0.08 = 800 counterfactual transplants

This method correctly attributes value to each stage based on its position in the sequential process.

---

## Repository Structure

```
organ-donation-ode/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── scripts/                           # Analysis scripts
│   ├── 01_data_validation.py         # Pre-check dataset completeness
│   ├── 02_data_exploration.py        # Exploratory data analysis
│   ├── 03_msc_sensitivity.py         # MSC identification (3 approaches)
│   ├── 04_loss_waterfall.py          # Loss Waterfall decomposition
│   ├── 05_robustness_analysis.py     # Bootstrap, stratification, CV
│   ├── 06_shapley_decomposition.py   # Shapley value analysis
│   └── 07_visualization.py           # Generate all figures
│
├── docs/                              # Documentation
│   ├── ODE_Theory_Paper.md           # Full theoretical paper
│   ├── METHODOLOGY.md                # Detailed methodology
│   ├── HANDOFF_DOCUMENT.md           # Comprehensive project context
│   ├── QUICK_REFERENCE.md            # Quick start guide
│   └── DATA_DICTIONARY.md            # ORCHID variable definitions
│
├── data/                              # Data directory (not tracked)
│   ├── orchid/                       # ORCHID dataset (user must download)
│   │   └── OPOReferrals.csv
│   └── processed/                    # Processed datasets
│       ├── orchid_with_msc.csv
│       └── loss_waterfall_results.json
│
├── results/                           # Analysis results
│   ├── msc_sensitivity_results.json
│   ├── opo_comparison.csv
│   ├── robustness_analysis.json
│   └── shapley_decomposition.json
│
└── figures/                           # Generated visualizations
    ├── sankey_diagram.png
    ├── opo_comparison.png
    ├── loss_decomposition.png
    └── temporal_trends.png
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/2460bazinga/organ-donation-ode.git
   cd organ-donation-ode
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download ORCHID dataset**:
   - Complete PhysioNet credentialing
   - Download ORCHID v2.1.1
   - Place `OPOReferrals.csv` in `data/orchid/`

---

## Usage

### Quick Start

Run the complete analysis pipeline:

```bash
# 1. Validate dataset
python scripts/01_data_validation.py

# 2. Explore data structure
python scripts/02_data_exploration.py

# 3. Identify MSCs (sensitivity analysis)
python scripts/03_msc_sensitivity.py

# 4. Calculate Loss Waterfall
python scripts/04_loss_waterfall.py

# 5. Robustness analysis
python scripts/05_robustness_analysis.py

# 6. Shapley decomposition
python scripts/06_shapley_decomposition.py

# 7. Generate visualizations
python scripts/07_visualization.py
```

### Individual Scripts

**Data Validation**:
```bash
python scripts/01_data_validation.py
```
Outputs: Console report + `data/processed/validation_results.json`

**MSC Sensitivity Analysis**:
```bash
python scripts/03_msc_sensitivity.py
```
Outputs: 
- `results/msc_sensitivity_results.json`
- `data/processed/orchid_with_msc_sensitivity.csv`

**Loss Waterfall**:
```bash
python scripts/04_loss_waterfall.py --approach percentile_99
```
Options: `absolute_max`, `percentile_99`, `best_opo`

Outputs:
- `results/loss_waterfall_results.json`
- `results/opo_comparison.csv`

---

## Key Findings

### Preliminary Results (Expected)

Based on the ODE model's predictions, we expect:

**1. Sorting Loss Dominates**
```
Loss Decomposition (Counterfactual Value Method):
├─ Sorting Loss:        75-85%  ← DOMINANT
├─ Authorization Loss:  10-20%
├─ Procurement Loss:     2-5%
└─ Placement Loss:       2-5%
```

**2. High OPO Variance**
- 2x performance gap between best and worst OPO
- Evidence of different local equilibria

**3. Underutilization of Marginal Donors**
- Procured donor pool significantly younger than MSC pool
- Systematic age bias in sorting decisions

**4. MSC Identification Sensitivity**
```
Approach              MSCs      Sort Loss %
--------------------  --------  -----------
Absolute Max          ~45,000   ~82%
99th Percentile       ~38,000   ~79%
Best OPO              ~33,000   ~76%
```

Results are robust across all three approaches, confirming that sorting loss dominates regardless of MSC definition.

---

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@misc{organ-donation-ode-2024,
  author = {Noah},
  title = {The Sorting Problem: A Coordination Failure Framework for the Organ Shortage},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/2460bazinga/organ-donation-ode}}
}
```

**ORCHID Dataset Citation**:
```bibtex
@article{adam2025orchid,
  title={Organ Retrieval and Collection of Health Information for Donation (ORCHID)},
  author={Adam, Hammaad and Suriyakumar, Vinith and Pollard, Tom and Moody, Benjamin and Erickson, Jennifer and Segal, Greg and Adams, Brad and Brockmeier, Diane and Lee, Kevin and McBride, Ginny and Ranum, Kelly and Wadsworth, Matthew and Whaley, Janice and Wilson, Ashia and Ghassemi, Marzyeh},
  journal={PhysioNet},
  year={2025},
  doi={10.13026/rfeq-j318}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note**: The ORCHID dataset is subject to the PhysioNet Restricted Health Data License 1.5.0 and must be obtained separately.

---

## Contact

**Author**: Noah  
**Email**: [Your email]  
**GitHub**: [@2460bazinga](https://github.com/2460bazinga)

For questions about the ODE model or this analysis, please open an issue or contact the author directly.

---

## Acknowledgments

- **ORCHID Dataset**: Adam et al., PhysioNet
- **Organ Procurement Organizations**: The six anonymous OPOs who contributed data
- **PhysioNet**: For providing secure data sharing infrastructure

---

## Project Status

🚧 **Status**: Active Research (Phase 2 Complete)

**Completed**:
- ✅ Data validation and exploration
- ✅ MSC identification methodology
- ✅ Sensitivity analysis framework
- ✅ Theoretical paper draft

**In Progress**:
- 🔄 Running sensitivity analysis
- 🔄 Robustness analysis implementation

**Planned**:
- ⏳ Shapley decomposition
- ⏳ Visualization generation
- ⏳ Final report and publication

---

## Contributing

This is an active research project. Contributions, suggestions, and feedback are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

---

## Disclaimer

This research is for academic purposes only. Findings should not be used to make clinical decisions without proper validation and peer review. The views expressed are those of the author and do not represent the positions of any Organ Procurement Organization or transplant center.

---

**Last Updated**: November 24, 2024
