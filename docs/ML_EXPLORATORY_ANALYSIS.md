# Exploratory Machine Learning Analysis

## Overview

Having established the core empirical findings (78.4% sorting loss, infrastructure constraints, free lunch), this exploratory analysis uses machine learning to discover **unexpected patterns** and generate **new hypotheses** for future research.

Unlike the confirmatory analysis (which tested specific ODE model predictions), this is **hypothesis-generating** research designed to surface non-obvious insights from the data.

---

## Motivation

### What We've Proven:
- ✅ Sorting loss dominates (78.4%)
- ✅ Infrastructure constraints drive losses (weekend effect)
- ✅ No quality trade-off exists (free lunch)

### What We Don't Know:
- ❓ What non-obvious factors predict sorting success/failure?
- ❓ Are there distinct subpopulations with different sorting patterns?
- ❓ What edge cases defy the general pattern?
- ❓ What interaction effects exist between variables?

### Why Machine Learning?

**Traditional statistical methods** (regression, ANOVA) test pre-specified hypotheses about linear relationships.

**Machine learning** excels at:
1. **Non-linear pattern detection** (neural networks)
2. **High-dimensional interaction effects** (gradient boosting)
3. **Unsupervised discovery** (clustering)
4. **Anomaly detection** (edge cases)

This is **exploratory**, not confirmatory. The goal is to generate new questions, not answer old ones.

---

## Experiment 1: Non-Obvious Predictors of Sorting Success

### Research Question:
**Beyond age and brain death status, what predicts whether an MSC will be approached?**

### Method:

**Target Variable**: `approached` (binary: 1 if MSC was approached, 0 otherwise)

**Features** (16 total):
- **Patient characteristics**: age, BMI, gender, race, cause of death
- **Donation pathway**: brain_death (DBD vs DCD)
- **Temporal factors**: hour of day, day of week, month, year, is_weekend, is_business_hours
- **Derived features**: is_elderly (≥60), is_pediatric (<18)
- **Organizational**: OPO identifier

**Models**:
1. **Random Forest** (interpretable baseline, feature importance)
2. **Neural Network** (non-linear patterns, 3 hidden layers: 64-32-16)
3. **Gradient Boosting** (interaction effects)

**Evaluation**: ROC-AUC, classification metrics, feature importance

### Expected Insights:

1. **Temporal Effects**: Do hour/day patterns predict sorting independent of volume?
2. **OPO Fixed Effects**: How much variance is explained by OPO identity?
3. **Interaction Effects**: Do age thresholds vary by cause of death or OPO?
4. **Non-linear Relationships**: Are there threshold effects (e.g., age 65 cliff)?

### Interpretation Framework:

**High Predictive Power (AUC > 0.75)**:
- Sorting decisions are systematic and rule-based
- Suggests conscious screening criteria (good for intervention design)

**Low Predictive Power (AUC < 0.60)**:
- Sorting decisions are noisy/random
- Suggests capacity constraints dominate (consistent with infrastructure hypothesis)

**Key Features**:
- If temporal features rank high → infrastructure constraint confirmed
- If patient features dominate → clinical screening is primary mechanism
- If OPO fixed effects are strong → coordination failure (different equilibria)

---

## Experiment 2: Hidden Clusters in MSC Population

### Research Question:
**Are there distinct subpopulations of MSCs with different sorting outcomes?**

### Method:

**Approach**: K-Means clustering on MSC characteristics

**Features**:
- age, BMI, brain_death, hour, day_of_week

**Dimensionality Reduction**: PCA (2 components for visualization)

**Cluster Selection**: Elbow method + Silhouette score

**Analysis**:
- Compare sorting rates across clusters
- Characterize each cluster (demographics, outcomes)
- Identify under-served populations

### Expected Insights:

**Potential Clusters**:
1. **"Perfect Donors"**: Young, DBD, high approach rate
2. **"Marginal but Viable"**: Elderly, DCD, low approach rate (opportunity!)
3. **"Weekend Referrals"**: Low volume periods, higher approach rate
4. **"High-Volume Casualties"**: Weekday referrals, fall through cracks

### Interpretation:

If distinct clusters exist with different sorting rates:
- **Policy Implication**: Design cluster-specific cultivation strategies
- **Equity Concern**: Are some populations systematically under-served?
- **Efficiency Gain**: Target interventions to high-loss clusters

---

## Experiment 3: Anomaly Detection (Edge Cases)

### Research Question:
**What cases defy the general pattern and why?**

### Method:

**Anomaly Types**:

1. **Young DBD Donors NOT Approached**
   - Filter: age < 40, brain_death = True, approached = False
   - **Why interesting**: These should be "perfect" donors. What went wrong?
   - **Hypothesis**: System failures (communication breakdown, staffing gaps)

2. **Elderly DCD Donors Successfully Transplanted**
   - Filter: age > 65, brain_death = False, transplanted = True
   - **Why interesting**: These defy conventional age limits. What made them work?
   - **Hypothesis**: Proof of untapped biological capacity

3. **Weekend Approaches**
   - Filter: day_of_week in [5, 6], approached = True
   - **Why interesting**: Weekend effect shows lower efficiency, but some succeed
   - **Hypothesis**: What's different about these cases?

4. **Authorized but NOT Procured**
   - Filter: authorized = True, procured = False
   - **Why interesting**: Family said yes, but organs not recovered
   - **Hypothesis**: Downstream coordination failures

### Analysis:

For each anomaly type:
- Count frequency
- Characterize common features
- Compare to general population
- Generate hypotheses for case study follow-up

### Expected Insights:

**Anomaly 1** → Evidence of preventable sorting failures  
**Anomaly 2** → Evidence of biological capacity exceeding utilization  
**Anomaly 3** → Evidence of capacity-dependent success  
**Anomaly 4** → Evidence of multi-stage coordination failures  

---

## Interpretation Guidelines

### What This Analysis Can Tell Us:

✅ **Patterns**: Non-obvious correlations and clusters  
✅ **Hypotheses**: New questions for future research  
✅ **Opportunities**: Under-utilized populations or time periods  
✅ **Anomalies**: Edge cases for qualitative follow-up  

### What This Analysis Cannot Tell Us:

❌ **Causation**: ML finds correlations, not causal mechanisms  
❌ **Mechanisms**: Why patterns exist (need qualitative research)  
❌ **Generalization**: ORCHID is 6 OPOs, not nationally representative  
❌ **Policy Prescriptions**: Need experimental validation before intervention  

### How to Use These Findings:

1. **Generate Hypotheses**: Use patterns to design targeted studies
2. **Identify Opportunities**: Focus interventions on high-loss clusters
3. **Case Studies**: Follow up on anomalies with qualitative research
4. **Validate Externally**: Test patterns in other datasets (SRTR, DonorNet)

---

## Expected Outputs

### Figures:
1. `ml_feature_importance_rf.png` - What predicts sorting success?
2. `ml_roc_comparison.png` - Model performance comparison
3. `ml_clustering_selection.png` - Optimal number of clusters
4. `ml_clusters_visualization.png` - PCA visualization of clusters
5. `ml_anomaly_examples.png` - Characteristics of edge cases

### Data:
1. `anomalous_cases.csv` - All identified anomalies for case study follow-up
2. `cluster_characteristics.csv` - Demographics and outcomes by cluster
3. `feature_importance.csv` - Ranked predictors from all models

---

## Potential Discoveries

### Hypothesis 1: Temporal Threshold Effects
**Pattern**: Sorting efficiency may not decline linearly with volume, but collapse at a threshold (e.g., >20 referrals/day)

**Implication**: Capacity expansion should target high-volume days, not uniform increases

### Hypothesis 2: OPO-Specific Age Biases
**Pattern**: Age thresholds may vary dramatically across OPOs (e.g., OPO1 rejects >60, OPO5 accepts >70)

**Implication**: Age bias is learned/cultural, not biological—can be changed through training

### Hypothesis 3: Cause of Death Interaction
**Pattern**: Age tolerance may depend on cause of death (e.g., stroke victims accepted older than trauma)

**Implication**: Screening criteria should be cause-specific, not age-universal

### Hypothesis 4: Weekend Paradox
**Pattern**: Weekend referrals may be higher-quality (hospitals only refer obvious cases when short-staffed)

**Implication**: Weekend effect is not just capacity—it's also case mix

### Hypothesis 5: Hidden High-Performers
**Pattern**: Some clusters may have high approach rates but low authorization rates (or vice versa)

**Implication**: Different bottlenecks for different populations—one-size-fits-all interventions will fail

---

## Limitations

### Methodological:
- **Overfitting risk**: Neural networks may learn noise, not signal
- **Feature engineering**: Limited by ORCHID variable availability
- **Class imbalance**: Only ~15% of MSCs approached (may bias models)
- **Temporal correlation**: Referrals are not independent (same OPO, same week)

### Interpretive:
- **Correlation ≠ Causation**: ML finds patterns, not mechanisms
- **Post-hoc analysis**: Exploratory findings need prospective validation
- **Multiple testing**: Many comparisons increase false discovery risk
- **External validity**: 6 OPOs may not represent national patterns

### Practical:
- **Computational cost**: Neural networks require GPU for large datasets
- **Interpretability**: Black-box models hard to explain to clinicians
- **Actionability**: Patterns don't automatically suggest interventions

---

## Next Steps

### Immediate:
1. **Run the analysis** on ORCHID data
2. **Review anomalies** for qualitative insights
3. **Validate patterns** in holdout data

### Short-term:
1. **Case studies**: Interview OPO staff about anomalous cases
2. **External validation**: Test patterns in SRTR data
3. **Interaction analysis**: Deep dive on gradient boosting feature interactions

### Long-term:
1. **Prospective study**: Test ML-generated hypotheses in new data
2. **Intervention design**: Use cluster profiles to target interventions
3. **Causal inference**: Use patterns to guide instrumental variable selection

---

## Ethical Considerations

### Equity:
- **Risk**: ML may codify existing biases (e.g., racial disparities in sorting)
- **Mitigation**: Explicitly test for demographic disparities in cluster analysis
- **Transparency**: Report any identified equity concerns prominently

### Interpretability:
- **Risk**: Black-box models may be used to justify opaque decisions
- **Mitigation**: Prioritize interpretable models (Random Forest) over performance
- **Transparency**: Always provide feature importance alongside predictions

### Actionability:
- **Risk**: Findings may be misinterpreted as causal or actionable
- **Mitigation**: Clearly label as hypothesis-generating, not confirmatory
- **Transparency**: Emphasize need for experimental validation before policy changes

---

## Conclusion

This exploratory ML analysis is designed to **generate new questions**, not answer old ones. By surfacing non-obvious patterns, hidden clusters, and anomalous cases, we can:

1. **Refine the ODE model** with empirically-grounded mechanisms
2. **Design targeted interventions** for specific subpopulations
3. **Identify research priorities** for future studies
4. **Challenge assumptions** about donor viability and screening

The goal is not to replace domain expertise with algorithms, but to **augment clinical intuition with data-driven discovery**.

---

**Status**: Ready to run  
**Estimated Runtime**: ~5-10 minutes on standard hardware  
**Dependencies**: scikit-learn, pandas, numpy, matplotlib, seaborn  
**Output**: 5 figures + 3 CSV files + console report
