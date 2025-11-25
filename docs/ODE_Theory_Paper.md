# The Organ Donation Equilibrium: A Coordination Failure Framework for the Organ Shortage

**Author:** Noah (Family Coordinator)  
**Data Source:** ORCHID dataset (Adam, H. et al., Yale University)  
**Date:** November 24, 2024

---

## Abstract

The persistent shortage of transplantable organs in the United States is conventionally attributed to the "authorization bottleneck"—the failure to secure consent from the families of potential donors. This paper challenges that paradigm by introducing the Organ Donation Equilibrium (ODE) model, a framework that re-identifies the primary source of organ loss as a systemic coordination failure occurring at the "sorting" stage. We argue that the critical decision is not the binary choice of whether to approach a family, but the preceding, resource-intensive process of identifying and cultivating viable candidates from a vast pool of noisy referrals. Organ Procurement Organizations (OPOs), operating under conditions of profound information asymmetry and misaligned performance incentives, rationally under-invest in this sorting process. This leads to a low-level equilibrium where a majority of medically suitable organs are lost before the donation conversation ever begins. The ODE model provides a new theoretical lens for understanding the organ shortage and generates a set of empirically testable hypotheses, suggesting that policy interventions should focus on correcting the coordination failures in sorting rather than exclusively on improving authorization rates.

---

## 1. Introduction

Every year, thousands of patients in the United States die while waiting for a life-saving organ transplant [1]. This enduring public health crisis persists despite broad public support for organ donation and decades of policy efforts aimed at increasing the supply of transplantable organs. The dominant narrative framing this shortage has historically focused on the **authorization stage** of the donation process. In this view, the primary bottleneck is the failure to secure consent from the next-of-kin of deceased individuals who are medically suitable for donation. Consequently, policy interventions have largely centered on public awareness campaigns, professional training for requestors, and other measures designed to increase family consent rates.

While not insignificant, we argue that this focus on the authorization bottleneck is misplaced and overlooks a far larger, upstream source of inefficiency. This paper posits that the majority of organ loss occurs at the **sorting stage**, a complex and costly process in which Organ Procurement Organizations (OPOs) must evaluate thousands of hospital referrals to identify the small fraction that represent truly viable donor candidates. This process is not a simple medical checklist; it is a dynamic resource allocation problem fraught with uncertainty, information asymmetry, and perverse incentives.

We introduce the **Organ Donation Equilibrium (ODE) model** to formalize this concept. The ODE model reframes the organ shortage as a **coordination failure** between the 58 geographically distinct OPOs and the hundreds of transplant centers they serve. In this equilibrium, OPOs rationally under-invest in the costly process of sorting and monitoring marginal referrals because their expectation of a successful placement is low. This becomes a self-fulfilling prophecy: because OPOs do not invest in cultivating marginal donors, the supply of such organs remains low, reinforcing the transplant centers’ and OPOs’ initial beliefs. The result is a stable, but highly inefficient, equilibrium where a significant number of medically suitable organs are systematically lost.

This paper proceeds as follows: Section 2 reviews the conventional "authorization bottleneck" framework. Section 3 introduces the ODE model, with a detailed exposition of the sorting problem as a process of engagement and monitoring. Section 4 formalizes the OPO's decision problem. Section 5 derives the model's empirical implications and testable hypotheses. Section 6 concludes with a discussion of the policy implications.

---

## 2. The Conventional View: The Authorization Bottleneck

The traditional understanding of the organ donation process is often depicted as a linear "waterfall" or funnel, where losses accumulate at discrete stages. A simplified version is as follows:

> **Total Deaths → Medically Suitable Deaths → Referrals → Approached → Authorized → Procured → Transplanted**

Within this framework, the most significant drop-off in public and policy discourse has been the transition from *Approached* to *Authorized*. National authorization rates hover between 60-75%, meaning that for every four families approached, at least one declines consent [2]. This refusal is a tangible, emotionally resonant event that is easy to measure and has thus become the primary target of intervention. The logic is straightforward: if we could increase the authorization rate from 75% to 90%, we could increase the number of donors by 20%.

This view, however, makes a critical and flawed assumption: that the pool of donors being approached for consent represents the full extent of the viable donor potential. It implicitly assumes that the upstream sorting process—from referral to approach—is highly efficient and that nearly all medically suitable candidates are correctly identified and advanced to the authorization stage. The ODE model argues that this assumption is not only incorrect but that it masks the single largest source of inefficiency in the entire system.

---

## 3. A New Framework: The Organ Donation Equilibrium (ODE) Model

The ODE model shifts the analytical focus from the authorization decision to the sorting process. We redefine sorting not as a single decision, but as a continuous, resource-intensive process of **engagement and monitoring** that begins the moment a hospital referral is made.

### 3.1 The True Nature of the Sorting Problem

A hospital referral is not a clean signal of a potential donor; it is a noisy signal of a patient who is near death and on mechanical ventilation. An OPO may receive thousands of such referrals annually. The OPO must then decide which of these referrals warrant the investment of significant resources. This investment includes:

*   **Initial Clinical Evaluation:** A preliminary assessment of the patient's chart to rule out absolute contraindications.
*   **Continuous Monitoring:** For referrals that pass the initial screen, OPO staff must remain engaged, sometimes for days, monitoring the patient’s physiological status. A patient’s suitability can change rapidly; an organ that is viable today may not be tomorrow.
*   **Logistical and Family Support:** This can involve coordinating with hospital staff, providing support to the family (long before any donation conversation), and managing the complex logistics required to maintain the option of donation.

This process is costly, both in terms of staff time and financial resources. The critical decision for the OPO is not, "Should we ask this family for donation?" but rather, **"Is this specific referral worth the high cost of engagement and monitoring, given the low probability of a successful outcome?"**

### 3.2 OPO Incentives and Risk Aversion

OPOs operate as regional monopolies and are evaluated by the Centers for Medicare & Medicaid Services (CMS) based on metrics that prioritize the number of transplants performed relative to a benchmark [3]. This creates a powerful incentive to avoid risk. An OPO that spends resources on a marginal donor—for example, an older patient or one with comorbidities—faces two primary risks:

1.  **Procurement without Placement:** The OPO successfully procures the organ, but no transplant center will accept it. This is a costly failure, as the OPO bears the full cost of procurement but gets no credit in its performance metrics.
2.  **Poor Post-Transplant Outcomes:** If a marginal organ is transplanted and fails, it can negatively impact the transplant center's own performance metrics, making them less likely to accept similar organs in the future.

Faced with these risks, the rational strategy for a risk-averse OPO is to focus its resources on "perfect" donors—young, healthy individuals who died from trauma—and to under-invest in the sorting and monitoring of more complex, marginal-but-viable candidates. This is not irrational behavior; it is a direct and predictable response to the incentive structure.

### 3.3 Information Asymmetry and Coordination Failure

The OPO's decision is made under a veil of profound information asymmetry. When evaluating a referral, the OPO does not know with certainty:

*   **The Transplant Center's True Acceptance Criteria:** While transplant centers publish general guidelines, their willingness to accept a marginal organ on any given day depends on the specific recipient, the surgeon on call, and their current risk tolerance. An organ they accept today, they may reject tomorrow.
*   **The Final Medical Viability:** The ultimate quality of an organ can only be fully assessed after procurement.

This lack of coordination between OPOs and transplant centers is the crux of the problem. Without a reliable signal from the demand side (transplant centers), the supply side (OPOs) defaults to a conservative strategy. They will not invest in procuring organs they are not confident will be accepted. This leads to a **low-level equilibrium of inefficiency**: OPOs don’t procure marginal organs because they believe transplant centers won’t accept them, and transplant centers don’t prepare for marginal organs because OPOs don’t procure them.

---

## 4. Formalizing the Model: The OPO's Decision Problem

Let us formalize the OPO's decision for a single referral, $i$. The referral has a vector of characteristics, $\theta_i$, some of which are observable (e.g., age, cause of death) and some of which are unobservable (e.g., true organ quality).

The OPO's decision process unfolds in two stages:

**Stage 1: The Sorting Investment**

The OPO must decide whether to invest a cost, $C_m$, to monitor and engage with the referral. If they do not invest, the process ends, and the potential organ is lost. If they invest, they learn more about the patient's trajectory and the probability of progressing to brain death or a controlled DCD scenario.

**Stage 2: The Procurement Decision**

If the patient becomes a candidate for donation, the OPO must decide whether to initiate the procurement process, which has a much larger cost, $C_p$. This decision is made based on the OPO's belief about the probability of successful placement, $P(T | \theta_i)$, and the revenue (or metric credit), $R$, from a successful transplant.

The OPO's expected payoff from procuring the organ is:

> $E[U_{procure}] = P(T | \theta_i) \cdot R - C_p$

The OPO will only procure if $E[U_{procure}] > 0$. The core of the ODE model is that the OPO's belief, $P(T | \theta_i)$, is endogenous and shaped by its past interactions with transplant centers. If past attempts to place marginal organs have failed, $P(T | \theta_i)$ will be low for any $\theta_i$ that deviates from the "perfect donor" profile.

Anticipating this, the OPO's initial decision to invest $C_m$ in sorting is also affected. The expected payoff from monitoring is:

> $E[U_{monitor}] = P(\text{candidate} | \theta_i) \cdot \max(0, E[U_{procure}]) - C_m$

If the expected payoff from procurement is zero for a large class of referrals, the OPO will rationally choose not to invest $C_m$ in monitoring them in the first place. This is the sorting loss: medically suitable candidates are filtered out at the very first step because the OPO, anticipating coordination failure down the line, decides it is not worth the initial investment.

---

## 5. Empirical Implications and Testable Hypotheses

The ODE model generates a set of clear, testable hypotheses that differ sharply from those of the authorization-bottleneck framework.

*   **Hypothesis 1 (The Dominance of Sorting Loss):** A correctly specified Loss Waterfall Decomposition will show that the largest percentage of organ loss occurs at the sorting stage (MSCs who are never approached), dwarfing losses at the authorization and placement stages.

*   **Hypothesis 2 (High Variance in OPO Performance):** OPO performance, particularly in sorting efficiency, will exhibit high variance. This variance is a signature of coordination failure, as different OPOs will have settled into different local equilibria with their respective transplant centers.

*   **Hypothesis 3 (Correlation between Sorting and Placement):** OPOs with higher sorting efficiency (i.e., they approach more marginal candidates) may have lower placement rates for those marginal organs. This reflects the underlying risk of procuring organs that are harder to place.

*   **Hypothesis 4 (Underperformance Relative to Biological Capacity):** The characteristics of the procured donor pool will be significantly narrower than the characteristics of the full pool of Medically Suitable Candidates. For example, the average age of procured donors will be much lower than the average age of all MSCs, indicating that OPOs are systematically leaving older, viable donors on the table.

These hypotheses can be tested using granular, referral-level data, such as that contained in the ORCHID dataset, by first developing a robust classifier for identifying MSCs based on clinical guidelines rather than observed system behavior.

---

## 6. Conclusion and Policy Implications

The Organ Donation Equilibrium model reframes the organ shortage not as a problem of public persuasion, but as a systemic market design failure. The low-level equilibrium in which the system is trapped is a rational response by individual actors (OPOs) to a flawed set of incentives and a lack of effective coordination mechanisms. The tragic consequence is the loss of thousands of viable organs each year.

If the ODE model is correct, then policy interventions must shift accordingly. Instead of focusing solely on authorization rates, efforts should be directed toward:

1.  **Reforming OPO Performance Metrics:** Incentives should be redesigned to reward OPOs for accurately identifying and pursuing all medically suitable candidates, not just for maximizing the number of successful transplants. This could involve credit for successful procurements, even if the organ is not placed.

2.  **Improving Coordination Platforms:** New systems are needed to provide better and more reliable information flow between OPOs and transplant centers. A national, standardized system for expressing real-time acceptance criteria could reduce the uncertainty that drives OPO risk aversion.

3.  **Investing in Organ Perfusion Technology:** Technologies like machine perfusion can extend the viability of marginal organs, reducing the risk of post-transplant failure and making transplant centers more willing to accept them.

By correctly identifying the sorting stage as the primary bottleneck, we can begin to design interventions that address the root cause of the organ shortage. The challenge is not to convince more families to say "yes," but to build a system that honors their generosity by ensuring that every viable organ has a chance to save a life.

---

## References

[1] Organ Procurement and Transplantation Network (OPTN). "National Data." U.S. Department of Health and Human Services. [https://optn.transplant.hrsa.gov/data/](https://optn.transplant.hrsa.gov/data/)

[2] Sheehy, E., et al. (2019). "An overview of the organ donation process." *Journal of the American Medical Association*.

[3] Centers for Medicare & Medicaid Services (CMS). "Organ Procurement Organization (OPO) Conditions for Coverage Final Rule." [https://www.cms.gov/](https://www.cms.gov/)
