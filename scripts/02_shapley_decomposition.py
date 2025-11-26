"""
Correct 3-Stage Shapley Value Decomposition for Welfare Loss Attribution
Endpoint: Procurement (not transplant)

Stages:
1. Sorting: Referral → Approach
2. Authorization: Approach → Authorization  
3. Procurement: Authorization → Procurement

We decompose the loss from "medically suitable referrals" to "procured donors"
"""

import math
from itertools import combinations

# Data from ORCHID dataset
total_referrals = 133101
approached = 47629
authorized = 38412
procured = 21847

# Estimate medically suitable referrals
# From Analysis 2: 64.2% of non-approached were medically suitable
unapproached = total_referrals - approached
medically_suitable_unapproached = 0.642 * unapproached
medically_suitable_total = approached + medically_suitable_unapproached

print("=" * 80)
print("3-STAGE SHAPLEY DECOMPOSITION (PROCUREMENT ENDPOINT)")
print("=" * 80)
print()

print("=== BASELINE DATA ===")
print(f"Total Referrals: {total_referrals:,}")
print(f"Medically Suitable (estimated): {medically_suitable_total:,.0f}")
print(f"Approached: {approached:,}")
print(f"Authorized: {authorized:,}")
print(f"Procured: {procured:,}")
print()

# Calculate stage-specific conversion rates
approach_rate = approached / medically_suitable_total
auth_rate_conditional = authorized / approached
proc_rate_conditional = procured / authorized

print("=== STAGE-SPECIFIC CONVERSION RATES ===")
print(f"Approach Rate (of suitable): {100*approach_rate:.1f}%")
print(f"Authorization Rate (of approached): {100*auth_rate_conditional:.1f}%")
print(f"Procurement Rate (of authorized): {100*proc_rate_conditional:.1f}%")
print()

# Calculate absolute losses at each stage
sorting_loss = medically_suitable_total - approached
authorization_loss = approached - authorized
procurement_loss = authorized - procured
total_loss = medically_suitable_total - procured

print("=== STAGE-SPECIFIC LOSSES (Absolute) ===")
print(f"Sorting Loss: {sorting_loss:,.0f} ({100*sorting_loss/medically_suitable_total:.1f}% of suitable)")
print(f"Authorization Loss: {authorization_loss:,.0f} ({100*authorization_loss/approached:.1f}% of approached)")
print(f"Procurement Loss: {procurement_loss:,.0f} ({100*procurement_loss/authorized:.1f}% of authorized)")
print(f"Total Loss: {total_loss:,.0f}")
print()

# Simple proportional attribution
print("=== SIMPLE PROPORTIONAL ATTRIBUTION ===")
print(f"Sorting: {100*sorting_loss/total_loss:.1f}%")
print(f"Authorization: {100*authorization_loss/total_loss:.1f}%")
print(f"Procurement: {100*procurement_loss/total_loss:.1f}%")
print()

# Shapley value calculation
def loss_with_failures(failing_stages):
    """
    Calculate total loss if only the specified stages fail.
    failing_stages: set of stages that operate at current rates
    Other stages operate perfectly (100% conversion).
    """
    remaining = medically_suitable_total
    
    # Sorting stage
    if 'sorting' in failing_stages:
        # Current sorting: only approach 47,629 out of ~102,502 suitable
        remaining = approached
    else:
        # Perfect sorting: approach all suitable
        remaining = medically_suitable_total
    
    # Authorization stage
    if 'authorization' in failing_stages:
        # Current authorization: 80.6% authorize
        remaining = remaining * auth_rate_conditional
    else:
        # Perfect authorization: 100% authorize
        pass  # remaining unchanged
    
    # Procurement stage
    if 'procurement' in failing_stages:
        # Current procurement: 56.9% procured
        remaining = remaining * proc_rate_conditional
    else:
        # Perfect procurement: 100% procured
        pass  # remaining unchanged
    
    # Loss is the gap between what we started with and what we end with
    loss = medically_suitable_total - remaining
    return loss

# Calculate v(S) for all coalitions
print("=== CHARACTERISTIC FUNCTION (Loss with failing stages) ===")
stages = ['sorting', 'authorization', 'procurement']
coalitions_loss = {}

for r in range(len(stages) + 1):
    for coalition in combinations(stages, r):
        coalition_set = frozenset(coalition)
        loss = loss_with_failures(coalition_set)
        coalitions_loss[coalition_set] = loss
        stage_names = set(coalition) if coalition else "no stages"
        print(f"Loss when {stage_names} fail: {loss:,.0f}")

print()

# Calculate Shapley values
def shapley_value_loss(stage, coalitions_loss, all_stages):
    """Calculate the Shapley value for a given stage's contribution to loss."""
    shapley = 0
    n = len(all_stages)
    
    # Iterate over all coalitions not containing this stage
    for r in range(n):
        for coalition in combinations([s for s in all_stages if s != stage], r):
            coalition_set = frozenset(coalition)
            coalition_with_stage = frozenset(coalition_set | {stage})
            
            # Marginal contribution to loss when this stage is added
            marginal_loss = coalitions_loss[coalition_with_stage] - coalitions_loss[coalition_set]
            
            # Weight
            weight = 1 / (n * math.comb(n - 1, r))
            
            shapley += weight * marginal_loss
    
    return shapley

print("=== SHAPLEY VALUE DECOMPOSITION ===")
shapley_values = {}
for stage in stages:
    sv = shapley_value_loss(stage, coalitions_loss, stages)
    shapley_values[stage] = sv
    print(f"Shapley Value ({stage}): {sv:,.0f} loss")

print()
print(f"Actual Total Loss: {total_loss:,.0f}")
print(f"Sum of Shapley Values: {sum(shapley_values.values()):,.0f}")
print()

# Attribution percentages
print("=== WELFARE LOSS ATTRIBUTION (Shapley) ===")
for stage in stages:
    percentage = (shapley_values[stage] / total_loss) * 100
    print(f"{stage.capitalize()}: {percentage:.1f}%")

total_pct = sum(shapley_values.values()) / total_loss * 100
print(f"\nTotal: {total_pct:.1f}%")
print()

# Comparison table
print("=== COMPARISON: PROPORTIONAL vs SHAPLEY ===")
print(f"{'Stage':<15} {'Proportional':<15} {'Shapley':<15}")
print("-" * 45)
for stage in stages:
    prop_pct = (locals()[f"{stage}_loss"] / total_loss) * 100
    shapley_pct = (shapley_values[stage] / total_loss) * 100
    print(f"{stage.capitalize():<15} {prop_pct:>6.1f}%{'':<8} {shapley_pct:>6.1f}%")

print()
print("=" * 80)
print("INTERPRETATION")
print("=" * 80)
print()
print("The Shapley decomposition accounts for interaction effects between stages.")
print("It answers: 'What is each stage's marginal contribution to total welfare loss,")
print("averaged across all possible orderings of stage failures?'")
print()
print("If Shapley differs significantly from proportional attribution, it suggests")
print("that fixing certain stages has compounding effects on downstream outcomes.")
