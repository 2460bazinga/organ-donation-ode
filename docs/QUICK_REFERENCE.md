# Quick Reference: MSC Analysis Continuation

## 🎯 Current Status
**Phase 2 Complete** - MSC methodology designed, ready to execute and analyze

## 📂 Key Files

### On VM (`~/physionet.org/files/orchid/2.1.1/`)
- `OPOReferrals.csv` - Main dataset (133,101 records)
- All validation scripts already run

### Ready to Download
```bash
# Main analysis script (CURRENT)
wget https://files.manuscdn.com/user_upload_by_module/session_file/94078908/gqcPeNPFFazeKNpW.py -O msc_sensitivity_analysis.py

# Run it
python3 msc_sensitivity_analysis.py
```

## 🔬 The Core Problem

**Question:** Which referrals are "Medically Suitable Candidates" (MSCs)?

**Challenge:** Can't just learn from successful cases (survivor bias)

**Solution:** Three-layer hybrid approach:
1. **Layer 1:** Absolute contraindications (clinical guidelines)
2. **Layer 2:** DBD vs DCD pathways (age ceilings differ)
3. **Layer 3:** Empirical ranges (THREE approaches for sensitivity)

## 📊 Three Approaches to Layer 3

| Approach | Logic | Method | Use Case |
|----------|-------|--------|----------|
| **A: Absolute Max** | "If done once, it's viable" | Max observed age | Upper bound |
| **B: 99th Percentile** | "If top 1% can do it" | 99th percentile | Primary analysis |
| **C: Best OPO** | "If best OPO can do it" | Best OPO's 95th pct | Benchmark |

## 🎯 Expected Output

```
Approach              MSCs      MSC%    Sort%    Auth%  Overall%  Sort Loss%
-------------------- ---------- -------- -------- -------- ---------- ------------
Absolute Max          ~45,000   ~34%    ~18%     ~62%     ~8%        ~82%
Percentile 99         ~38,000   ~29%    ~20%     ~62%     ~7%        ~79%
Best Opo              ~33,000   ~25%    ~21%     ~62%     ~7%        ~76%
```

**Key Metric:** Sort Loss % should be >70% (validates ODE model)

## ✅ Next Steps

### Immediate (Phase 2 completion)
1. Run `msc_sensitivity_analysis.py` on VM
2. Review output table
3. Choose primary approach (likely B: 99th percentile)

### Phase 3: Robustness (4 layers)
1. **Stratified:** By year, OPO, age group, COD
2. **Bootstrap:** 1,000 resamples for confidence intervals
3. **Cross-validation:** Train/test splits
4. **Sensitivity:** Exclude OPO2, exclude 2015, etc.

### Phase 4: Shapley Decomposition
Calculate contribution of each stage to overall outcome

### Phase 5: Visualization
- Sankey diagram (flow through stages)
- OPO comparison charts
- Loss decomposition pie chart
- Temporal trends

### Phase 6: Documentation
- Final report (Markdown + PDF)
- All visualizations
- Code documentation

## 🚨 Critical Insights

### 1. Survivor Bias
**Problem:** Learning from successful cases codifies system inefficiency  
**Solution:** Use clinical guidelines + maximum observed (not central tendency)

### 2. Liver Age Ceiling
**DBD:** No effective ceiling (even 90-year-olds can donate)  
**DCD:** ~70 years (expanding over time)

### 3. National Benchmark
**Expected MSCs:** 25,345 - 30,414 (scaled for 6 OPOs × 7 years)  
**Don't force match** - use as validation check only

### 4. Referral Definition
**ORCHID "referral"** = Mechanically ventilated + referred by hospital  
**NOT** all deaths or all eligible deaths

## 📐 Loss Waterfall Formula

**Counterfactual Value Method:**

For each stage, calculate:
- Downstream success rate (if this stage succeeded)
- Multiply by number lost at this stage
- = Counterfactual transplants lost

**Sorting Loss CF** = (MSCs not approached) × (success rate if approached)  
**Authorization Loss CF** = (Approached not authorized) × (success rate if authorized)  
**Procurement Loss CF** = (Authorized not procured) × (success rate if procured)  
**Placement Loss** = Procured - Transplanted (actual, not counterfactual)

## 🔧 Technical Details

### VM Access
```bash
gcloud compute ssh orchid-vm --project=bold-case-478800-a1 --zone=us-central1-a
```

### Activate Environment
```bash
source ~/research-env/bin/activate
```

### Data Location
```bash
cd ~/physionet.org/files/orchid/2.1.1/
```

### File Transfer
```bash
# From VM to local
gcloud compute scp orchid-vm:~/file.csv . --project=bold-case-478800-a1 --zone=us-central1-a
```

## 📚 Key References

1. **ORCHID Dataset:** https://physionet.org/content/orchid/2.1.1/
2. **OPTN Policies:** https://optn.transplant.hrsa.gov/policies-bylaws/policies/
3. **ASTS DCD Guidelines:** Reich et al. (2009)
4. **Full Handoff:** See `HANDOFF_DOCUMENT.md`

## ⏱️ Timeline Estimate

- Phase 3 (Robustness): 2-3 hours
- Phase 4 (Shapley): 1 hour
- Phase 5 (Visualization): 2-3 hours
- Phase 6 (Documentation): 2-3 hours

**Total:** ~8-10 hours

## ✨ Success Criteria

1. **MSC identification is defensible** (avoids survivor bias)
2. **Sorting Loss % > 70%** (validates ODE model)
3. **Results are robust** (tight confidence intervals)
4. **OPO variance documented** (2x performance gap)

## 🤝 User Preferences

- **Rigorous methodology** over quick results
- **Sensitivity analysis** over single estimate
- **Transparency** about assumptions
- **Publication-ready** quality

---

**Read `HANDOFF_DOCUMENT.md` for complete details.**
