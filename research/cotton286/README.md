# AgriVision — Cotton 286-Class Research Expansion

> **RESEARCH EXPERIMENT REPOSITORY SANDBOX**  
> **STATUS**: Research Candidate Evaluation  
> **PRODUCTION FIREWALL**: IMMUTABLE & RELEASE FROZEN  
> **DATE**: September 2026  

---

## Executive Summary

This research package evaluates whether the historical `crop-diseases-model` (EfficientNet-B5 + CBAM) can be extended from **281 disease classes** to **286 classes** by adding five Cotton pathology/health conditions:
1. `Cotton - Aphids` (Index 281)
2. `Cotton - Bacterial Blight` (Index 282)
3. `Cotton - Healthy` (Index 283)
4. `Cotton - Powdery Mildew` (Index 284)
5. `Cotton - Target Spot` (Index 285)

### Strict Production Firewall Invariants
- **Production Model A** (`weights/efficientnet_b5_cbam_best.pt`): `b4db2209bfc416716d55339d0c21cb0c2d62ebe0eee089d6d00712dd198717b7` — **UNCHANGED & RELEASE FROZEN**.
- **Production Campaign 2** (`weights/crop_ood_expert_campaign2.pt`): `d71680b970be84f77eeb77b81118874ba2a2fd362790ed5e2d25b6e34a515f9b` — **UNCHANGED & RELEASE FROZEN**.
- All research outputs are confined to `research/cotton286/` and `weights/research/`.

---

## Manifest & File Structure

```
research/cotton286/
├── README.md                      # This overview document
├── dataset_audit.json             # Comprehensive audit of Cotton_Leaves source images
├── cotton_split_manifest.json     # 70% Train / 15% Dev / 15% Sealed Test split
├── rehearsal_manifest.json        # 281-class rehearsal data provenance & sampling
├── class_expansion_audit.json     # Bit-for-bit weight preservation audit (0..280)
├── init_286.pt                    # 286-class initialized checkpoint prior to training
├── training_config.json           # Exact hyperparameters, LRs, and loss formulation
├── training_metrics_seed42.json   # Seed 42 training history & validation curve
├── training_metrics_seed43.json   # Seed 43 training history & validation curve
├── cotton_metrics.json            # Locked Sealed Test evaluation & confusion matrix
├── old281_regression.json         # 281-class regression benchmark
├── distillation_metrics.json      # Teacher-student logit drift & agreement
├── ood_regression.json            # OOD Energy and MSP score regression
├── cotton_negative_test.json      # Broadleaf non-cotton specificity test
├── api_research_test.json         # Local API contract & safety invariant checks
├── checkpoint_manifest.json       # SHA-256 hashes of all candidate checkpoints
├── final_decision.json            # Machine-readable verdict
└── FINAL_REPORT.md                # Full scientific walkthrough & decision
```

---

## Training Objective

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{new\_class}} + \lambda_{\text{rehearsal}} \cdot \mathcal{L}_{\text{old\_class}} + \lambda_{\text{distill}} \cdot \mathcal{L}_{\text{distill}}$$

- **$\mathcal{L}_{\text{new\_class}}$**: Cross-entropy with label smoothing ($\epsilon = 0.05$) on the 5 Cotton classes.
- **$\mathcal{L}_{\text{old\_class}}$**: Cross-entropy on rehearsal samples across original classes.
- **$\mathcal{L}_{\text{distill}}$**: Knowledge distillation from the frozen 281-class teacher model ($T = 2.0$) over logits $0..280$.

---

## Decision Taxonomy

The final verdict must be exactly one of:
- `COTTON_EXPANSION_SUCCESS`
- `COTTON_EXPANSION_FAILED`
- `COTTON_EXPANSION_REQUIRES_MORE_DATA`
- `COTTON_EXPANSION_INVALID`
