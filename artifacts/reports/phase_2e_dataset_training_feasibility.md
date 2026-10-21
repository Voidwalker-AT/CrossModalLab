# Phase 2E Report — Dataset Strategy & Fine-Tuning Feasibility

**No dataset downloads. No model training. No holdout use.**

---

## 1. Dataset shortlist & licenses
See `docs/phase_2e_dataset_strategy.md`: Winoground (gated, 366 MB, CVPR’22), GQA (CC BY 4.0), HallusionBench (Apache-2.0, onboarded), MoHoBench (MIT, onboarded), VizWiz (ToU check), Visual Genome (full set infeasible), ARO (eval).  
**In-house Tier A** remains the controlled training backbone.

## 2. Storage & annotation budget
5k / 20k / 50k ≈ **0.3–0.5 / 1.3–2 / 3.5–5 GB**; Tier A annotation cost **0**; Tier B/C prefer official labels.

## 3. Novelty collision matrix
`docs/phase_2e_novelty_audit.md`: LoRA+abstention **not novel alone**; formal chain/fork + exact-image CF + modality ablations = **plausible gap** (not established).

## 4. QLoRA feasibility
NF4 **inference** does **not** imply QLoRA **training** fits 4 GB.  
**Verdict:** full multimodal 1500-token QLoRA **UNVERIFIED / likely OOM**; first safe probe = **short text-only / short-image LoRA** with checkpointing after explicit authorization.

## 5. Candidate losses
CE; evidence-aware CE/margin; counterfactual consistency; combined with \(\lambda\) grid — defined in `docs/phase_2e_training_feasibility.md` + `src/training/losses.py`.

## 6–7. Baselines & experimental design
B0–B6 arms; balanced accuracy, abstention P/R, CF consistency, falsification thresholds (e.g. fork gain without >5 pp supported-BA drop).

## 8. Tests & verification
| | |
|---|---|
| Schema audit contrast-v2 | **PASS** (720 records, splits/CF/balance/holdout OK) |
| ULVO unittest | **183 OK** |
| System Python | **183 OK (16 skips)** |
| portability / compileall | **0** |

## 9. Git
See push SHA.

## 10. Recommended implementation path (await approval)
1. Freeze **5k Tier A** generator settings (same solver).  
2. **Text-only LoRA feasibility** experiment (authorized later) before multimodal QLoRA.  
3. Only then consider B2–B5 arms with 2D neutral prompt as frozen base protocol.  
4. Refresh novelty search before paper claims.

**Stopped after Phase 2E feasibility & design.**
