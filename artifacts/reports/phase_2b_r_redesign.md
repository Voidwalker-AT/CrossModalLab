# Phase 2B-R Report — Causal Patching Redesign (No GPU Run)

**Branch:** `feature/phase-2b-recovery-audit` → integrated to `main`  
**Historical JSONL SHA preserved:** `50bb225b…e337` (16 records unchanged)

---

## 1. Recovery branch review
- Synced with origin; 1 commit ahead of main (`d41cce6`).
- Clean tree; 155 tests green before redesign additions.
- Scientific status remains **exploratory / post-hoc modified**.

## 2. Exact counterfactual pairs
- New classifier: `EXACT_IMAGE_CHANGED_TEXT` / `EXACT_TEXT_CHANGED_IMAGE` / `MULTI_FACTOR`.
- Generated `datasets/exact_pairs_v1/`: **3 families × 2 records**, **identical image_sha256** per family (Yes ↔ Undetermined).
- Jittered contrast-v2 pairs are **not** eligible for primary causal attribution.

## 3. Candidate scoring corrections
- `score_complete_candidates` keeps `residual_patch_context` active for **all** candidate re-forwards.
- Guards `token_position ∈ [0, prompt_len)` so continuation positions cannot be silently patched as prompt-final.

## 4. Hook / token alignment
- Hook lifetime unit-tested (N applications while context open).
- No-op equivalence previously measured max|Δlogits|=0 on engineering gate.

## 5. Memory telemetry redesign
- `src/causal/telemetry.py`: allocated/reserved/max_* + nvidia-smi + host RAM; **reset peaks at documented boundary**; stop if allocated>3800 MiB or host <1500 MiB.
- Clarifies that >4096 MiB torch max is not fabrication (WDDM shared).

## 6. Reduced pilot estimate (not executed)
- **1 family**, EXACT_IMAGE_CHANGED_TEXT pair, L18 prompt-final.
- Conditions: baselines + no-op + source_to_target + same_label + 1 negative.
- **≤12 forwards**, **≤20 min**, stop at 3800 MiB allocated.
- Spec: `configs/phase_2b_r/pilot_exec_spec_v1.json` (`execution_authorized: false`).

## 7. Tests
| Command | Exit |
|---|---|
| ULVO unittest | **161 OK** |
| System Python | **161 OK (10 skips)** |
| doctor / portability / compileall | **0** |

## 8. Git
See SHAs below (feature push + main integration). No force-push.

## 9. Limitations
Exact-image pilot still small; causal uniqueness not implied; 2B historical matrix incomplete and control-dominated.

## 10. Recommendation
**CONDITIONAL GO** for **one-pair** GPU pilot after explicit authorization, using exact pairs + new telemetry + scoring context.  
**NO-GO** for full matrix or confirmatory claims.

**Stopped after redesign. No GPU patching executed in 2B-R.**
