# Phase 2A — Causal Intervention Design & Pre-Registration

**Status:** DESIGN ONLY — **no scientific activation patching executed**  
**Model:** SmolVLM2-2.2B-Instruct @ `482adb537c021c86670beed01cd58990d01e72e4` (NF4)

---

## 1. Probe validity audit (Phase 1F)

| Item | Value |
|---|---|
| Reproduced L18_last BA | **0.778** |
| Group-bootstrap CI | **[0.593, 0.963]** |
| Validation families | **9** |
| Probe specs evaluated | **13** |
| L2 grid evaluations | **52** |
| Total selection events | **65** |
| Selection optimism | **HIGH** (best-of-many on development val) |
| Final holdout | **Untouched** |

**Do not treat L18 as a proven logical representation.** Exploratory decode only.

## 2. Causal research questions

See `configs/phase_2a/causal_preregistration_v1.json` for null/alt/outcomes/falsification:

- **RQ-C1:** Intervention changes answer margins?  
- **RQ-C2:** Effect specific vs generic disruption?  
- **RQ-C3:** Late vs early layers?  
- **RQ-C4:** Sufficiency gains without damaging correct definite answers?  

## 3. Counterfactual pairs (development)

Contrast v2 families provide:

| Pair type | Examples | Single-variable? |
|---|---|---|
| Same image, changed text | entailed ↔ underdetermined | **Yes** |
| Same text, changed image | contradicted ↔ underdetermined | **Yes** |
| Multi-factor | entailed ↔ contradicted | **No** — excluded from single-variable claims |

Audit: `artifacts/causal/phase_2a/counterfactual_pair_audit_v1.json`  
Token rule: patch **last position only if** `prompt_token_length` matches across runs; else abort.

## 4. Intervention definitions

Post-block residual replacement at layers **3/12/18/23**, primary **L18 last token**.

Directions: `clean_to_cf`, `cf_to_clean`, `no_op`, `random_source`, `same_label`, `unrelated_token`.

CPU validators in `src/causal/interventions.py` (synthetic tensors only).

## 5–6. Outcomes & stats

Primary: **raw margin change** on complete candidates ` Yes` / ` No` / ` Undetermined`.  
Secondary: target LL change, answer flips, control effects, damage on correct answers.  
NIE only with \|baseline margin\| ≥ 1e-6; always report raw.  
Independent unit: **family**; group bootstrap; sign-flips; BH if multiple confirmatory contrasts.  
Confirmatory threshold: **≥30 sealed families** (current 9 = exploratory).

## 7. Hardware feasibility

VRAM ~3331 MiB; sequential clean/cf forwards; single model copy; selected-row CPU offload; bounded pilot ≤9 families × ~8 forwards ≈ **72** max.

## 8. CPU engineering tests

`tests/test_phase_2a_interventions.py` — replacement, shapes, no-op, directions, hashes, non-finite rejection. **Not** scientific results.

## 9. Pre-registration

Frozen: `configs/phase_2a/causal_preregistration_v1.json`  
`execution_authorized: false` until explicit approval.

## 10. Recommendation

**CONDITIONAL GO** to **design-complete** state for a future bounded exploratory causal pilot (≤9 dev families), still **NO-GO** for confirmatory claims or holdout evaluation without a larger sealed family set and user authorization.

**Stopped after Phase 2A.**
