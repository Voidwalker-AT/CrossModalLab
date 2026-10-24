# Narrow causal protocol — approval required, not executed

Question: does a specified representation causally influence the contrast between contradicted and underdetermined cross-modal evidence? Status: **NO-GO for execution** until matched-baseline, adapter, alignment and resource checks pass and the user approves the exact budget. This document is not that approval.

Use the same-image, same-query No/U text-swap pair from the new development factorial design. Both are valid tasks; “clean/corrupted” denotes experimental source/recipient conditions, not a claim that the fork is defective evidence. Record baseline complete-sequence candidate scores first. Prefer a source/recipient pair with the expected opposed No-versus-U margin signs selected only on a dedicated development screen; report screening failures and reserve new families for later confirmation. The 12-forward pilot alone need not meet that criterion.

## Preconditions

- Validate actual decoder modules separately: Smol has 24 language layers, Qwen 28; never transfer a hook path or layer index by assumption. `src/models/model_adapter.py` defines an interface/spec, not a verified Qwen hook implementation.
- Existing `src/instrumentation/capture_integrity.py`, sparse multi-layer capture and `src/causal/patch_runner.py` supply reusable validation patterns. CPU-test a model-specific adapter before a new GPU intervention. The legacy hybrid causal scorer appends token IDs/masks without the repaired Qwen modality/position treatment; **do not route Qwen through it unvalidated**. This mission does not modify that frozen legacy path.
- Require image bytes/grids, image-token ranges, source/recipient sequence lengths, selected token strings, semantic roles, dtype and [2048] shape checks. Equal integer token indices alone are insufficient. The same word-order swap may tokenize differently. Reject an invalid alignment rather than truncating or borrowing a nearby token.
- Capture one sparse residual output row at a prespecified readout position. One 2048-element float32 row is 8 KiB; this is not a neuron trace. For Qwen, propose zero-based layer 13 (lower central layer of 28), chosen by architecture before intervention outcomes. Smol's development-selected layer-18 probe is not justification for Qwen's layer.

## Exact proposed first intervention budget

One model, one development family, one layer, one token selector. Six scoring conditions: unmodified source; unmodified recipient; recipient→recipient no-op; source→recipient replacement; an unrelated-token matched-shape control; and a norm-matched deterministic shuffled-coordinate control. Three candidate forwards per condition = **18 scoring forwards**, plus two separately counted passive capture forwards = **20 total**. The passive captures must be checked for unchanged outputs against their unhooked equivalents; if that entails extra forwards, revise the budget before approval, never add them silently. No head scan, MLP scan, gradient attribution or multi-layer sweep is included.

Primary outcome is the raw log-likelihood contrast `D = s(No) − s(Undetermined)` and intervention change `ΔD = D_patched − D_recipient`. Report all three candidate scores, not only a favorable label flip. For No→U-recipient replacement, the proposed directional hypothesis is positive ΔD; it is a hypothesis, not a verified mechanism. Avoid normalized indirect effect as the primary endpoint because a small source–recipient denominator is unstable; report any ratio with the denominator and a prespecified numerical eligibility rule based on score-repeat precision, not chosen after seeing effects.

## Interpretation and expansion

An observed patch effect can show sensitivity to that intervention at that site. It does not establish a unique necessary/sufficient circuit, a neuron encoding evidence sufficiency, or model-wide generalization. Targeted changes must be compared directly with no-op, unrelated-token and shuffled controls; the old Phase 2B unrelated-token control exceeded the targeted mean absolute effect and cannot be counted as positive support. Retain all failures and missing records.

One family is a feasibility case, with raw values and no population confidence claim. An approved expansion needs independent development/confirmation families, paired family bootstrap B≥2,000, and preregistered contrasts/multiplicity correction; alternative-path and multi-component tests are needed for stronger claims. Estimate the 20-forward cost using newly measured per-model timings and load/audit overhead; a central-time multiplication alone is insufficient. Do not execute this protocol from the current behavioral allowance.
