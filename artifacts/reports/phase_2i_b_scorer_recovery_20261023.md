# Phase 2I-B scorer recovery — 2026-10-23

Status: **CPU-VERIFIED repair; GPU verification BLOCKED by the mandatory RAM gate.**
No optimizer step, training campaign, final-holdout evaluation, package change,
checkpoint modification, or model download was performed.

## Repository preflight

The starting branch was `feature/phase-2b-recovery-audit`, at
`505d803d6cc2f25bb18310301f9c272ae41a0a12`. After fetching origin, HEAD,
origin/main, and the original tracking branch had zero commit divergence.
The recovery is isolated on `codex/repair-qwen-candidate-scoring`.

Four draft paths were untracked at arrival:

- `src/training/differentiable_scoring.py`
- `tests/test_phase_2i_b_es.py`
- `configs/phase_2i_b/es_ablation_config_v1.json`
- `scripts/run_phase_2i_b_es_ablation.py`

The scorer and tests were repaired; the frozen configuration was preserved.
The unfinished training runner is preserved unmodified and excluded from the
recovery commit. The verification script uses the tracked CE runner's LoRA
setup and does not depend on the unfinished ablation runner.

## Verified inventory and scientific status

Source inspection confirmed graph solving/generation/verification under
`src/symbolic`, passive and multilayer capture under `src/instrumentation`,
linear probes/confound helpers under `src/probes`, patching/pair classification/
telemetry under `src/causal`, candidate evaluation under `src/evaluation`, and
LoRA, sampling, masking, parsing, and loss utilities under `src/training`.
Both pinned model loaders and portable path/configuration modules exist.
Their presence and passing CPU tests do not establish empirical effectiveness.

The saved Phase 2I-A metrics were independently recomputed from the 18 raw
final completions using the existing parser: balanced accuracy 0.666667,
recalls Yes=1, No=1, Undetermined=0, invalid rate=0. The training log contains
50 rows from 50 unique families, with 1- or 2-token supervision. This is an
audit of stored artifacts, not a rerun of training or model evaluation.

**Output-quality caveat:** zero of the 18 saved raw completions is exactly a
clean Yes/No/Undetermined label after stripping whitespace. The existing
parser accepts leading labels with trailing text, including mojibake-like
suffixes in these artifacts. Its zero invalid rate therefore does not imply
clean generation. The parser and historical results were not modified.

UNVERIFIED in this recovery: historical probe/cross-model/causal measurements
were not independently rerun. No unique circuit, general failure mechanism,
loss novelty, ES effectiveness, independent-test improvement, or deployment
reliability is established by this repair. Reused development validation,
small sample size, and model-selection optimism remain scientific risks.

## Reproduced root cause

The installed Transformers 5.20.1 Qwen forward explicitly accepts
`mm_token_type_ids`. Its `get_rope_index` indexes that tensor with the attention
mask. The original scorer appended candidate IDs and extended the attention
mask, but left modality IDs at prompt length.

`ulvo run . -- python -u scripts/verify_phase_2i_b_scorer.py` failed before the
repair inside the actual installed Qwen positional code. It used the local
pinned processor and train-split example `xc_id_0001::entailed`, without model
weights or a final-holdout sample:

| Field | Prompt | Original first-candidate forward | Repaired forward |
|---|---|---|---|
| input_ids | [1,224] | [1,225] | [1,225] |
| attention_mask | [1,224] | [1,225] | [1,225] |
| mm_token_type_ids | [1,224] | [1,224] | [1,225] |
| pixel_values | [576,1536] | unchanged | unchanged |
| image_grid_thw | [1,3], value [1,24,24] | unchanged | unchanged |

Neither ordinary `token_type_ids` nor explicit `position_ids` was emitted.
Candidate IDs were Yes=[7414], No=[2308], Undetermined=[16887,46637].
Spatial merge size is 2; image token ID is 151655; text hidden size is 2048.

Observed exception: mask [225] versus modality tensor [224] at
`input_token_type[attention_mask[batch_idx].bool()]`. This reproduces the
reported 229/228 failure mechanism on a different development prompt.
The regression test separately exercises a 228-token prompt.

## Minimal repair and mathematical meaning

Candidate continuations append exactly once. Attention-mask prefixes are
preserved and extended with ones; modality/type prefixes are preserved and
extended with text zeros. Images and grids remain unchanged. Qwen recomputes
M-RoPE for the complete sequence rather than receiving stale explicit positions.
For text-only explicit positions, positions extend consecutively. Cached,
pre-embedded, supervised, truncated-logit, empty, or malformed inputs are rejected.
KV caching is disabled for these complete differentiable forwards.

The score remains the unnormalized sequence sum
`s(c|x) = sum_k log p(c_k | x, c_<k)`, with token k read from
logit position `prompt_length - 1 + k`. There is no length normalization,
answer-token duplication, label/oracle injection, detach, or CE substitution.
The existing ES hinge and frozen coefficient/margin are unchanged.

## Fresh verification

| Check | Observed result |
|---|---|
| ULVO project doctor | PASSED |
| Full ULVO unittest discovery after final code edits | 243 tests, OK, no skips |
| Standard Python 3.13 unittest discovery | 243 tests, OK, 57 optional-dependency skips |
| Portability script through ULVO | PASSED |
| Compileall over src/scripts/tests | exit 0 |
| Reproduction command after repair | exit 0; actual Qwen positional computation passes all candidates |

The 17 Phase 2I-B tests cover complete likelihoods against an independent
token-at-a-time oracle; causal-shift gradients; single/multitoken candidates;
the original Qwen mask mismatch; mask/type/position alignment; unchanged
images, grids, and inputs; no duplicate tokens; invalid input rejection;
frozen fixture weights after backward; and the preserved ES objective.
Fixtures are explicitly synthetic demonstrations, not empirical results.

Installed versions verified: Python 3.11.16, torch 2.6.0+cu124,
torchvision 0.21.0+cu124, Transformers 5.20.1, bitsandbytes 0.50.2,
PEFT 0.21.2. A separate live 16-by-16 CUDA matrix multiplication returned the
expected sum 4096.0 on the RTX 2050; dedicated VRAM is 4096 MiB.
The GPU verifier requires 4 GiB available host RAM and caps PyTorch
allocation below dedicated VRAM. It never creates an optimizer.

`--gpu` stopped before processor/model loading at **1.426582 GiB** available
host RAM. Real NF4 likelihoods, LoRA gradients, parameter-hash invariance, and
peak GPU memory are **NOT TESTED / UNVERIFIED**. CPU positional validation
cannot substitute for that check. No attempt was made to free other processes'
memory or bypass the gate. GPU runtime/memory cost remains UNVERIFIED.
The final processor/position check took about 0.50 seconds internally;
the full ULVO CPU suite took 6.70 seconds, excluding gateway/import overhead.

Machine-readable evidence under `artifacts/training/phase_2i_b_recovery/`:

- `positions_ee7bc1782aa74f0592ecd05d2db01ce4.json`: original failure and traceback.
- `positions_593335f502de414c966e165482a9d07d.json`: final repaired metadata check.
- `gpu_c99cbdfb8d3e4fccbc79d2c439e7d9e8.json`: RAM gate failure before loading.

## Next decision

Free at least 4 GiB of host RAM, then run the bounded `--gpu` verification
with temporary/cache writes redirected inside the workspace and offline
model access. The unexercised GPU path records scores, gradients, exact
parameter hashes, and memory; it performs no optimizer step.
Only after it passes should a separate decision authorize training.
The draft ablation runner also needs resume/provenance review: its existing
`completed_steps` bookkeeping does not restore adapter/optimizer state, and
its missing-checkpoint path can clear an existing metrics log. These paths
were inspected but not executed or changed during scorer recovery.

**STOP:** no Phase 2I-B ablation or subsequent research phase is authorized
by this engineering recovery result.
