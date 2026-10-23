# Phase 2I-B numerical invariance and telemetry repair — 2026-10-23

**GO for auditing the unfinished ablation runner.** This is not authorization
for training. The single bounded real-model verification passed; no optimizer
was created or stepped. The unfinished runner was neither edited nor executed.

## Root cause and scoped repair

The previous verifier treated raw-byte hash equality as numerical invariance.
Installed PEFT k-bit preparation upcasts eligible BF16 parameters to FP32;
installed bitsandbytes `Linear4bit.forward` lazily casts bias storage to its
compute dtype. The previous run flagged 104 frozen vision/projector biases.
The current run confirms that all 104 changed FP32-to-BF16 storage without
any numerical difference. A raw hash alone could not make that distinction.

Changes are limited to the verifier, its reusable verification helper, CPU
regressions, and new reports/evidence. The scorer, losses, CE baseline,
checkpoint loaders, pinned revisions, dependency specifications, historical
evidence, and unfinished ablation runner were preserved.

## Numerical methodology

`src/training/scorer_verification.py` records a private temporary raw baseline
for each named parameter, including name, original dtype, shape, trainability,
finite-value status, and storage hash. It copies at most 262,144 elements per
chunk from GPU to CPU and stores the complete baseline on disk inside the
workspace, avoiding a full CPU or GPU model clone. Each float64 comparison
array has at most 2 MiB of elements; several bounded working arrays are used.
Only the audit's exclusively created temporary directory is cleaned up.

The comparison checks the same named parameter set and shapes, final dtype,
storage hashes, numerical equality, maximum absolute difference, relative
difference over nonzero references, zero-reference changes, trainability, and
unexpected frozen gradients. Non-finite values fail, including an unverifiable
non-finite baseline. Float comparisons use float64, which exactly represents
FP16/BF16/FP32 values; integer equality remains exact and native. Int64
differences use Python integers to avoid subtraction overflow or loss of bits.

There is **no universal tolerance**. Exact numerical equality is required for
unchanged values. A rounding change may pass only for an explicitly named
bitsandbytes bias dtype transition and exact equality to `before.to(final_dtype)`.
An additional mutation beyond that cast fails. Classification distinguishes:

- `EXACT_NUMERICALLY_UNCHANGED`
- `DTYPE_CHANGED_NUMERICALLY_EQUIVALENT`
- `DTYPE_CHANGED_WITH_EXPECTED_ROUNDING`
- `NUMERICALLY_CHANGED`
- `UNVERIFIABLE`

The real-model result required **no rounding exception**: every parameter
was numerically equal, with maximum absolute difference **0.0**.

## One bounded GPU execution

Executed exactly once after focused CPU tests passed:

```powershell
ulvo run . -- python -u scripts/verify_phase_2i_b_scorer.py --gpu
```

Temporary/cache environment variables were redirected inside the workspace;
Hugging Face access was offline and the existing pinned checkpoint was reused.
The verifier records seed 20261023. Preflight ULVO doctor and CUDA computation
passed; available host RAM was 5.32 GiB in preliminary Python preflight, above
the mandatory 4 GiB. No competing model worker was observed. Desktop/browser
GPU processes were present; WDDM does not provide their individual VRAM totals.

Model: Qwen/Qwen3-VL-2B-Instruct, revision
`89644892e4d85e24eaac8bacfd4f463576704203`, NF4 with BF16 compute.
Example: train-split `xc_id_0001::entailed`; no final holdout was used.

| Candidate | Token IDs | Complete summed log-likelihood |
|---|---|---:|
| Yes | 7414 | -15.145495414733887 |
| No | 2308 | -15.751511573791504 |
| Undetermined | 16887, 46637 | -15.693758964538574 |

All three complete-answer forwards and finite-score checks passed. Actual
installed Qwen positional computation also passed for the complete modality
sequences; the two-token Undetermined continuation was scored in full. These
scores are engineering observations on one example, not accuracy estimates,
calibrated probabilities, or evidence of a beneficial training objective.

## Parameter and gradient results

| Check | Observed result |
|---|---|
| All parameter tensors checked | 849 |
| Frozen tensors checked | 625 |
| Frozen tensors, same dtype and identical values | 521 |
| Frozen tensors, FP32-to-BF16 and identical values | 104 |
| Trainable LoRA tensors, same dtype and identical values | 224 |
| Expected-rounding classifications | 0 |
| Numerical changes / unverifiable tensors | 0 / 0 |
| Maximum absolute difference over all parameters | 0.0 |
| Trainable parameter elements | 802,816 |
| LoRA gradient tensors | 224, all finite, nonzero signal present |
| Frozen parameters with gradients | 0 |
| Optimizer steps / parameter updates | 0 / 0 |

The result establishes numerical invariance across these evaluated forwards
and backward pass, while recording the actual dtype/storage changes. It does
not establish invariance for every possible input or future dependency version.

## Telemetry and cleanup

`run_observed` samples resources during execution and records final telemetry
in a finally path on success or failure, preserving the original exception
even when sampling or cleanup also fails. Assertion/CUDA exceptions are stored
by type, message, and traceback. Cleanup status and cleanup errors are separate.
Host commit usage is measured through Windows `GetPerformanceInfo`; unavailable
measurements are explicitly labeled on other platforms.

PyTorch peak counters were reset immediately before NF4 loading, covering model
loading, snapshot capture, scoring, backward, and comparison. The NVIDIA and
host measurements below are samples, not continuous extrema.

| Measurement | Observed value |
|---|---:|
| Internal elapsed runtime | 82.8979 seconds |
| Peak PyTorch allocated VRAM | 2709.690 MiB |
| Peak PyTorch reserved VRAM | 2776.000 MiB |
| Maximum sampled NVIDIA physical usage | 3193 MiB |
| Minimum sampled available host RAM | 4400.895 MiB |
| Maximum sampled system commit usage | 20,566,044,672 bytes |
| Sampling errors | 0 |
| Cleanup | PASSED |

The verifier enforces the existing resource policy of 1500 MiB host headroom
and 3800 MiB maximum PyTorch allocation, requires reserved memory and sampled
physical usage to stay within 4096 MiB, and caps PyTorch allocation below the
dedicated VRAM ceiling. All observed gates passed. Continuous physical-memory
extrema between samples remain UNVERIFIED. The snapshot directory was confirmed
absent after cleanup, and ULVO returned to READY with zero active leases.

## Fresh validation and preservation

- Focused new CPU regressions: **12 passed**, covering reversible conversion,
  explicitly permitted rounding, unapproved rounding, genuine and one-ULP
  mutations, NaN/Inf, non-finite baselines, frozen gradients, valid trainable
  gradients without updates, int64 precision, changed shapes/parameter sets,
  failure telemetry, cleanup exceptions, and historical-artifact preservation.
- Full ULVO suite: **255 tests, OK**, no skips; 7.842 seconds internally.
- Standard Python suite: **255 tests, OK**, 69 optional-dependency skips.
- Portability script: PASSED.
- Compileall over `src`, `scripts`, and `tests`: exit 0.
- ULVO project doctor after verification: PASSED, READY, zero active leases.
- Hash comparison: **14 pre-existing files unchanged**, including historical
  reports/evidence, all three requirements files, and the unfinished runner.
- No installation, model download, optimizer step, training campaign, holdout
  evaluation, causal patching, dashboard work, or destructive Git operation.

Synthetic fixtures are marked `is_synthetic_demo = True` and watermarked
SYNTHETIC DEMO; their results are quarantined from empirical benchmark metrics.

## Evidence and next decision

New machine-readable evidence:
`artifacts/training/phase_2i_b_recovery/gpu_1b71a5485dff4fd1a0e8a95a442f789d.json`.
It contains per-parameter records, candidate IDs/scores, seed, installed
versions, gradient validation, resource samples, measured allocator peaks,
cleanup status, and explicit no-optimizer/holdout fields.
The paired `gpu_numerical_execution_8f82ab7147664c2c84254b6dca6d064f` execution
log preserves exact stdout/stderr. The matching `.txt` copy is byte-identical
and is included in Git; the original `.log` is also retained locally.

The next authorized research-engineering decision is to audit and repair the
unfinished runner's checkpoint/resume and provenance behavior. Its safety is
still UNVERIFIED; this task did not run or repair it. Training and subsequent
research phases require their own prerequisites and explicit authorization.
ES efficacy, scientific novelty, and a causal failure mechanism remain
UNVERIFIED. Stop after this verifier repair, bounded run, and controlled push.
