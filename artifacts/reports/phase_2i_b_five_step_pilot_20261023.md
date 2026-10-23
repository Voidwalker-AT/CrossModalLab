# Phase 2I-B five-step pilot — verified engineering, preliminary science

Date: 2026-10-23. Authorizing mission: maximum five actual optimizer steps, safe main integration, manuscript kickoff. No additional training is authorized by this report.

## Identity and preflight

Recovery branch started at `295d7a1df5d9d29d3778446bf3d548d22e1ddf5b`. Fetched origin/main was `505d803d6cc2f25bb18310301f9c272ae41a0a12`, an ancestor with no divergent partner commits. Preexisting untracked failure evidence was preserved. The read-only runner plan passed before execution: 50 scheduled train cases/families; 18 validation cases/six families; no model load or optimizer steps in planning. Full plan, first-five IDs/labels/image and prompt hashes, runtime, checkpoint file sizes, and history hashes are in `artifacts/training/phase_2i_b_pilot_20261023/preflight_and_plan.json`.

Available physical host RAM at recorded preflight was 4,475,617,280 bytes (4.168 GiB), above the 4 GiB gate. CUDA matrix multiplication passed; RTX 2050 dedicated VRAM is reported by NVIDIA as 4096 MiB. Desktop graphics clients were present, with no separate compute workload listed; none were terminated. The actual runner repeats the RAM/CUDA gate before loading.

Pinned Qwen3-VL-2B revision: `89644892e4d85e24eaac8bacfd4f463576704203`. Existing local files only, offline mode, NF4/double quantization/BF16 compute. Runtime: Python 3.11.16, torch 2.6.0+cu124, transformers 5.20.1, PEFT 0.21.2, bitsandbytes 0.50.2, NumPy 2.4.6, safetensors 0.8.0. No packages or models were installed/downloaded.

Run ID: `56e277e421e442b08d33b107fc934494`.
Identity hash: `800898e1515a7f828cd5a94f2aac9651a5a7a0bad14934bdeb7817d3433525fc`.
Run directory: `artifacts/training/phase_2i_b/runs/56e277e421e442b08d33b107fc934494`.

Executed once, exit 0:

```sh
ulvo run . -- python scripts/run_phase_2i_b_es_ablation.py --execute-training --new-run --stop-after 5
```

Exact output is retained in `artifacts/training/phase_2i_b_pilot_20261023/execution_output.txt`. Final status is PAUSED at step five, not a completed 50-step experiment. Batch/accumulation=1; language q/k/v/o LoRA r=2, alpha=4; frozen vision/projector/base; AdamW lr=1e-4, betas=.9/.999, eps=1e-8, weight decay=.01; no scheduler. Objective remains gold-span CE + .25 ES, margin 1, full unnormalized candidate-sequence log-likelihoods. No tuning or objective substitution occurred.

## Exact trajectory

| Step / family / label | CE | ES | Total | Gradient L2 / tensors | Gold tokens | Step seconds | Peak allocated / reserved MiB | NVIDIA sample / available RAM MiB | Checkpoint |
|---|---:|---:|---:|---|---:|---:|---|---|---|
| 1 / xc_id_0094 / Yes | 14.891876 | 0 | 14.891876 | 17.180753 / 224 | 1 | 4.7116 | 3015.963 / 3232 | 3722 / 2923.047 | step_0001 published |
| 2 / xc_id_0069 / No | 16.306824 | 1.114895 | 16.585548 | 16.114928 / 224 | 1 | 4.6998 | 3033.665 / 3176 | 3398 / 3195.434 | step_0002 published |
| 3 / xc_id_0001 / Undetermined | 7.565945 | 0 | 7.565945 | 11.032014 / 224 | 2 | 4.2458 | 3019.691 / 3304 | 3782 / 3184.184 | step_0003 published |
| 4 / xc_id_0160 / Yes | 16.695772 | 3.702376 | 17.621367 | 33.806626 / 224 | 1 | 4.1869 | 3024.848 / 3178 | 3675 / 3203.602 | step_0004 published |
| 5 / xc_id_0156 / No | 17.152073 | 2.037066 | 17.661339 | 23.537223 / 224 | 1 | 4.4211 | 3018.531 / 3304 | 3263 / 3235.426 | step_0005 published |

The journal retains full precision and all three candidate LLs per step. Exactly five optimizer updates are corroborated by five unique contiguous metrics and 224 AdamW slots with matching step counters in each committed checkpoint. These are different examples; a varying trajectory is not a fixed-input learning curve.

## Parameter, resource and saved-state acceptance

All five steps had finite losses, finite nonzero aggregate LoRA gradients, 224 gradient tensors, and no frozen gradients. Chunked disk-backed comparisons found **625 exact numerically unchanged frozen tensors and 224 numerically changed finite LoRA tensors at every step**, relative to the captured post-baseline initialization. No dtype changes needed acceptance in this pilot. Earlier verifier evidence separately explains benign bitsandbytes bias conversions. The original comparison’s top-level `passed:false` means that not *all* tensors stayed unchanged (LoRA correctly changed); the training-specific `acceptance.passed:true` is the applicable criterion and explicitly requires those updates. Per-tensor evidence is retained unchanged.

There were 265 resource samples (nominal one-second sampling, actual intervals include query cost). Complete execution took 286.972 seconds, including load, 72 validation generations across pre/post prompt conditions, five updates, checkpoint writes and audits. Summed step timing was 22.2652 seconds. Maximum sampled NVIDIA usage was 3795 MiB; minimum sampled available RAM was 2912.504 MiB; maximum sampled system commit was 23,727,001,600 bytes. Highest step torch allocation was 3033.665 MiB and reserved allocation 3304 MiB. No sampling error or safety-gate failure occurred. Sampling cannot certify an unseen instantaneous physical peak. NVIDIA usage returned to 409 MiB after process exit; no worker was terminated.

Read-only `verify_saved_pilot.py` deserialized/validated all six checkpoints (0–5), all adapter values and hashes, the optimizer layouts/hyperparameters/moments/counters, and Python/NumPy/CPU/CUDA RNG structures. Journal/hash chain passes, with no incomplete checkpoint, duplicate, missing state, or contradictory metric. Initial optimizer slots are empty; each trained checkpoint has 224 valid slots. No extra optimizer step or real-model resume was used for this check. Saved CPU tests exercise restoration; exact real-model resumed-trajectory equivalence remains UNVERIFIED.

Forty files in the preflight history inventory remained byte-for-byte unchanged, including CE and scorer/runner recovery evidence. A separate final check also verified all 133 historical hashes recorded by the earlier runner-recovery audit. The final-holdout images were not opened or evaluated. The manifest is hashed/read for development selection, whose records alone reach model inputs. Binary adapter/optimizer/RNG files are retained locally and excluded from Git; metadata and raw completions remain reviewable.

## Development comparison with corrected CE

Same pinned model, frozen sample-prefix IDs/labels, source hashes, 18 validation IDs/six families, historical parser and greedy 16-token generation. The regenerated frozen baseline matches CE’s stored baseline metrics.

| Condition | BA | Recall Yes / No / U | Invalid | Predicted U | Exact-label completions |
|---|---:|---|---:|---:|---:|
| Frozen, standard | .388889 | 1/6, 4/6, 2/6 | 0/18 | 6/18 | 18/18 |
| CE step five | .388889 | 3/6, 4/6, 0/6 | 6/18 | 0/18 | 0/18 |
| CE+ES step five | .500000 | 3/6, 4/6, 2/6 | 0/18 | 6/18 | 18/18 |
| Frozen, explicit abstention | .444444 | 2/6, 4/6, 2/6 | 0/18 | 6/18 | 18/18 |
| CE+ES step five, explicit abstention | .444444 | 2/6, 4/6, 2/6 | 0/18 | 6/18 | 18/18 |

In the pilot standard condition four of six U predictions are on sufficient examples. Four of six actual underdetermined examples still fail. The CE step-five explicit-abstention evaluation is unavailable; it is not inferred. BA difference CE+ES minus CE is .111111 (two cases), paired family bootstrap 95% descriptive interval [0,.222222], 2000 draws/seed20261023. This is not a confirmatory significance result or a generalization claim. Small repeatedly inspected development families, malformed CE outputs accepted by the prefix parser, token-length effects, one seed and five steps remain confounds.

`paper/tables/development_summary.json` contains metrics, source hashes, family scores and intervals; `family_predictions.csv` contains every raw completion/prediction by family/condition. Build script uses only stored records, stdlib and the existing parser. CE’s first five step times total 4.8694 seconds, maximum recorded allocated 2731.104 MiB and reserved 2830 MiB. The pilot’s corresponding update timing is 22.2652 seconds and 3033.665/3304 MiB. These are descriptive process measurements, not controlled runtime benchmarks; CE lacks equivalent periodic telemetry/audit overhead.

## Engineering changes and recommendation

Added a training-specific interpretation of existing numerical comparisons, required before checkpoint publication; added two CPU regressions for missing/finite LoRA changes and frozen/nonfinite violations; included the verification source in run identity; recorded periodic resources; excluded transient writer locks from Git. Scoring, loss constants, data, optimizer and sampling remain frozen. Manuscript source, registries, 13 primary-source literature entries, figure inventory and dated submission plan are present under `paper/`.

Fresh verification: ULVO doctor passed; full ULVO suite ran 282 tests, all passed; standard Python ran 282 tests with 92 optional-dependency skips, passed; portability and compileall passed. Paper files, local links, Python/JSON syntax, source hashes and deterministic table regeneration passed. Dependency manifests, frozen configs and datasets have no Git changes. The saved-state audit's initial repeat failed with FileExistsError on its immutable acceptance output; the helper now checks equality with an existing report, preserving it, and was reverified without a training retry.

**CONDITIONAL GO for proposing a separately authorized matched continuation/replication protocol. NO GO for an effectiveness or mechanism claim.** No remaining 45 steps were run or authorized. Next research task: independent contributor review of saved-state/parser evidence and preregistration of family-disjoint evaluation and matched replication; obtain explicit authorization before sustained training or final-holdout use.
