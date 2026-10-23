# Phase 2I-B runner audit and repair — 2026-10-23

**CONDITIONAL GO for a separately authorized five-step CE+ES pilot.** No GPU
training, VLM loading, optimizer step, final-holdout evaluation, installation,
model download, causal patching, or dashboard work was performed in this task.
This report certifies CPU-tested runner engineering, not optimizer-step hardware
feasibility or effectiveness of evidence-sufficiency training.

## Findings, reproduction, and fixes

At task entry the recovery branch was `codex/repair-qwen-candidate-scoring`,
HEAD `537c6d6bcc52af11017e4bb7a884c36e13858129`. The runner was untracked.
Its original bytes are preserved in
`artifacts/training/phase_2i_b_runner_recovery/9833770dd2f04040a84b54a3133206d5/original_runner.py.txt`,
SHA-256 `ddec99ebc271c335ddd02dd419a04b3d9603db5c6942e2bb7d2de95b978a8e4b`.
Existing source and artifact schemas, pinned protocols, loss implementations,
and the CE control runner were inspected before changes.

Confirmed defects:

- `completed_steps` accepted duplicate rows and used only their maximum step.
- The missing-checkpoint branch erased metrics and silently restarted at zero.
- A fresh model/optimizer was created on every invocation, with no adapter,
  optimizer, or RNG restoration despite skipping recorded steps.
- Checkpoint metadata was published before optimizer serialization finished.
  Fixed checkpoint/output names could overwrite earlier artifacts.
- No run identity, dataset/schedule hash, parameter binding, RNG state, atomic
  multi-file publication, or meaningful resume validation existed.
- The RAM gate failed open on exceptions. There was no reliable exception or
  cancellation cleanup path and no finite-gradient guard before stepping.
- The abstention-prompt baseline ran outside the disabled-adapter context, and
  evaluation did not explicitly select eval mode or preserve training RNG.

The original duplicate acceptance and metric erasure were reproduced by executing
the actual original function/branch on quarantined synthetic files. The failing
command observed accepted duplicates and a zero-byte journal. Regression tests
now reject duplicates and preserve a journal when checkpoints are missing.

The existing model adapter, prompt construction, gold-span masking, candidate
scorer, and loss definitions were retained. Changes concern run lifecycle,
provenance, validation, and safeguards. The frozen configurations and historical
CE implementation/results were not edited.

## Checkpoint/resume protocol

`src/training/run_state.py` supplies a single-writer store with an operating-system
file lock. Locks release on process exit; stale lock files need no deletion.
Each intentional fresh experiment gets a new UUID directory and immutable
`run.json`, with initial PREPARED status. There is no implicit fresh/resume mode.
Status/attempt events are separately published immutable JSON files. Nonterminal
attempts without completion remain evidence of interrupted work.

Each step, including initial step zero, is committed to a new checkpoint directory
containing:

- PEFT `adapter_config.json` and `adapter_model.safetensors`.
- AdamW state with named trainable-parameter layout and all moment tensors.
- Python, NumPy (when present), CPU PyTorch, and initialized CUDA RNG states.
- Metadata binding the step, run and experiment hashes, component SHA-256 values,
  previous metadata hash, complete metric row, and metric-prefix hash.
- Explicit scheduler=None; this frozen protocol has no scheduler to restore.

Files are saved and fsynced in an exclusively created `.incomplete-*` directory.
Metadata is written after required components. A same-filesystem directory rename
publishes the checkpoint, then the corresponding row is fsynced to the append-only
metric journal. Existing committed directories are never reused or replaced.

Resume validates the contiguous checkpoint chain, every required component hash,
the run identity, metric rows, optimizer settings, parameter order, moments,
state steps, adapter configuration/keys/shapes/dtypes, and RNG structure/topology.
It restores weights into the existing adapter, preserving optimizer parameter
objects, loads optimizer state, then restores RNG last. A step count is never
used as proof that these states were restored.

After an interrupted append, `--recover-metrics` is required. Recovery uses rows
already stored in committed checkpoints, completing only a byte-prefix journal
or appending missing rows. It never truncates or replaces existing bytes. A
duplicate, contradictory, mixed-identity, or non-prefix journal is rejected.
Incomplete checkpoint directories are retained and reported; only fully published
checkpoints can be restored. A missing initial checkpoint requires a new run ID.
Legacy CE/unfinished-run artifacts are not silently migrated or resumed.

These are process-interruption guarantees exercised with fault injection. Full
power-loss/filesystem durability and GPU continuation equivalence remain UNVERIFIED.

## Experiment identity and history

`src/training/phase_2i_b_protocol.py` binds the pinned model/revision and NF4 settings,
full frozen protocols and hashes, LoRA configuration, optimizer/scheduler settings,
complete ordered 50-step schedule, dataset manifest hash, selected prompt/image
hashes, validation IDs, parser/prompt configuration, source hashes, and runtime
versions. Changes to these fields reject resume instead of silently continuing.
The manifest is hashed in full; only train/validation records are selected and
their image files opened. No held-out images were opened or held-out prompts
sent to a model, and no holdout metrics were computed.

The five-step budget is execution control, not a different schedule. A resumed
run can explicitly extend its budget within the same frozen 50-step horizon.
The validated development plan contains 50 unique training families and 18 cases
from six disjoint validation families. Source/environment changes require a
different identity; exact cross-machine/GPU replay is not asserted.

Both baseline prompt conditions disable the adapter and use eval mode. Development
evaluation saves/restores RNG and model mode, so it cannot consume the continuation
RNG stream. Evaluation files and execution reports use immutable paths; step
evaluation is named with its actual step, not always mislabeled as step five.

Historical verification: 133 pre-existing artifact/config/source files, including
saved checkpoint binaries, all three requirements files, both frozen protocols,
the scorer, and the CE runner matched baseline hashes. The hashed set totals
52,669,774 bytes; the plan evidence includes its complete file/hash mapping.
Unrelated untracked failure evidence was left in place. No caches, model weights,
adapter weights, or optimizer binaries are included in the commit.

## Mathematical audit

For complete, summed candidate sequence log-likelihoods `s_Y`, `s_N`, `s_U`,
the unchanged objective is:

```text
insufficient evidence: ES = max(0, 1 + max(s_Y, s_N) - s_U)
sufficient evidence:   ES = max(0, 1 + s_U - s_gold)
L_total = L_CE + 0.25 * L_ES
```

Candidate order is Yes, No, Undetermined; sufficiency is zero exactly for the
Undetermined target. CE supervises only the gold answer-token span. Although the
legacy frozen metadata calls masking `gold_answer_span_suffix`, the actual exact
span implementation is recorded explicitly in the new identity.

All candidates are scored uniformly from the user prompt/image, with no gold
label, symbolic graph edge, or oracle annotation supplied to that scoring forward.
Labels/sufficiency are used only as supervised loss targets. The code neither
detaches candidate scores nor normalizes by answer length nor replaces ES with CE.
One total loss is backpropagated once; optimizer selection is all and only named
trainable LoRA parameters. Non-finite or absent/nonzero-free gradients and any
frozen gradient fail before a future optimizer step.

CPU gradient tests verify exact loss composition and coefficient application for
all three targets. A counterexample holds gold abstain probability (and thus CE)
fixed while changing the best definite competitor, producing different ES values.
The hinge is therefore not mathematically identical to CE. This does not establish
independent information, novelty, or empirical benefit. The pre-registered summed
likelihood gives multi-token Undetermined a token-length penalty; that design and
its calibration consequences are preserved and must be examined empirically.

## Execution safety and limitations

Normal invocation cannot accidentally train: `--execute-training` and an explicit
`--new-run` or `--resume` are required. Default step budget is five, with a frozen
maximum of fifty. Read-only `--plan` works without torch/PEFT and creates no run.
Future GPU execution requires CUDA and at least 4 GiB available physical host RAM;
failure to measure RAM fails closed. Model access is local/offline and NF4 is
checked against the experiment identity. Temporary files must remain in the
workspace; PyTorch allocation is capped below dedicated VRAM. Resource gates run
between stages and before/after an optimizer operation. Step telemetry includes
GPU allocator peaks and available host RAM; failure paths record memory, exception
type/traceback, incomplete status, and cleanup results. Step peak counters reset
after input packing and before the CE forward.

UNVERIFIED: CE plus three candidate graphs and AdamW first-step allocation have
not been executed together. The preceding scorer-only GPU measurement does not
prove optimizer-step feasibility. Sudden termination may prevent a final report;
durable step/attempt records and validated checkpoints remain authoritative.
GPU kernels can also prevent bitwise replay despite restored RNG. These limitations
are reasons for a bounded, observed pilot rather than automatic 50-step execution.

## Fresh verification

| Check | Observed result | Exit |
|---|---|---:|
| Focused runner regressions | 25 passed | 0 |
| Full ULVO unittest discovery | 280 tests, OK, no skips; 8.985 seconds | 0 |
| Standard Python discovery | 280 tests, OK, 90 optional-dependency skips | 0 |
| ULVO portability script | PASSED | 0 |
| Compileall: src/scripts/tests | PASSED | 0 |
| ULVO project doctor | PASSED; final READY, zero active leases | 0 |
| CPU development plan | 50 scheduled cases/families, 18 validation cases; no model | 0 |
| Historical/hash verification | 133 files unchanged; original runner archive exact | 0 |

The 25 new tests cover fresh initialization, valid restore, missing/corrupt
checkpoints, wrong revision/dataset/coefficient, missing/incorrect optimizer and
RNG state, duplicate/mixed metrics, duplicate commits, interrupted saves and torn
journal recovery, single-writer exclusion, historical immutability, exact
differentiable objective gradients, no CE substitution, holdout guards, and CPU
planning. A tiny synthetic PEFT model additionally exercises actual safetensors
adapter serialization/restoration without replacing optimizer-bound parameters.
Fixtures are quarantined as SYNTHETIC DEMO. Optimizer steps are guarded against
in the fixtures; AdamW moments are hand-constructed, not obtained by optimization.

Machine-readable plan and preservation evidence:
`artifacts/training/phase_2i_b_runner_recovery/9833770dd2f04040a84b54a3133206d5/validated_development_plan.json`.
Its final CPU identity hash is
`c1e6bfb222e0cf2c10dad9ce0144434d3ccf460c7c70b34bb1ca93d41311a797`.
Exact test outputs are retained locally in the same directory's log files.

## Next decision

Only after separate user authorization, recheck RAM, CUDA, competing GPU workers,
and the CPU plan, then consider a new five-step run. Inspect each step's gradients,
losses, memory, checkpoint publication, and journal consistency; stop on errors.
This task did not authorize or execute that pilot. Development-set reuse, the
existing prefix-based parser's permissive trailing-output handling, limited sample
size, length effects, and lack of an independent test or causal mechanism remain
scientific limitations. No main merge or further phase is implied by this repair.
