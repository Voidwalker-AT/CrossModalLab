# Qwen RAM recovery and bounded development verification

Status: **VERIFIED bounded candidate-scoring execution**. Historical generated-answer reproduction: **NOT ESTABLISHED**.

## Admission and process safety

Initial read-only measurement: approximately 6.06 GiB available physical RAM of 15.649 GiB. The saved process inventory measured 5.956 GiB. Initial NVIDIA telemetry: 4096 MiB total, 182 MiB used, 3781 MiB free. No cleanup was necessary. No process was stopped; memory recovered by agent cleanup: **0 MiB**. Later changes in available RAM cannot be attributed to cleanup.

Inspected 50 matching application/tool processes, with executable, command line, creation time, parent/children, session, memory and TCP ownership recorded locally. Working directories were unavailable through CIM. REA leaf PIDs 37688 and 17312 each used approximately 241 MiB and traced through launchers to active Codex PID 11280. Retained the REA processes and all Codex/browser/helper processes: idle/client-free ownership was not established and cleanup was unnecessary. No Python model worker, Ghidra/Java analysis worker or x64dbg debugger was observed. The GPU entry with unavailable NVIDIA process metadata, PID 1920, was identified as Desktop Window Manager and retained. ULVO agent: offline, zero active leases; project environment: ready, zero active processes before execution.

Pinned local Qwen3-VL-2B-Instruct revision `89644892e4d85e24eaac8bacfd4f463576704203`, NF4 double quantization, BF16 compute, SDPA, frozen base without adapter. CUDA execution succeeded. Before worker launch: 6.002 GiB available host RAM. Gates: at least 4096 MiB available RAM, initial free dedicated VRAM at least 3500 MiB, physical GPU use at most 3900 MiB, 4 GiB disk reserve. Existing parent admission computes available VRAM from total minus used; the independent initial NVIDIA free reading also passed. Shared GPU memory was not used as an admission reserve.

## Frozen inputs and results

Protocol: `paper/overleaf_evidence_20261025/audit/replication_protocol.json`, SHA256 `247c6aef4b06d6d2282d87e983fcefd6662e00a8a8fb21a2ff15f5dc50fa3ff3`. The original protocol and Overleaf package remain unchanged. Cases are allowlisted validation examples, despite `ho` in their family ID; no frozen final-holdout records were opened by this worker. Image and prompt hashes were checked against the frozen protocol. Effective chat templates, hashes, input token IDs and shapes are preserved in `candidate_scores.jsonl`.

Candidates are leading-space ` Yes`, ` No`, ` Undetermined`, token IDs `[7414]`, `[2308]`, `[16887,46637]`. Scores are full summed teacher-forced log-likelihoods, including both Undetermined tokens. Three forwards per case, nine actual attempted forwards, nine finite candidate scores. The persisted ledger records every entered top-level call. No retry. The unused three forwards are not permission for another run.

| Case ID | Gold | Yes LL | No LL | Undetermined LL | Candidate argmax |
|---|---|---:|---:|---:|---|
| xc_ho_0011::entailed | Yes | -15.542109489 | -16.292110443 | -15.542119026 | Yes |
| xc_ho_0011::contradicted | No | -22.698657990 | -17.323657990 | -17.448665619 | No |
| xc_ho_0011::underdetermined | Undetermined | -20.762607574 | -16.450107574 | -16.075117111 | Undetermined |

Candidate argmax agrees with all three gold labels, but these are **three dependent cases from one family**, not evidence of general accuracy. The entailed case has a Yes-versus-Undetermined gap of only 0.000009536743. Its stability under numerical or prompt perturbations is **UNVERIFIED**. No such perturbation experiments were performed.

Generated text, parsed generated answer and exact-format compliance are **NOT APPLICABLE**. Historical 18-case frozen-Qwen observations used greedy generation; the earlier candidate pilot used different samples. Neither is an exact matched comparison to these nine forwards. No historical metric is replaced, merged or reclassified. The earlier RAM-blocked admission and failed 108-forward screen remain separate preserved evidence. No causal or corrective-method conclusion follows.

## Resource and parameter verification

Worker elapsed: 70.524 seconds; model load: 7.85 seconds. Minimum sampled available host RAM across parent and worker: 4213.254 MiB (4.115 GiB). Maximum sampled physical GPU use: 1958 MiB. PyTorch allocator peaks: 1584.735 MiB allocated, 1678.000 MiB reserved. Sampling interval approximately 0.5 seconds plus sampling overhead; physical figures are not continuous peaks. No sampled gate violation or sampler error.

All **625 parameter tensors** were exactly numerically unchanged, with identical before/after byte hashes and no gradients. All parameters remained frozen. This protocol used no optimizer, backward pass, training, LoRA adapter, causal patching, package change or model download. It does not test LoRA gradient propagation. Worker launcher PID 9428 exited with code zero; actual Python worker PID 18384 exited. GPU physical use returned to 165 MiB; immediately post-worker available RAM was 6.133 GiB. Normal shutdown was verified; emergency cancellation through a Windows launcher chain was not exercised.

Final post-test shutdown check: 6.326 GiB available host RAM; GPU 180 MiB used / 3783 MiB free; zero remaining replication workers; active Codex preserved. ULVO was quiescent with zero active processes. See `shutdown_check.json`.

## Fresh verification and preservation

See `verification.json` for exact commands and exit codes. All ten checks passed: ULVO doctor; six recovery tests; 17 scorer tests; budget/research audit tests; guarded full integrity suite; portability; compilation; ordinary-Python recovery tests; read-only evidence validation; Git whitespace check. Full suite: 353 tests, 3 explicitly skipped, zero failures/errors and zero restricted open attempts. CPU fixtures are tests, not new VLM experiments.

All **1039 protected files** matched their pre-run byte hashes, including the original 12 untracked artifacts, historical evidence, the complete Overleaf directory and three protected submission archives. Overleaf ZIP SHA256 remains `dd83fcc4b48d0d532d088ea92999a39ed7fe1c62a4ae07c0a65aa09e54141459`.

The initial staged whitespace check flagged preserved CRLF bytes in the original loader snapshot and raw shutdown receipt. Those evidence bytes were retained. The focused check `git -c core.whitespace=cr-at-eol diff --cached --check` passed, treating CRLF as line endings while retaining other whitespace checks. This is a per-command option; no Git configuration was changed.

`executed_sources/` preserves six executed source files, verified against the hashes recorded before admission. `plan.json`, `execution.json`, `worker_execution.json`, `candidate_scores.jsonl`, `forward_ledger/`, `summary.json` and `verification.json` constitute the compact scientific record. Full command lines and raw terminal/check logs stay local under `.cache/qwen_replication_20261025_464531de/` and are excluded from Git. No weights, optimizer binaries or caches are included.

## Reproducibility and remaining limits

Read-only validation (no model loading):

```text
python scripts/verify_qwen_ram_recovery.py artifacts/publication/20261025/qwen_replication_20261025_464531de
```

Do not rerun the inference command: the requested bounded run is complete, and existing output cannot be resumed or reset. Additional GPU work requires a separate authorized protocol. Full weight-file hashing, numerical robustness, wider family coverage, generated-answer reproduction, and emergency-cancellation testing remain unverified. No new research claim or manuscript change is made. This report is distinct from the immutable October/Overleaf releases.
