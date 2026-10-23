# Reproducibility and known limitations

The presentation uses stored development evidence. Runtime requests perform no generation, optimizer calls, checkpoint reloads or causal interventions. The packaged server uses Python's standard library and project symbolic/parser modules; it imports neither ULVO nor Torch. Model caches and GPU environments are unnecessary.

The package contains the exact 18 allowlisted validation records and their original PNG bytes. It omits the complete dataset manifest and final holdout. All six conditions share sample IDs, prompt/image hashes and logical labels. Source-prompt hashes for abstention-prompt conditions identify the original record; the recorded augmentation rule is separate.

The original bootstrap-table JSON is byte-reproducible with recorded Python 3.11.16. A Python 3.13 rebuild changed three values at the last floating-point digit because summation behavior differs; counts, displayed results and scientific interpretation were unchanged. The original table bytes were restored by rebuilding with Python 3.11 and verified against Git. Use that recorded runtime when checking exact table hashes; dashboard startup works on ordinary Python 3.13 as tested locally.

Pinned Qwen identity: `Qwen/Qwen3-VL-2B-Instruct@89644892e4d85e24eaac8bacfd4f463576704203`. Historical Smol identity: `HuggingFaceTB/SmolVLM2-2.2B-Instruct@482adb537c021c86670beed01cd58990d01e72e4`. NF4, double quantization, BF16 compute. These identify past experiments; the presentation does not load either model. The pilot run is `56e277e421e442b08d33b107fc934494`, paused at step five.

`strict_answer_v1` extracts historical label prefixes and rejects conflicting labels. Its name does not imply exact formatting. The dashboard reports semantic correctness and exact raw-label compliance separately. Raw suffixes and decoding replacement characters remain preserved. Balanced accuracy averages three class recalls; invalid outputs remain errors. Descriptive intervals in the paper resample whole families 2,000 times with seed 20261023 and do not correct for development reuse or selection.

## Scope of reproducibility

The delivered archive has a file manifest. Verify it with:

```sh
python submission/verify_package.py .
python submission/smoke_package.py .
```

This checks the exact payload inventory, byte sizes and SHA-256 values. It does not reconstruct training or assert model-binary availability. The package omits adapters/optimizer/RNG binaries deliberately. Real-model restoration was previously audited read-only; identical resumed optimization trajectories remain untested because no new step was authorized.

For source contributors, locked Node dependencies are needed only to rebuild the frontend. A fresh `npm ci --ignore-scripts --no-audit --no-fund` can require network access; it is a deliberate contributor setup operation, not a dashboard runtime action. No fresh dependency installation was performed for this package. The Python-only extracted archive is the offline demonstration path.

## Scientific limits

- Eighteen repeatedly inspected cases / six validation families, one training seed, no independent generalization estimate.
- CE and CE+ES have a five-step comparison; CE-50 has a different training budget. No beneficial ES effect has been established.
- Historical parser and malformed CE completions confound interpretation. Correct abstention, inappropriate abstention and unsupported definite answers must be considered together.
- Candidate answer lengths differ. The fixed ES score sums token log likelihoods without normalization.
- Probe accessibility is correlational. Surviving intervention controls do not identify a relation-specific causal circuit; historical lost records are not recoverable from summaries.
- Architecture schematics and logical countermodels are explanations, not measured activations, observed scenes or neuron-level mechanisms.
- Primary-machine extracted-package startup does not establish partner hardware/browser compatibility. Physical partner reproduction remains pending.
- Reference inspection depth is recorded as abstract/metadata unless otherwise stated. Exhaustive novelty clearance, full-text review, author agreement, asset/license review and the university's final formatting requirements remain human/research follow-up.

No external submission, deployment-readiness certification or peer-review acceptance is claimed.
