# Dual-model evidence inventory

Audit base commit: `532e2276e585881a7d2a7f032bc32e09a31d0ec1`. All repository source links below refer to this fixed commit: https://github.com/Voidwalker-AT/CrossModalLab/tree/532e2276e585881a7d2a7f032bc32e09a31d0ec1 . `dual_model_audit.json` supplies hashes for 278 sources and a 108-file JSON/JSONL research inventory. It contains exact case/image/prompt identities, raw answers, candidate values, architecture configuration hashes, model revisions and phase-specific provenance. Binary activation vectors were read on CPU for integrity, not included in the ZIP. Checkpoint/model binaries are excluded.

| Model | Architecture from pinned local configuration | Implementation | Quantization |
|---|---|---|---|
| Qwen/Qwen3-VL-2B-Instruct @ 89644892e4d85e24eaac8bacfd4f463576704203 | Qwen3VLForConditionalGeneration; 28 text layers, hidden 2048, 16 attention / 8 KV heads; 24-block vision config, patch size 16 | src/models/qwen3vl2_loader.py; native multimodal M-RoPE; SDPA | NF4, double quantization, BF16 compute |
| HuggingFaceTB/SmolVLM2-2.2B-Instruct @ 482adb537c021c86670beed01cd58990d01e72e4 | SmolVLMForConditionalGeneration; 24 text layers, hidden 2048; 27 vision layers, patch size 14 | src/models/model_loader.py; canonical masked-scatter merger compatibility patch | NF4, double quantization, BF16 compute |

Architectural configuration is not an observed circuit. Full weight-file content was not rehashed in this mission. Historical raw metadata and loader identities agree with pinned revisions; no model was loaded or downloaded.

## Status taxonomy

**VERIFIED HISTORICAL EXECUTION** means surviving records establish a historical execution with checked identity/counts. **NEWLY VERIFIED EXECUTION** here refers only to the separately completed nine-forward Qwen run, whose saved artifacts are rechecked now; this mission has no new execution. **CPU-ONLY RECOMPUTATION** means fresh arithmetic/hash checks on stored records. **INCOMPLETE EXPERIMENT** and **UNVERIFIED CLAIM** remain explicit where logs, controls or design do not support an inference.

| Evidence | Model | Surviving record / scope | Classification and paper use |
|---|---|---|---|
| Phase 0D feasibility / extended validation | Smol | artifacts/evaluations/smolvlm2_feasibility_audit_results.json; smolvlm2_extended_validation_results.json; telemetry/smolvlm2_quantized_profile.json | VERIFIED HISTORICAL records; tiny grounding/triad/hook checks. Old narrative overclaims are not endorsed; omit primary performance claims. |
| Phase 0D.1 merger diagnostics | Smol | artifacts/evaluations/phase_0d1_matched_diagnostics_results.json; inputs_merger_audit_results.json | Historical compatibility/engineering evidence, not a new reasoning result. |
| Phase 0D2 smoke | Smol | artifacts/evaluations/phase_0d2_smoke/; three overlapping examples | Historical smoke, not additional independent cases. |
| Phase 0D2 main | Smol | artifacts/evaluations/phase_0d2/; 36 rows / 30 matched groups; three resumed IDs | VERIFIED HISTORICAL + CPU recomputation. Main descriptive paragraph/table. |
| Phase 1A passive observation | Smol | artifacts/instrumentation/phase_1a/activations_meta_v1.jsonl; 10 captures / 200 rows | Historical passive captures, CPU finite/norm/identity checks. Stored zero logit drift, not a fresh passivity experiment. |
| Phase 1B dependency pilot | Smol | artifacts/evaluations/phase_1b/; 36 stored predictions / 12 families | Historical outputs inspected; source split is test-labeled, closed dataset not reopened. Supplementary provenance only. |
| Phase 1D probes | Smol | artifacts/probes/phase_1d/; earlier 36 train / 9 validation pilot | Label/condition confounding; superseded exploratory pilot. Omit main result. |
| Phase 1E contrast probes | Smol | artifacts/probes/phase_1e/; earlier 45-record feature/probe pilot | Historical feature reuse, not independent replication of Phase 1F. |
| Phase 1F reliability | Smol | artifacts/probes/phase_1f/; 108 feature rows / 36 families; 81/27 evaluation rows | CPU-checked features, disjoint families, 13 stored probe specs. Correlational, development-selected; acquisition-role discrepancy disclosed. |
| Phase 2B interventions | Smol | artifacts/causal/phase_2b/; 16 surviving records / 10 interventions; layer 18 | INCOMPLETE. Deleted earlier records, ambiguous total attempts, controls larger than targets. No relation-specific causal mechanism. |
| Phase 2F QLoRA feasibility | Qwen | artifacts/training/phase_2f/qlora_feasibility_report_v1.json | Historical engineering feasibility. Synthetic text-only fixture must not enter empirical performance. No Smol fine-tuning attribution. |
| Phase 2G earlier pilot | Qwen | artifacts/training/phase_2g/phase_2g_report_v1.json and train_metrics_append.jsonl | Earlier reported 50-update run, not the selected Phase 2I-A CE control. Report has empty steps field despite separate journal; do not infer per-step execution from total_steps alone. Omit main results. |
| Phase 2I-A CE control | Qwen | artifacts/training/phase_2i_a/phase_2i_a_report_v1.json; append-only journal; checkpoints/ | Historical CE 5 and CE 50 generation endpoints on 18 development cases / six families. Selected control, no confusion with Phase 2G. |
| Phase 2I-B recovery and pilot | Qwen | artifacts/training/phase_2i_b/runs/56e277e421e442b08d33b107fc934494/ | Historical CE+ES paused at five updates. Saved states step_0000 to step_0005; original audit records optimizer/RNG/component validation. No new resume or optimizer action in this mission. |
| Earlier Phase 3 partial pilot | Both planned | artifacts/phase3/20261024_v2/pilot/; Smol scores survive, combined execution failed | INCOMPLETE cross-model execution. Never count it as a completed two-model study. |
| Successful isolated pilot | Both | artifacts/publication/20261025/pilot/; two matched cases / one family / six forwards per model | VERIFIED HISTORICAL execution, fresh CPU identity/ledger/parameter receipt checks. Optional matched preference supplement only. |
| 108-forward follow-up screen | Smol attempted; Qwen not started | artifacts/publication/20261025/screen108/execution.json | FAILED: one attempted forward, no complete scored case. Not a 108-case successful evaluation; final invariance not established. |
| Initial Overleaf replication admission | Qwen planned | paper/overleaf_evidence_20261025/audit/replication_admission.json | BLOCKED then: zero forwards, insufficient RAM. Immutable historical record. |
| Subsequent RAM-admitted scoring | Qwen | artifacts/publication/20261025/qwen_replication_20261025_464531de/ | NEWLY VERIFIED prior execution: nine forwards, three cases / one family; 625 unchanged parameter tensors. Current mission rechecks CPU only. |

No inspected evidence supports Qwen activation/probe plots or Smol training/checkpoint benefit. The selected Smol captures/probes and causal artifacts identify Smol explicitly. Existing generic instrumentation APIs do not establish measurements for Qwen. Earlier `initial_pilot_15_results.json`, `pilot_15_corrected_v2.json` and response-bias files must be attributed from their own recorded model metadata; they are not silently reassigned to SmolVLM2. The full source file inventory is provided to retain those historical distinctions.

## Checkpoint and failed-work provenance

Selected CE+ES states are six distinct saved checkpoints (steps 0-5), not six additional experiments. The original Overleaf audit's `recomputed.json#checkpoint_audit` records 224 trainable tensors, component hashes, CPU-reloaded optimizer moments and valid RNG states for each. That receipt is inherited evidence, not a new checkpoint reload here. Current source hashes protect JSON state/configuration records; this audit does not deserialize optimizer binaries or execute a training resume. CE control, earlier Phase 2G and CE+ES checkpoints remain separate directories and protocols.

Raw image identity is a SHA256 of actual pixels/file bytes, not merely matching object names. Prompt identity is a hash of full task text; native chat templates and image tokenization remain model-specific. Source receipts preserve the old provenance rather than rewriting run IDs, split labels, failed-screen status or recovery history.
