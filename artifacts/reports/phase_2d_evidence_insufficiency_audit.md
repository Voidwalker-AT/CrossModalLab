# Phase 2D Report — Cross-Model Evidence-Insufficiency Audit

**No causal patching. Holdout untouched. Historical 2C-C results preserved.**

---

## 1. Independent Phase 2C-C audit
Recomputed from raw JSONL:
- **36** lines → **36** unique eval IDs · 18/model  
- Revisions match pins · image/prompt/GT mismatches **0**  
- BA Smol **0.444** / Qwen **0.556** · fork abstention **0/6** both  
- Parsing of raw forks: definite Yes/No only (no “Undetermined” token)

## 2. Candidate-scoring validation
2C-C used **greedy generation only** (no candidate-LLM scoring) for paired evals — parsing is direct from `raw_completion`. No multi-token Undetermined scoring defect in that path. Candidate scoring remains validated separately in SmolVLM harnesses (not used as primary 2C-C metric).

## 3. Timing discrepancy
2C-C latency lacked `torch.cuda.synchronize()`. **2D measures synchronized timing.**

| Path | Mean latency |
|---|---:|
| SmolVLM2 visual formats | **17.9 s** |
| SmolVLM2 text-only | **1.3 s** |
| Qwen visual formats | **0.52 s** |
| Qwen text-only | **0.27 s** |

Gap persists under sync (eager vs SDPA, architecture). **Do not claim Qwen “faster generally”** from this; methods now comparable.

## 4–5. Format & evidence-sufficiency results (24 new evals/model)

### Text-only insufficient (no image, one textual premise)
| Model | Abstention |
|---|---:|
| SmolVLM2 | **6/6 (100%)** |
| Qwen3-VL | **6/6 (100%)** |

### Cross-modal fork format sensitivity (6 forks × 3 formats)

| Format | Smol abstain | Qwen abstain |
|---|---:|---:|
| Entailed/Contradicted/Insufficient Evidence | 0/6 | **3/6** |
| True/False/Cannot Be Determined | 0/6 | 0/6 |
| Neutral abstain-permitted (Yes/No/Undetermined) | **5/6** | 0/6 |

## 6. Cross-model comparison
- Both models **can** abstain on pure text insufficiency.  
- Both fail under default cross-modal Yes/No/Undetermined forks (2C-C: 0/6).  
- **Format wording changes abstention rates substantially** (Smol 0→0.83 on neutral; Qwen 0→0.5 on entailed vocabulary) — **response-format sensitivity**.  
- Qwen: strong **No-bias** on cross-modal forks (13/18 No on format set).  
- Smol: **Yes-bias** (12/18 Yes).

## 7. Alternative explanations
| Hypothesis | Support |
|---|---|
| Pure format/wording bias | **Partial** — large format effects; not complete (Qwen neutral still 0) |
| Cannot produce abstention token | **Rejected** — 100% text-only abstention both models |
| Cross-modal-specific insufficiency failure | **Supported** — fail with image+text, succeed text-only |
| Relational representation failure | **Open** — behavioral only |
| Candidate-scoring defect | **Not implicated** in 2C-C (generation-only) |

**Not a neural mechanism claim.**

## 8. Tests & resources
ULVO/system tests; GPU within 4 GB; 48 new evals total (24+24); no holdout.

## 9. Git
See SHA below.

## 10. Recommended next mechanistic experiment
**Smallest valid follow-up:**  
**Evidence-sufficiency representation probing on SmolVLM2** at `last` token on **text-only insufficient vs cross-modal fork** (same logical label, different modality) — tests whether sufficiency is decodable when the model *does* abstain vs when it does not.  

Alternative: one-pair activation patching **only after** correcting format sensitivity by using **neutral abstain-permitted prompt** as the frozen protocol (reduces pure wording confounds).  

**Not yet:** cross-model causal patching.

**Stopped after Phase 2D.**
