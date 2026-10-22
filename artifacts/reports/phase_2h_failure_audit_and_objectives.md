# Phase 2H Report — QLoRA Failure Diagnosis & Evidence-Aware Objectives

**No GPU training. No holdout. No custom-loss execution on the VLM.**

---

## A. Independent Phase 2G metric audit (raw artifacts)

| Check | Result |
|---|---|
| Optimizer steps | **50 unique** (1–50), **0 duplicates** |
| Train labels in schedule | Yes 17 / No 17 / Undetermined 16 |
| Families exposed | **17** train families (max 3 exposures each) |
| Holdout | `holdout_used: false` |
| Loss trajectory | 7.03 → 4.33 @5 → 1.27e-4 @20 → **2.75e-5 @50** |
| Baseline BA | **0.389** (recomputed from `baseline_eval_v1.json`) |
| Final val BA | **0.222** (report; support 6/6/6) |
| Final confusion | Heavy **Invalid** predictions (Yes:3, No:4, Undet:3 invalid) — parse failures after training |
| Checkpoints | step_005, step_050 with matching `config_hash` / dataset sha / 802816 params |
| Limitation | **No per-sample baseline/final raw completions saved** — only aggregate + sample_ids; cannot reparse baseline texts |

## B. Training exposure & diversity (contrast-v2 train = 504 records / 168 families)

| Metric | Value |
|---|---:|
| Unique image SHA | **504** (1:1 with records) |
| Unique prompts | **504** |
| Unique A,C pairs | 83 |
| Labels | 168/168/168 |
| Single condition | `CROSS_MODAL_TWO_PREMISE` |
| Images with multiple labels | **0** (each image only one label — paired by design across different images) |
| 50-step exposure | Only **17 families** (curriculum head), not full 168 |

## C. Dataset leakage / shortcuts
- Same image SHA never carries two labels → image-only lookup **cannot** solve multi-label without memorizing **different** images per label (504 unique images).  
- Train images vs val: family-disjoint by construction.  
- Layout/gap diversity present (74 unique gap values).  
- **Valid relational signal exists** (both premises required flag on contrast set).  
- Residual risk: finite templates → prompt-structure shortcuts remain possible but less than unique-image label mapping.

## D. Supervision & optimization findings
| Hypothesis | Evidence | Against | Testable w/o GPU? | Future falsification |
|---|---|---|---|---|
| **H1 Repetition/memorization** | Loss→3e-5; only 17 families×3 in 50 steps; val **dropped** 0.389→0.222 | — | Partial (curriculum math) | Longer training on more families; early stopping on val |
| **H2 Shortcut learning** | Single template condition; Invalid outputs after train | Images unique per record | Partial | Multi-template eval; lexical probes |
| **H3 Excessive optimization / LR** | Loss collapse by step ~15 | Gradients finite; no NaN | Partial | LR sweep, grad clip experiment |
| **H4 Masking defects** | Last-4 span includes chat tail; may supervise non-answer tokens | Unit tests show prompt masked | **CPU fixture** only | Tokenize gold answers and supervise exact span |
| **H5 Eval/parse defects** | Final confusion shows many **Invalid** → parser failures dominate apparent drop | Baseline used same parser | **Yes from metrics** | Preserve raw completions; improve parser or constrained decode |
| **H6 Cross-modal integration failure** | Underdetermined recall stayed **0** | Trained on Undetermined class | Needs features | Probe sufficiency direction |
| **H7 Capacity** | r=2 only | 802k params may be enough for memorization | No | r=4/8 ablation |

**Strongest joint story:** H1 (memorization of short answer spans) **+ H5 (Invalid generations collapsing parse accuracy)**; H3 contributing. **Not proven unique.**

## E. Failed-run provenance
1. First run: 5 steps then `hashlib_sha(dict)` crash → metrics for steps 1–5 written, adapter **not** saved.  
2. Recovery cleared `train_metrics_append.jsonl` and reran from 0 (same seed/losses 7.03→4.33 reproduced).  
3. Original 5-step partial file **not retained** (deleted for clean restart) — **irrecoverable limitation**.  
4. Code change (`hashlib_sha` dict fix, resume logic) occurred **before** the successful 50-step run; not tuned on outcomes beyond crash fix.

## F. Literature collision matrix (verified where marked)

| Work | Cite | Overlap with our objectives | Status |
|---|---|---|---|
| **SelectiveNet** | Geifman & El-Yaniv, ICML 2019, arXiv:**1901.09192** | Selective prediction / reject option; joint CE+rejection | **[Card/Abstract]** |
| **MoHoBench** | Zhu et al., AAAI 2026 | Unanswerable VQA honesty / refusal | **[Full-Text/Repo]** (prior matrix) |
| **HallusionBench** | Guan et al., CVPR 2024 | Paired hallucination controls | **[Full-Text Inspected]** |
| **CF-VLM** | Zhang et al. 2025 | Counterfactual VLM fine-tuning | **[Prior matrix]** |
| **Winoground / ARO** | Thrush 2022; Yuksekgonul ICLR’23 arXiv:**2210.01936** | Compositionality eval (not our loss) | **[Full-Text Inspected]** |
| Project novelty_matrix | `docs/novelty_matrix.md` | Broad collision audit | **[Full-Text Inspected]** |

**Gap after review:** Combining **solver-grounded sufficiency hinges** + **exact-image/text CF pairs** + **VLM LoRA** is a plausible *combination* gap — **not claimed novel** without a broader post-design search.

## G. Mathematical objectives (implemented)

Common: \(x\) multimodal input, \(y \in \{\text{Yes,No,Und}\}\), \(p_\theta\) = softmax of **class logits** from sequence scoring (not calibrated confidence).

**A (CE):** \(\mathcal{L}_{CE} = -\log \mathrm{softmax}(z)_{y}\)

**B (CF):** \(\mathcal{L}_{CF} = \frac12[\mathrm{CE}(z(x),y) + \mathrm{CE}(z(x_{cf}), y_{cf})]\)  
(same-label pairs keep shared \(y\); no forced flip)

**C (ES):** sufficiency \(s\in\{0,1\}\) from solver;  
margin mode: \(s=0 \Rightarrow \mathrm{ReLU}(m+\max_{y\neq u} z_y - z_u)\);  
\(s=1 \Rightarrow \mathrm{ReLU}(m+z_u-\max_{y\neq u} z_y)\) (anti-gratuitous-abstention)

**D:** \(\mathcal{L} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{CF} + \mu \mathcal{L}_{ES}\), \(\lambda,\mu\ge 0\)

Failures: non-finite logits; negative λ/μ; missing CF/ES tensors; empty batch — all raise.

## H. Gradient-test evidence
**19/19 unit tests PASS** including finite grads, direction (CE ∂L/∂z_correct < 0), extreme-logit stability, insufficient-vs-sufficient margins, zero-λ equals CE, determinism.

## I. Proposed ablation (Phase 2I — not executed)

| Arm | Objective |
|---|---|
| A | Frozen base |
| B | Strong prompt-only |
| C | CE-QLoRA (Phase 2G) |
| D | CE+CF |
| E | CE+ES |
| F | Combined |

**Controls:** same 50 steps, r=2, targets, optimizer, seed, val families, prompts, model revision.  
**Metrics:** BA, macro-F1, per-class recall, abstention P/R, unsupported definite, over-abstention, CF consistency, train stability, wall-clock.  
**Pilot first:** ≤50 steps on development train only; **no holdout**.

## J. Remaining risks
- Baseline/final **raw text not stored** (limitation)  
- Invalid-parse confound must be fixed or measured separately  
- Tiny val (18 cases)  
- CF/sufficiency benefits **unproven**  

## K. Tests
| | Exit |
|---|---|
| `tests.test_phase_2h_losses` | **19 OK** |
| Full ULVO suite (pre) | **196 OK** |
| `py_compile` losses | **0** |

## L. Git
See push SHA.

## M. Recommendation for Phase 2I
**CONDITIONAL GO** to implement a **development-only** pilot of Arm E/F (ES or combined) **only if**:
1. Raw completions are logged (fix H5 measurement),  
2. Gold-answer span masking is verified on tokenized labels,  
3. Pre-registered λ/μ and step budget frozen.  

**NO-GO** for claiming objective novelty or performance gains yet.

**STOP after Phase 2H.**
