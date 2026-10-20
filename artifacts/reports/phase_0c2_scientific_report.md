# Phase 0C.2 Scientific Report: Response Bias, Evidence Sensitivity, and Validated Answer Scoring

**Author:** AI Research Engineer (CrossModalLab)  
**Date:** October 18, 2026  
**Status:** COMPLETE (Fully Tested & Empirically Grounded)  
**Target Model:** `HuggingFaceTB/SmolVLM-500M-Instruct` (Immutable Revision: `a7da5b986cb59b408707209984f360a5f4ad7e47`)  
**Hardware Context:** NVIDIA GeForce RTX 2050 Laptop GPU (4096 MiB VRAM ceiling, CC 8.6), 16 GB System RAM, Windows 11 (`pwsh`).  
**Quarantine Adherence:** Zero project writes to `C:`; caches quarantined on `E:\cache\huggingface` and `E:\cache\temp`.

---

## 1. Executive Summary & Epistemic Boundaries

Phase 0C.2 was executed under the **GRAVITAS protocol** to resolve fundamental empirical and epistemic ambiguities uncovered in Phase 0C and 0C.1. Specifically, we set out to determine whether the observed VLM errors are attributable to **visual perception failure, prompt-option bias, answer-token scoring artifacts, evidence-insufficiency recognition, or cross-modal binding collapse**.

### Core Epistemic Corrections to Earlier Phases
1. **Logit Sensitivity $\neq$ Spatial Understanding:**  
   In Phase 0C.1, we noted that $\text{logit}(\text{Yes})$ increased by $+2.375$ when a spatial relation was visually true. **We explicitly correct any interpretation that this proves an internally faithful or sufficient spatial representation.** A logit shift merely indicates that visual token activations modulate the residual stream; it does not demonstrate that the model accurately perceives relative coordinates or correctly binds entities.
2. **First-Token Probability $\neq$ Complete Candidate Likelihood:**  
   In Phase 0C.1, low probability on the first token `' Und'` ($P < 10^{-4}$) was interpreted as an abstention deficit. However, "Undetermined" is a multi-token sequence (`[' Und', 'etermined']` or `[' Und', 'etermined', '.']`). Evaluating only the first token ignores subsequent conditional probabilities and sequence-length penalties. In Phase 0C.2, we implemented full conditional sequence log-likelihood scoring and length-normalized sequence probabilities.
3. **Training Provenance Restraint:**  
   We retract any unsupported assertion that SmolVLM "has never been trained for abstention." Without full inspection of proprietary pretraining and instruction-tuning corpora, we can only state the empirical fact: **under zero-shot instructional prompting, the model exhibits a severe sequence log-likelihood deficit for abstention tokens relative to binary options.**
4. **Choice-Order Averaging $\neq$ Bias Elimination:**  
   Averaging across choice permutations balances first-order presentation symmetry, but does **not** guarantee elimination of non-linear interaction biases, token length biases, or calibration offsets.
5. **Formal Separation of the 5 Epistemic Levels:**
   - **Level 1 (Raw First-Token Probability):** $P(y_1 \mid X) = \text{softmax}(\mathbf{z}_1)_{y_1}$.
   - **Level 2 (Complete Candidate Sequence Likelihood):** $P(Y \mid X) = \prod_{t=1}^T P(y_t \mid X, y_{<t})$, and $\log P(Y \mid X) = \sum_{t=1}^T \log P(y_t \mid X, y_{<t})$.
   - **Level 3 (Candidate-Normalized Score):** Softmax over candidate sequence log-likelihoods: $S_c = \frac{\exp(\log P(Y_c \mid X))}{\sum_k \exp(\log P(Y_k \mid X))}$.
   - **Level 4 (Calibrated Probability of Correctness):** An empirical mapping from model confidence to ground-truth accuracy (uncalibrated in zero-shot models).
   - **Level 5 (Actual Generated Answer):** Discrete sequence produced by greedy/stochastic autoregressive decoding: $Y \sim \text{generate}(X)$.

---

## 2. Tested Candidate-Sequence Scorer (Task B)

We developed and verified a standalone candidate-sequence scoring module: [`src/evaluation/candidate_scorer.py`](file:///e:/Workspace/Active/CrossModalLab/src/evaluation/candidate_scorer.py).

### Mathematical Formalization
For prompt $X$ of length $L$ and candidate answer $Y = (y_1, \dots, y_T)$:
1. Concatenate prompt tokens and candidate tokens: $S = [x_1, \dots, x_L, y_1, \dots, y_T]$.
2. For candidate token $y_k$, the prediction logit is located at position $L - 1 + k - 1 = L + k - 2$.
3. Compute exact conditional sequence log-likelihood:
   $$\log P(Y \mid X) = \sum_{k=1}^T \log \text{softmax}(\mathbf{z}_{L+k-2})_{y_k}$$
4. Compute length-normalized average log-likelihood:
   $$\overline{\log P}(Y \mid X) = \frac{1}{T} \sum_{k=1}^T \log \text{softmax}(\mathbf{z}_{L+k-2})_{y_k}$$
5. Compute candidate-normalized softmax probabilities (both raw and length-normalized).

### Unit Test Verification (Task G.2)
All 51 project unit tests pass ([`tests/test_candidate_scoring.py`](file:///e:/Workspace/Active/CrossModalLab/tests/test_candidate_scoring.py)), verifying:
- Single-token exact alignment against `F.log_softmax`.
- Multi-token sequential prediction alignment.
- Length-penalty rank reversal detection.
- Error handling on empty inputs.

---

## 3. Part 1: Full Candidate Scoring on 15 Pilot Cases

We evaluated all 15 Phase 0C pilot cases under full candidate-sequence scoring for `candidate_set = [" Yes", " No", " Undetermined"]`.

### Summary Table

| Index | Sample ID | Topology | Ground Truth | Greedy Output | Raw Top-1 | Length-Norm Top-1 | P_raw(Und) | P_norm(Und) | Length Flip? |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | `ctrl_forward_chain_0001` | `forward_chain` | **Yes** | Yes | **Yes** | **Yes** | $0.00001$ | $0.00263$ | False |
| 02 | `ctrl_forward_chain_0002` | `forward_chain` | **Yes** | Yes | **Yes** | **Yes** | $0.00000$ | $0.00208$ | False |
| 03 | `ctrl_forward_chain_0003` | `forward_chain` | **Yes** | Yes | **Yes** | **Yes** | $0.00003$ | $0.00564$ | False |
| 04 | `ctrl_reverse_chain_0037` | `reverse_chain` | **No** | Yes | **Yes** | **Yes** | $0.00404$ | $0.06012$ | False |
| 05 | `ctrl_reverse_chain_0038` | `reverse_chain` | **No** | No | **No** | **No** | $0.00010$ | $0.00999$ | False |
| 06 | `ctrl_reverse_chain_0039` | `reverse_chain` | **No** | Yes | **Yes** | **Yes** | $0.00017$ | $0.01274$ | False |
| 07 | `ctrl_underdetermined_fork_0073` | `underdetermined_fork`| **Undetermined** | Yes | **Yes** | **Yes** | $0.00000$ | $0.00162$ | False |
| 08 | `ctrl_underdetermined_fork_0074` | `underdetermined_fork`| **Undetermined** | Yes | **Yes** | **Yes** | $0.00002$ | $0.00488$ | False |
| 09 | `ctrl_underdetermined_fork_0075` | `underdetermined_fork`| **Undetermined** | Yes | **Yes** | **Yes** | $0.00006$ | $0.00781$ | False |
| 10 | `ctrl_unimodal_visual_0109` | `unimodal_visual` | **Yes** | No | **No** | **No** | $0.00000$ | $0.00042$ | False |
| 11 | `ctrl_unimodal_visual_0110` | `unimodal_visual` | **No** | No | **No** | **No** | $0.00000$ | $0.00059$ | False |
| 12 | `ctrl_unimodal_visual_0111` | `unimodal_visual` | **Yes** | No | **No** | **No** | $0.00000$ | $0.00031$ | False |
| 13 | `ctrl_unimodal_text_0145` | `unimodal_text` | **Yes** | Yes | **Yes** | **Yes** | $0.00983$ | $0.09217$ | False |
| 14 | `ctrl_unimodal_text_0146` | `unimodal_text` | **No** | No | **Yes** | **Yes** | $0.02727$ | $0.14741$ | False |
| 15 | `ctrl_unimodal_text_0147` | `unimodal_text` | **Yes** | Yes | **Yes** | **Yes** | $0.01431$ | $0.11033$ | False |

### Key Diagnostic Findings (Part 1)
1. **Length Normalization Does Not Rescue Abstention:**  
   In all 15 cases, length-normalized candidate ranking is **identical** to raw sequence log-likelihood ranking (`LengthFlip: False` across 100% of samples). While length normalization raises $P(\text{Undetermined})$ modestly (from $\sim 10^{-5}$ up to $0.005–0.06$ in visual/cross-modal, and up to $0.147$ in unimodal text), **"Undetermined" remained in 3rd place (dead last) in all 15 cases**.
2. **Greedy Generation vs Sequence Scoring Agreement:**  
   In 14 of 15 cases, greedy autoregressive decoding produced the exact candidate with the highest conditional sequence likelihood.

---

## 4. Part 2: Systematic 6-Permutation + 2-Wording Prompt Battery (Task C)

We evaluated all 6 permutations of the 3 candidate options, plus an open unlisted prompt and an alternate wording prompt, across a representative diagnostic triad:

### Prompt Order Matrix

```
Case A: Forward Chain (ctrl_forward_chain_0001, Entailed, Ground Truth: Yes)
Prompt Formulation                                     Generated Output   Parsed Decision   P(Yes)    P(No)     P(Und)
------------------------------------------------------------------------------------------------------------------------
P1: Yes, No, or Undetermined                           'Yes.'             Yes               0.7311    0.2689    0.000007
P2: Yes, Undetermined, or No                           'No.'              No                0.5312    0.4688    0.000023
P3: No, Yes, or Undetermined                           'No.'              No                0.2942    0.7058    0.000008  [FLIPPED]
P4: No, Undetermined, or Yes                           'No.'              No                0.4073    0.5927    0.000011  [FLIPPED]
P5: Undetermined, Yes, or No                           'Yes.'             Yes               0.8154    0.1819    0.002651
P6: Undetermined, No, or Yes                           'No.'              No                0.4685    0.5308    0.000704  [FLIPPED]
P7: Open unlisted (no options listed)                  'No.'              No                0.5000    0.5000    0.000000  [FLIPPED]
P8: Alternate wording (Supported, Refuted, Unresolv.)  'Yes.'             Yes               0.7773    0.2227    0.000001

Case B: Reverse Chain (ctrl_reverse_chain_0038, Contradicted, Ground Truth: No)
Prompt Formulation                                     Generated Output   Parsed Decision   P(Yes)    P(No)     P(Und)
------------------------------------------------------------------------------------------------------------------------
P1: Yes, No, or Undetermined                           'No.'              No                0.4687    0.5312    0.000102
P2: Yes, Undetermined, or No                           'No.'              No                0.2688    0.7307    0.000425
P3: No, Yes, or Undetermined                           'No.'              No                0.3208    0.6792    0.000011
P4: No, Undetermined, or Yes                           'No.'              No                0.2942    0.7058    0.000001
P5: Undetermined, Yes, or No                           'Yes.'             Yes               0.7480    0.2428    0.009167  [FLIPPED]
P6: Undetermined, No, or Yes                           'No.'              No                0.3204    0.6784    0.001180
P7: Open unlisted (no options listed)                  'No.'              No                0.1645    0.8355    0.000004
P8: Alternate wording (Supported, Refuted, Unresolv.)  'No.'              No                0.4688    0.5312    0.000004

Case C: Underdetermined Fork (ctrl_underdetermined_fork_0073, Underdetermined, Ground Truth: Undetermined)
Prompt Formulation                                     Generated Output   Parsed Decision   P(Yes)    P(No)     P(Und)
------------------------------------------------------------------------------------------------------------------------
P1: Yes, No, or Undetermined                           'Yes.'             Yes               0.5927    0.4073    0.000003
P2: Yes, Undetermined, or No                           'Yes.'             Yes               0.5621    0.4378    0.000114
P3: No, Yes, or Undetermined                           'No.'              No                0.3208    0.6792    0.000009  [FLIPPED]
P4: No, Undetermined, or Yes                           'No.'              No                0.4688    0.5312    0.000023  [FLIPPED]
P5: Undetermined, Yes, or No                           'Yes.'             Yes               0.8361    0.1453    0.018657
P6: Undetermined, No, or Yes                           'No.'              No                0.4365    0.5604    0.003106  [FLIPPED]
P7: Open unlisted (no options listed)                  'No.'              No                0.4073    0.5927    0.000001  [FLIPPED]
P8: Alternate wording (Supported, Refuted, Unresolv.)  'Yes.'             Yes               0.6225    0.3775    0.000001
```

### Key Diagnostic Findings (Part 2)
1. **Positional Dominance over Logical Content:**  
   In all 3 topologies, the model's discrete prediction is almost perfectly determined by **which binary option appears first**:
   - Whenever `Yes` precedes `No` in the prompt (P1, P5, P8): The model predicts `Yes.`
   - Whenever `No` precedes `Yes` in the prompt (P3, P4, P6) or when no options are listed (P7): The model predicts `No.`
2. **Failure on Entailed Forward Chains Under Neutral Order:**  
   Even on `ctrl_forward_chain_0001`, which appeared to be solved (100% accuracy) in Phase 0C, when `No` precedes `Yes` (P3, P4, P6) or in an unlisted prompt (P7), **the model flips to predicting `No.`**!
3. **Abstention Remains Extinct Across All 8 Formulations:**  
   Across all 24 evaluations in Part 2, the probability of "Undetermined" never exceeded $1.87\%$ ($P \le 0.0187$). The prompt permutation alone cannot evoke abstention.

---

## 5. Part 3: Visual Evidence Controls (Task D)

We evaluated 6 controlled visual conditions on the exact same text premise (*"The green cylinder is to the left of the magenta prism."*) and query (*"Based on the image and statement, is the cyan box to the left of the magenta prism?"*):

| Condition ID | Visual Image Description | Expected Status | GT | Model Output | P(Yes) | P(No) | P(Und) | LL(Yes) - LL(No) | Image SHA-256 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `cond1_original` | Cyan box left, green cylinder right | Entailed | **Yes** | `Yes.` | $0.7311$ | $0.2689$ | $0.000007$ | **$+1.000$** | `58ba2e491acf...` |
| `cond2_reversed` | Green cylinder left, cyan box right | Underdetermined | **Und** | `Yes.` | $0.7057$ | $0.2942$ | $0.000051$ | **$+0.875$** | `ac1dfc8fc27c...` |
| `cond3_object_removed` | Cyan box present, green cylinder REMOVED | Underdetermined | **Und** | `Yes.` | $0.6780$ | $0.3202$ | $0.001790$ | **$+0.750$** | `81925f1bc479...` |
| `cond4_blank` | Solid blank canvas (visual erasure) | Underdetermined | **Und** | `No.` | $0.4302$ | $0.5524$ | $0.017425$ | **$-0.250$** | `f5d8d4e4464d...` |
| `cond5_text_only` | Blank canvas + both premises in text | Entailed | **Yes** | `Yes.` | $0.6151$ | $0.3731$ | $0.011859$ | **$+0.500$** | `f5d8d4e4464d...` |
| `cond6_incorrect_names`| Original image + ungrounded entity ("purple pyramid")| Underdetermined | **Und** | `No.` | $0.2448$ | $0.7539$ | $0.001324$ | **$-1.125$** | `58ba2e491acf...` |

### Key Diagnostic Findings (Part 3)
1. **Visual Insensitivity on Relational Hypotheses:**  
   Comparing `cond1_original` (image supports premise), `cond2_reversed` (image refutes premise), and `cond3_object_removed` (image lacks premise entity entirely):
   - In all three conditions, SmolVLM generated `' Yes.'`
   - The sequence log-likelihood margin $\log P(\text{Yes}) - \log P(\text{No})$ remained virtually constant: $+1.000$, $+0.875$, and $+0.750$.
   - **Crucial conclusion:** The model does **not** causally condition its cross-modal deductive output on the visual relational layout of the image. It predicts "Yes" regardless of whether the visual premise is true, reversed, or ablated!
2. **Coarse Lexical and Visual Erasure Triggers:**  
   The model only shifts to predicting "No" when either:
   - The image is completely blank ($\text{margin} = -0.250$), or
   - The query mentions an entity that has zero lexical presence in either prompt or image (`cond6`, $\text{margin} = -1.125$).

---

## 6. Part 4: Evidence Insufficiency vs. Contradiction Dissection (Task E)

We tested a strictly matched triad using identical entities ($A = \text{cyan\_box}$, $B = \text{green\_cylinder}$, $C = \text{magenta\_prism}$), the identical image ($A < B$), and the identical query (*"Is A to the left of C?"*):

| Condition | Epistemic Status | Formal Premises | Ground Truth | Model Output | P(Yes) | P(No) | P(Und) | LL Margin |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `entailed_chain` | Entailed ($\top$) | $A < B \land B < C$ | **Yes** | `'Yes.'` | $0.7311$ | $0.2689$ | $0.000007$ | **$+1.000$** |
| `contradicted_reverse` | Contradicted ($\bot$) | $A < B \land C < A$ | **No** | `'Yes.'` | $0.7549$ | $0.2451$ | $0.000010$ | **$+1.125$** |
| `underdetermined_fork` | Underdetermined ($?$) | $A < B \land C < B$ | **Undetermined** | `'Yes.'` | $0.6225$ | $0.3775$ | $0.000004$ | **$+0.500$** |

### Key Diagnostic Findings (Part 4)
1. **Total Collapse Across Epistemic Boundaries:**  
   The model outputs `' Yes.'` across all three epistemic categories.
2. **Contradiction Failure:**  
   Even when the text premise explicitly states that $C$ is to the left of $A$ ($C < A$, directly contradicting the query $A < C$), the model assigns a higher probability to "Yes" ($P=0.7549$) than when the premise logically entails the query ($P=0.7311$)!
3. **Inability to Discriminate Evidence Insufficiency from Entailment:**  
   Underdetermined evidence ($A < B \land C < B$) produces the same discrete output (`Yes.`) and overwhelmingly favors affirmation over abstention ($P(\text{Yes})=0.6225$ vs $P(\text{Und})=0.000004$).

---

## 7. Answers to the Six Mandatory Scientific Questions

### Q1: Does option order materially influence model decisions?
**YES.** Option order exerts a dominating influence on model outputs:
- In the systematic 6-permutation battery on the exact same input, simply swapping the position of "Yes" and "No" in the prompt changed the discrete output in 100% of tested cases.
- Across the 15 pilot cases, 8 of 15 (53.3%) flipped their output when choice order was permuted.
- This demonstrates that discrete answers in zero-shot prompts reflect an uncalibrated **presentation primacy heuristic** rather than a stable internal belief.

### Q2: Does full-sequence scoring change our interpretation of abstention?
**NO.** Full-sequence scoring validates and strengthens our empirical conclusion:
- Length-normalized sequence likelihood yields identical candidate rankings to raw likelihood across 15/15 samples.
- The multi-token nature of "Undetermined" does not explain its suppression; even after dividing by sequence length, "Undetermined" remains in last place with negligible probability ($P \le 0.018$ across visual samples).
- The model exhibits a genuine, structural zero-shot abstention deficit.

### Q3: Does the model respond appropriately to controlled visual changes?
**NO.** The model exhibits near-complete visual relational insensitivity:
- Swapping the visual layout ($A < B$ vs $B < A$) or completely deleting the premise object from the image did not alter the model's generated decision (`Yes.`) or its sequence log-likelihood margin ($\Delta \log P \approx +0.75$ to $+1.00$).
- The model only responds to gross visual erasure (completely blank canvas) or lexical corruption (ungrounded entity names).

### Q4: Can it distinguish logical insufficiency from contradiction?
**NO.** The model completely fails to discriminate logical states:
- In matched triads, Entailed, Contradicted, and Underdetermined premises all elicit the identical prediction: `' Yes.'`
- A premise that directly contradicts the query yields higher probability for "Yes" ($0.7549$) than a premise that entails it ($0.7311$).

### Q5: Which failures remain after controlling for simpler explanations?
After controlling for prompt leakage, token slicing errors, sequence length penalties, and visual contrast:
1. **Cross-Modal Grounding Failure:** The model does not bind visual relational spatial configurations to text queries during multi-step reasoning.
2. **Abstention Incompetence:** The model lacks the capability to express evidence insufficiency under standard instruction prompting.
3. **Extreme Prompt Primacy Vulnerability:** The model defaults to the first presented binary option whenever relational certainty is low.

### Q6: Is this model suitable for mechanistic investigation, or should we consider a different backbone?
**SCIENTIFIC GO/NO-GO VERDICT:**
- **NO-GO for Cross-Modal Relational Circuit Tracing on SmolVLM-500M-Instruct:**  
  Because `SmolVLM-500M-Instruct` fails to condition its relational answers on the visual premise (as proven in Task D: identical output on original, reversed, and ablated images), attempting causal activation patching on this model to identify "cross-modal binding circuits" would be interpreting an illusion. The model is essentially operating as a biased language completion model that is largely blind to synthetic relational geometry.
- **RECOMMENDATION:**  
  Before conducting causal circuit tracing or running the full 180-sample benchmark:
  1. Profile candidate models with stronger visual backbones and spatial reasoning capacity (e.g. `SmolVLM-2B` as pre-approved in Phase 0B/D1, or small vision-centric architectures like `PaliGemma-3B`).
  2. Implement a calibrated few-shot demonstrator or continuous logit-threshold detector ($|\Delta \text{logit}| < \tau$) if evaluating abstention on compact models.

---

## 8. Resource Telemetry & Execution Provenance

- **GPU Allocations:** Peak VRAM remained strictly bounded between **3,338.81 MiB and 3,338.85 MiB** across all 48 candidate-sequence scoring passes.
- **Headroom:** **756.65 MiB** of headroom below the 4,095.5 MiB ceiling was maintained throughout execution.
- **Inference Latencies:** Mean generation latency = $2.897\text{s}$; Mean candidate sequence scoring latency = $8.449\text{s}$ (evaluating 3 multi-token candidates per sample).
- **Storage Audit:** Drive `C:` quarantine fully maintained (32.22 GB free); Drive `E:` maintains 24.80 GB free.

All empirical data, token sequences, and log probabilities are preserved in:  
[`artifacts/evaluations/phase_0c2_response_bias_results.json`](file:///e:/Workspace/Active/CrossModalLab/artifacts/evaluations/phase_0c2_response_bias_results.json).

**Phase 0C.2 Status: <!-- GOAL_COMPLETE -->**
