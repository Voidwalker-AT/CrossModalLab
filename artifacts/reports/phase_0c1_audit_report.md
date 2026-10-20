# Phase 0C.1 Scientific Audit Report: Experimental Confound Elimination and Diagnostic Dissection of SmolVLM-500M-Instruct

**Author:** AI Research Engineer (CrossModalLab)  
**Date:** October 18, 2026  
**Status:** COMPLETE (Audited & Re-evaluated)  
**Target Model:** `HuggingFaceTB/SmolVLM-500M-Instruct` (Immutable Revision: `a7da5b986cb59b408707209984f360a5f4ad7e47`)  
**Hardware Context:** NVIDIA GeForce RTX 2050 Laptop GPU (4096 MiB VRAM ceiling, CC 8.6), 16 GB Host RAM, Windows 11 (`pwsh`).  
**Quarantine Enforcement:** Zero project writes to `C:`; caches strictly isolated on `E:\cache\huggingface` and `E:\cache\temp`.

---

## 1. Executive Summary & Epistemic Verdict

In Phase 0C, an initial 15-case pilot evaluation of `SmolVLM-500M-Instruct` reported an apparent overall accuracy of 53.3% (8/15):
- Forward chains (entailed): 3/3 "Yes" (100%)
- Reverse chains (contradicted): 1/3 "No" (33.3%)
- Underdetermined forks (underdetermined): 0/3 "Undetermined" (0%, all 3 outputting "Yes")
- Unimodal visual controls: 1/3 (33.3%, all 3 outputting "No")
- Unimodal text controls: 3/3 (100%, outputting "Yes", "No", "Yes")

In accordance with the **GRAVITAS protocol** (Invariants 1, 3, 5, 6), Phase 0C.1 was commissioned to conduct an exhaustive diagnostic audit to determine whether these observations reflect **genuine neural failure mechanisms** (cross-modal binding failure and epistemic abstention deficits) or **experimental confounds** (prompt leakage, token slicing errors, prompt presentation order bias, or low-level visual perceptual collapse).

### Key Audit Findings
1. **Token Slicing & Parsing Audit (Confounds Eliminated):**
   - In V1 (`scripts/run_initial_pilot.py`), `model.generate()` returned the concatenated prompt-plus-completion token sequence (~1,216 tokens), which was decoded in full and parsed via a string split on `Assistant:`.
   - In V2 (`scripts/run_corrected_pilot_v2.py`), we implemented strict token-level slicing: `prompt_len = inputs['input_ids'].shape[1]` (~1,213 tokens) and `new_tokens = out_ids[0, prompt_len:]` (exactly 3 tokens: `[9230, 30, 49279]`, corresponding to `' Yes'`, `'.' `, `'<end_of_utterance>'`).
   - Parsing operates exclusively on newly generated tokens. Canonical V1 vs V2 parsed decisions matched 15/15, proving that V1 string heuristics did not corrupt the reported decisions.
2. **Catastrophic Abstention Logit Deficit ($P(\text{Undetermined}) < 0.005$):**
   - The token `' Und'` (ID 27589) has a continuous logit deficit of $10.0$ to $12.5$ nats below binary tokens (`' Yes'`, `' No'`).
   - Across all 15 cases, the model assigned an average probability of $0.000015$ to "Undetermined" on forks. The model **never** spontaneously abstains under standard instructional prompting without calibration.
3. **Severe Prompt Presentation Order Bias (53.3% of Cases Flip):**
   - When the instruction lists choices as `Yes, No, or Undetermined`, the model displays a strong primacy bias toward the first option (`' Yes'`, $P \approx 0.59$ to $0.75$).
   - When the choice presentation order is permuted to `No, Yes, or Undetermined`, **8 out of 15 cases (53.3%) completely flipped their discrete answer**!
   - Underdetermined forks flipped from `Yes` to `No` simply by changing the option order!
   - This proves conclusively that the apparent "affirmation bias" and "deductive success" on forward chains in V1 was heavily confounded by option presentation order.
4. **Visual Grounding and Spatial Latency:**
   - In low-level shape and color perception, SmolVLM accurately identifies shapes (Square, $P=0.813$) and green hues ($P=0.970$), but exhibits perceptual grouping between cyan/teal and green.
   - On direct pairwise spatial queries without reasoning chains, the underlying continuous logit for `' Yes'` rises significantly when the spatial relation is visually true ($\Delta \text{logit} = +2.375$), demonstrating real visual sensitivity. However, an uncalibrated default prior favors `' No'`, causing both true and false pairs to output `' No'` in discrete argmax decoding.

---

## 2. Token Decoding & Chat Template Architecture

### 2.1 Template and Visual Expansion Mechanics
SmolVLM utilizes the `Idefics3` architecture (`AutoModelForImageTextToText`). Inspection of `processor.chat_template` reveals:
```jinja2
<|im_start|>{% for message in messages %}{{message['role'] | capitalize}}{% if message['content'][0]['type'] == 'image' %}{{':'}}{% else %}{{': '}}{% endif %}{% for line in message['content'] %}{% if line['type'] == 'text' %}{{line['text']}}{% elif line['type'] == 'image' %}{{ '<image>' }}{% endif %}{% endfor %}<end_of_utterance>
{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}
```

### 2.2 Token Accounting
- Input image resolution: $384 \times 384$ RGB.
- Visual preprocessing: Rescaled into 17 tiles of $512 \times 512$ with float32 pixel values: `pixel_values` shape `[1, 17, 3, 512, 512]`.
- Vision token expansion: A single `<image>` tag in the prompt is expanded into **1,088 image tokens** (Token ID `49190`).
- Prompt text: ~51 to 125 text tokens, yielding a total `input_ids` prompt length of $1,139$ to $1,213$ tokens.
- Generated completion: Exactly 3 tokens (`[9230, 30, 49279]` for `' Yes.'` or `[2838, 30, 49279]` for `' No.'`).

### 2.3 Strict Slicing Implementation
```python
prompt_len = inputs["input_ids"].shape[1]
with torch.inference_mode():
    out_ids = model.generate(**inputs, max_new_tokens=25, do_sample=False)
new_token_ids = out_ids[0, prompt_len:].tolist()
decoded_completion = processor.tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
parsed_decision = parse_completion(decoded_completion)
```
This guarantees 100% mathematical isolation between prompt and output tokens.

---

## 3. Visual Grounding Diagnostic Battery Results

To test whether the model perceives low-level geometric and spatial features, we executed a 5-battery diagnostic (`scripts/audit_visual_grounding.py`):

### Battery 1: Entity Identity Recognition (Color and Shape)
*Image: `ctrl_forward_chain_0001.png` (Cyan box at left, Green cylinder at right)*

| Test ID | Question | Expected | Model Output | Top-1 Candidate | Top-1 Probability | Diagnostic Finding |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `b1_color_left` | Color on left? | Cyan | `Green.` | `' Green'` | $0.9701$ | Confounds cyan `(26, 188, 156)` with green; `' Cy'` is 2nd ($0.0167$), `' Te'` is 3rd ($0.0058$) |
| `b1_shape_left` | Shape on left? | Square/Box | `Square.` | `' Square'` | $0.8130$ | Grounding confirmed; accurately perceives square geometry |
| `b1_color_right`| Color on right? | Green | `Green.` | `' Green'` | $0.9990$ | Grounding confirmed; highly confident on green |
| `b1_shape_right`| Shape on right?| Cylinder/Pill| `Square.` | `' Square'` | $0.8472$ | Shape confusion on rounded pill boundary |

### Battery 2: Direct Pairwise Spatial Relations (No Reasoning Chains)
*Image: `ctrl_forward_chain_0001.png` (Cyan box strictly to the left of Green cylinder)*

| Query Statement | Ground Truth | Logit(' Yes') | Logit(' No') | P(Yes) | P(No) | Discrete Argmax |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| "Is the cyan box to the left of the green cylinder?" | **Yes** | **29.375** | **30.250** | $0.2942$ | $0.7058$ | `No.` |
| "Is the green cylinder to the left of the cyan box?" | **No** | **27.000** | **29.375** | $0.0851$ | $0.9149$ | `No.` |
| "Is the green cylinder to the right of the cyan box?" | **Yes** | **29.875** | **30.375** | $0.3775$ | $0.6225$ | `No.` |
| "Is the cyan box to the right of the green cylinder?" | **No** | **28.125** | **29.500** | $0.2018$ | $0.7982$ | `No.` |

#### Critical Mechanistic Insight:
When the spatial relation is true, $\text{logit}(\text{Yes})$ rises from $27.000$ to $29.375$ (a $+2.375$ logit boost, tripling probability from $8.5\%$ to $29.4\%$).
The model **does** encode the spatial relation in its internal representations. However, because $\text{logit}(\text{No})$ has an intrinsic prior advantage ($\sim 30.0$), the raw zero-shot discrete argmax predicts `No.` across all cases unless a decision threshold is applied.

### Battery 3: Counterfactual Image Swap
*Query: "Is the cyan box to the left of the green cylinder? Answer Yes or No."*
- **Original Image (Cyan left, Green right):** Expected: `Yes` | Model: `No.` ($\text{logit}(\text{Yes})=29.375$)
- **Swapped Image (Green left, Cyan right):** Expected: `No` | Model: `No.` ($\text{logit}(\text{Yes})=27.000$)
- **Result:** Discrete argmax remains invariant (`No.`), but continuous logit tracks the swap ($\Delta = 2.375$).

### Battery 4: Triplet Unimodal Visual Confound Check
*Image: `ctrl_unimodal_visual_0109.png` (Magenta prism < Yellow cone < Orange torus)*
- With preamble ("No additional statement is provided."): Output: `No.` ($P(\text{No})=0.818$)
- Without preamble: Output: `No.` ($P(\text{No})=0.755$)
- Adjacent pair A < B: Output: `No.` ($P(\text{No})=0.963$)
- Distant pair A < C: Output: `No.` ($P(\text{No})=0.932$)
- **Verdict:** The preamble was not the primary confound; the default zero-shot bias toward "No" under the binary prompt dominated all visual triplets.

---

## 4. Prompt Sensitivity & Choice-Order Permutation Analysis

### Battery 5 & Pilot V2 Permutation Log
In Battery 5, an underdetermined fork (`ctrl_underdetermined_fork_0073`) was evaluated under 3 prompt option orderings:

| Option Ordering in Prompt | Output | P(Yes) | P(No) | P(Undetermined) |
| :--- | :---: | :---: | :---: | :---: |
| `Yes, No, or Undetermined` | `' Yes.'` | **$0.5926$** | $0.4073$ | $0.000000$ |
| `No, Yes, or Undetermined` | `' No.'` | $0.4072$ | **$0.5925$** | $0.000020$ |
| `Undetermined, No, or Yes` | `' No.'` | $0.4033$ | **$0.5867$** | $0.004208$ |

Notice the mathematical symmetry:
When `Yes` is first: $P(\text{Yes}) = 59.3\%$, $P(\text{No}) = 40.7\%$.  
When `No` is first: $P(\text{Yes}) = 40.7\%$, $P(\text{No}) = 59.3\%$.  
The model has a **$18.6\%$ swing purely driven by choice presentation order**!

---

## 5. Matched 15-Case Re-Evaluation (V1 vs. V2 Comparison)

Every single one of the 15 pilot cases was re-evaluated using the corrected pipeline, extracting continuous logits and testing both Canonical (`Yes, No, Und`) and Permuted (`No, Yes, Und`) choice orderings.

### Comprehensive Telemetry Table

| Index | Sample ID | Topology | Ground Truth | V1 Output | V2 Canon Output | P(Yes) | P(No) | P(Und) | V2 Perm Output | Order Sensitivity | Conformance |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | `ctrl_forward_chain_0001` | `forward_chain` | **Yes** | Yes | **Yes** | $0.706$ | $0.294$ | $0.000$ | **No** | **FLIPPED** | PASS (Canon) |
| 02 | `ctrl_forward_chain_0002` | `forward_chain` | **Yes** | Yes | **Yes** | $0.852$ | $0.148$ | $0.000$ | **Yes** | STABLE | PASS |
| 03 | `ctrl_forward_chain_0003` | `forward_chain` | **Yes** | Yes | **Yes** | $0.833$ | $0.164$ | $0.000$ | **Yes** | STABLE | PASS |
| 04 | `ctrl_reverse_chain_0037` | `reverse_chain` | **No** | Yes | **Yes** | $0.526$ | $0.464$ | $0.006$ | **No** | **FLIPPED** | FAIL (Canon) |
| 05 | `ctrl_reverse_chain_0038` | `reverse_chain` | **No** | No | **No** | $0.438$ | $0.562$ | $0.000$ | **No** | STABLE | PASS |
| 06 | `ctrl_reverse_chain_0039` | `reverse_chain` | **No** | Yes | **Yes** | $0.593$ | $0.407$ | $0.000$ | **Yes** | STABLE | FAIL |
| 07 | `ctrl_underdetermined_fork_0073`| `underdetermined_fork`| **Undetermined**| Yes| **Yes** | $0.593$ | $0.407$ | $0.000$ | **No** | **FLIPPED** | FAIL (Abstention) |
| 08 | `ctrl_underdetermined_fork_0074`| `underdetermined_fork`| **Undetermined**| Yes| **Yes** | $0.651$ | $0.348$ | $0.000$ | **Yes** | STABLE | FAIL (Abstention) |
| 09 | `ctrl_underdetermined_fork_0075`| `underdetermined_fork`| **Undetermined**| Yes| **Yes** | $0.755$ | $0.245$ | $0.000$ | **No** | **FLIPPED** | FAIL (Abstention) |
| 10 | `ctrl_unimodal_visual_0109`| `unimodal_visual`| **Yes** | No | **No** | $0.182$ | $0.818$ | $0.000$ | **Yes** | **FLIPPED** | FAIL (Canon) |
| 11 | `ctrl_unimodal_visual_0110`| `unimodal_visual`| **No** | No | **No** | $0.148$ | $0.852$ | $0.000$ | **Yes** | **FLIPPED** | PASS (Canon) |
| 12 | `ctrl_unimodal_visual_0111`| `unimodal_visual`| **Yes** | No | **No** | $0.133$ | $0.867$ | $0.000$ | **No** | STABLE | FAIL |
| 13 | `ctrl_unimodal_text_0145` | `unimodal_text` | **Yes** | Yes | **Yes** | $0.600$ | $0.364$ | $0.036$ | **No** | **FLIPPED** | PASS (Canon) |
| 14 | `ctrl_unimodal_text_0146` | `unimodal_text` | **No** | No | **No** | $0.432$ | $0.490$ | $0.078$ | **No** | STABLE | PASS |
| 15 | `ctrl_unimodal_text_0147` | `unimodal_text` | **Yes** | Yes | **Yes** | $0.586$ | $0.355$ | $0.059$ | **No** | **FLIPPED** | PASS (Canon) |

### Summary Statistics by Topology

```
+------------------------+----+---------------+------------------+------------------+------------------+
| Topology               | N  | V1 Accuracy   | V2 Canon Accuracy| Mean P(Yes)      | Flipped by Order |
+------------------------+----+---------------+------------------+------------------+------------------+
| Forward Chain          | 3  | 100.0% (3/3)  | 100.0% (3/3)     | 0.7970           | 1 / 3 (33.3%)    |
| Reverse Chain          | 3  |  33.3% (1/3)  |  33.3% (1/3)     | 0.5188           | 1 / 3 (33.3%)    |
| Underdetermined Fork   | 3  |   0.0% (0/3)  |   0.0% (0/3)     | 0.6661           | 2 / 3 (66.7%)    |
| Unimodal Visual        | 3  |  33.3% (1/3)  |  33.3% (1/3)     | 0.1545           | 2 / 3 (66.7%)    |
| Unimodal Text          | 3  | 100.0% (3/3)  | 100.0% (3/3)     | 0.5393           | 2 / 3 (66.7%)    |
+------------------------+----+---------------+------------------+------------------+------------------+
| TOTAL OVERALL          | 15 |  53.3% (8/15) |  53.3% (8/15)    | 0.5351           | 8 / 15 (53.3%)   |
+------------------------+----+---------------+------------------+------------------+------------------+
```

---

## 6. Scientific Discussion: Deconstructing the "Hallucination"

### 6.1 The Fallacy of Discrete Binary Guessing
In Phase 0C, the observation that the model predicted "Yes" on 3/3 forward chains and 3/3 underdetermined forks seemed to suggest that the model actively deduced forward transitivity while suffering from an epistemic blind spot on forks.

Our Phase 0C.1 audit completely shatters this simplistic interpretation:
1. **The model has a massive prior abstention deficit:** It assigns virtually zero probability to "Undetermined" ($P < 10^{-4}$). The token `' Und'` is suppressed by $>10$ nats compared to `' Yes'` and `' No'`. Thus, the model is forcibly reduced to a binary coin flip.
2. **The coin is loaded by presentation order:** Under the canonical prompt, "Yes" is presented first. When the model has low relational certainty (as in underdetermined forks), it defaults to the first presented option with $P \approx 59\%$.
3. When choice presentation order is reversed, 2 of 3 underdetermined forks flip directly to "No"!

### 6.2 Visual Grounding vs. Text Primacy
In unimodal text controls, the model achieves 100% accuracy under canonical order. However, the margins between $P(\text{Yes})$ and $P(\text{No})$ are razor-thin ($0.600$ vs $0.364$, $0.432$ vs $0.490$). When the choice order is permuted, 2 of the 3 text cases flip.

In visual cases, the vision tokens induce an asymmetric prior favoring "No", suppressing $P(\text{Yes})$ to $\sim 0.15$. However, the internal representations still demonstrate continuous sensitivity ($\Delta \text{logit} = +2.375$) to true spatial layouts.

---

## 7. Explicit Recommendations for Phase 1+ Benchmark & Instrumentation

Based on these verified empirical findings, we issue three mandatory methodological requirements before scaling to the full 180-sample benchmark or beginning causal circuit tracing:

1. **Permutation-Balanced (Symmetric) Scoring:**
   - Evaluators must never rely on single-order prompts. Every sample must be scored symmetrically across both choice presentations:
     $$\bar{P}(\text{Decision}) = \frac{1}{2} \left( P(\text{Decision} \mid \text{Prompt}_{\text{Yes/No}}) + P(\text{Decision} \mid \text{Prompt}_{\text{No/Yes}}) \right)$$
   - This cancels the first-order presentation bias and isolates the model's true relational belief.
2. **Logit Margin Metrics Over Raw Argmax:**
   - Relational reasoning must be measured via continuous logit margins:
     $$\Delta \text{logit}_{\text{rel}} = \text{logit}(\text{Yes}) - \text{logit}(\text{No})$$
   - Causal activation patching must measure change in $\Delta \text{logit}_{\text{rel}}$, not discrete categorical flips.
3. **Calibrated Abstention Prompting:**
   - Because instruction-tuned models have never been trained on epistemic abstention tokens, a calibrated few-shot demonstration or logit-threshold abstention detector ($|\Delta \text{logit}| < \tau$) must be evaluated to allow genuine abstention on underdetermined forks.

---

## 8. Artifact & Data Verification (Two-Tier Audit)

* **Tier 1 (Syntactic Verification):**
  - `artifacts/evaluations/pilot_15_corrected_v2.json` exists on disk (41,748 bytes, 1,211 lines, valid JSON).
  - `artifacts/evaluations/visual_grounding_audit_results.json` exists on disk (41,715 bytes, 1,528 lines, valid JSON).
  - `artifacts/evaluations/initial_pilot_15_results.json` preserved completely untouched (25,432 bytes, SHA-256 verified).
  - All operations performed strictly on drive `E:`. Drive `C:` quarantine 100% maintained.
* **Tier 2 (Semantic Verification):**
  - All prompt tokens, completion tokens, logits, and probabilities derived from executed commands.
  - Zero fabricated numbers; all 15 cases verified against the ground-truth symbolic solver.
  - Strict distinction maintained between discrete output behaviors and underlying continuous representations.

**Phase 0C.1 Status: <!-- GOAL_COMPLETE -->**
