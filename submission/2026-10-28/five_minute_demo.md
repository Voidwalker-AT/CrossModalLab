# Five-minute live demonstration

Target duration: **5:00**. This is a timed presentation script, not a claim that a human rehearsal has already occurred. All quoted predictions below come from [the exact-record companion](demo_examples.json); no live inference is used. Allow about 20 seconds across transitions. If time is short, omit the CE-50 format illustration and keep the limitations.

## Preparation (before the clock)

Run `python dashboard/backend/server.py --port 8765`, open localhost:8765 and choose **Failure Investigator**, **CE + ES · step 5**, all logical states, errors-only unchecked. Search `xc_ho_0011`. Select its **Underdetermined** case. Keep the graph query visible. Confirm the full ID is `xc_ho_0011::underdetermined`. The `ho` substring is an entity-family naming convention; the record's actual split is **validation**, not final holdout. Start at the top of the page.

## 0:00–0:40 — Motivation and the real input

“CrossModalLab asks whether a vision-language model distinguishes a false conclusion from a conclusion that the evidence cannot determine. That distinction matters whenever an assistant describes relationships from incomplete visual information. Our controlled benchmark is a diagnostic study; it does not establish readiness for navigation or other real applications.

“Here the image says the golden disc is left of the teal wedge. The text says the crimson block is left of the teal wedge. The question asks whether the golden disc is left of the crimson block. Cross-modal reasoning means combining the image premise and the textual premise to answer that question.”

Action: point to the original stored image, textual premise and query; scroll to the three panels.

## 0:40–1:40 — The logical fork and unsupported answer

“Call these objects A, B and C. We know A is left of B and C is left of B. We do not know the order of A and C. Both A–C–B and C–A–B satisfy the supplied premises. The correct label is therefore Undetermined, not No.”

Action: select both compatible-order buttons. In Evidence & Reasoning, point to the solid visual edge and dashed textual edge; toggle the purple query. It is a question, not an extra premise.

“For recorded sample `xc_ho_0011::underdetermined`, frozen Qwen and the five-step CE-plus-ES checkpoint both generated exactly `No`. This is an unsupported definite answer. The symbolic solver establishes the consequence of the supplied premises, independently checked by reachability and enumeration of consistent orderings. It does not reveal what the Transformer internally computed.”

## 1:40–2:20 — Entailed and contradicted controls

Action: return to Failure Investigator. Search remains `xc_ho_0011`. Select **Entailed**, then **Contradicted**.

“The entailed control is `xc_ho_0011::entailed`: the forward chain A–B–C entails Yes. The contradicted control is `xc_ho_0011::contradicted`: C–B–A rules out A being left of C, so No is justified. The graph highlights only supplied premise paths. The same word No can be correct on a reverse chain and unsupported on a fork.”

For the contradicted case, CE+ES step five generated exactly `No`. Optional: switch to **CE · step 50**. Its parsed answer is No, but its raw output is a leading space, `No`, then fifteen U+FFFD replacement characters (`" No" + "\uFFFD" × 15`). Show the unedited output. This is semantically correct under the historical prefix parser and **malformed** under exact-label formatting. Never read it as a clean `No` generation. The companion JSON preserves the actual characters, not this explanatory shorthand.

## 2:20–3:15 — What the results show

Action: open **Experiment Analytics**.

“These are 18 recorded development cases in six families, with six cases per class. Frozen Qwen has balanced accuracy 7/18, or 38.9%. CE at step five is also 7/18; CE-plus-ES at step five is 9/18, or 50%. The difference is just two cases, with a descriptive paired family-bootstrap interval that includes zero. It does not establish superiority.

“Historical CE at step 50 reaches 12/18 under the parser, but gets none of the six underdetermined cases right and has no exact-label-compliant outputs. Training budgets differ. Five-step losses come from different examples and are not a fixed-input learning curve. We preserve these failures instead of presenting a single favorable accuracy number.”

## 3:15–4:00 — What the Transformer view means

Action: open **Architecture Explorer**, select a decoder layer and then **Candidate scoring**.

“This view explains the conceptual image-to-token and Transformer pipeline, using the stored model identity. Layer selection is an educational architecture view, not measured neuron activity. Attention weights and external probes would be correlational evidence. Existing intervention controls do not establish a relation-specific causal circuit.

“The training pilot scores the complete candidate answers. Undetermined spans two tokens, so scoring only its first token would be wrong. The evidence-margin objective is exploratory; summed likelihoods also have token-length effects.”

## 4:00–5:00 — Reproducibility, contribution and limits

Action: open **Reproducibility**, point to model revision, dataset and source hashes; download provenance.

“The dashboard replays stored outputs and original images with hash checks. It runs on ordinary Python without loading a model or using a GPU. Our contribution for this milestone is a working scientific workbench that links inputs, formal ground truth, raw answers, class-sensitive metrics and reproducibility information.

“The CE-plus-ES run remains paused after five steps. No new training or final-holdout evaluation was performed for this demonstration. Generalization, calibration, method superiority and a causal mechanism remain unverified. A selected linear probe shows accessibility, not model usage. Our preliminary paper reports those limits. Next are independent partner reproduction, preregistered evaluation and stronger controls toward the December research target.”

End at 5:00. Invite questions; do not start training or click into unavailable live inference.

## Recovery and likely questions

- Port unavailable: run on 8766 and use that URL. Never terminate an unknown process.
- Runtime failure: show the packaged PNG figures and inspected screenshots, state that this is a fallback and explain the observed failure honestly.
- “Are the examples real photographs?” No. These are actual controlled rendered benchmark stimuli used in recorded experiments, not fabricated tutorial results. They are distinct from quarantined synthetic demonstrations.
- “Does the model think along this graph?” UNVERIFIED. The graph is symbolic ground truth, not a measured internal trace.
- “Did ES beat CE?” Not established. One five-step development pilot is insufficient, and the interval includes zero.
- “Does it run on the partner laptop?” Local ordinary-Python/package startup was tested on this primary machine. Physical second-machine verification remains pending.
