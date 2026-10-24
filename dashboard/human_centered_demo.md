# Five minutes: from an image to an unsupported answer

Start with `wsctl run crossmodallab dashboard`, or use the ordinary-Python launcher documented in README.md. Select **Start the five-minute guided tour**. Next/Previous reuse the existing research views and reset the featured Qwen case to `xc_ho_0011::underdetermined`, CE+ES step five. Use Tab and Enter for the controls. Normal navigation remains available; Exit returns to normal exploration.

This is a presentation script, not evidence of a completed human rehearsal. A projector rehearsal and partner-machine test remain pending.

| Time | Tour stop | Explain and demonstrate |
|---|---|---|
| 0:00–0:30 | Research problem | A confident answer is not always justified. We test whether a vision-language model can tell false evidence from insufficient evidence. These are controlled rendered benchmark inputs used in actual experiments. |
| 0:30–1:10 | Image and text | Point to the original golden disc and teal wedge. The image gives disc left of wedge; the text adds crimson block left of wedge. Ask whether the disc is left of the block. Cross-modal reasoning combines the two sources. |
| 1:10–1:55 | Correct logic | Select both compatible-order buttons. Disc–block–wedge and block–disc–wedge both satisfy the premises. The answer is Undetermined, not No. Green and gold arrows are facts; purple is only the question. The graph is a symbolic oracle, not the model’s thought process. |
| 1:55–2:30 | Recorded failure | The CE+ES five-update checkpoint generated exactly `No` for this case. A definite answer is unsupported. No internal neural measurements have been recorded for this Qwen example; the cause remains unverified. |
| 2:30–3:05 | Checkpoints | All four standard-prompt checkpoints give an extracted `No` on this fork. CE step five has `No` plus fifteen U+FFFD characters; CE step 50 has a leading space, `No`, then fifteen U+FFFD characters. Raw strings are preserved. Formatting and correctness are separate; no improvement appears on this case. Steps are optimizer updates, not epochs. |
| 3:05–4:00 | Historical neural evidence | The upper architecture view is conceptual. Below it, separately labeled historical SmolVLM2 measurements show vector magnitudes and external probes. On the default last-token probe, balanced accuracy is 33.3%, 51.9%, 77.8%, and 74.1% at layers 3, 12, 18, and 23. These are 27 development validation cases in nine families; the layer and settings were development-selected. Decodability is not proof of model usage; these are not Qwen activations. |
| 4:00–5:00 | Results and limits | Qwen has 18 development cases, six families and six cases per class. CE+ES five updates: 9/18 correct; CE five updates: 7/18. The paired descriptive family interval includes zero. Two correct abstentions coexist with four incorrect abstentions. CE step 50 gets 12/18 under the parser but zero underdetermined cases and zero exact-format outputs. No CE+ES step-50 result exists. Show sources in Reproducibility after exiting the tour. |

Optional controls after the clock: `xc_ho_0011::entailed` is Yes (forward chain); `xc_ho_0011::contradicted` is No (reverse chain). The CE+ES five-update checkpoint records exactly those answers. This shows why the word No can be justified on one case and unsupported on another.

If the application cannot start, open the versioned evidence folder’s Gallery.html. State that this is the offline fallback. It contains actual exported figures and screenshots, not live inference. Do not launch training, download a model, or replace missing measurements with an illustration presented as data.

Keep visible: one exploratory seed, reused development cases, unequal checkpoint budgets, malformed-output sensitivity, no final holdout evaluation, no established causal circuit. IEEE suitability and scientific novelty remain unverified.
