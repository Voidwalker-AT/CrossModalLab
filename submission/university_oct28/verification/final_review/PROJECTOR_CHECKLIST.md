# Projector and five-minute presenter checklist

Automated browser rehearsal is complete; tick the following only after an actual human check.

- [ ] Before entering the room: run `wsctl run crossmodallab dashboard` on the registered machine, or the ordinary-Python alternative in the [demo guide](../../DEMO_GUIDE.md). Open the health endpoint; expect `read_only: true`, `gpu_loaded: false`. Do not launch a model.
- [ ] Set the projector to 1920×1080 if available; test from the back row. Use browser zoom if necessary. Screenshots alone cannot verify room lighting or viewing distance.
- [ ] Click **Start the five-minute guided tour**. Confirm CE+ES step 5 and `xc_ho_0011::underdetermined`. Say the first-stop explanation in plain language: combining image and text still leaves two possible orders; correct Undetermined, recorded No.
- [ ] Show the original yellow disc/teal wedge image and the textual crimson-block premise. Enlarge the image if needed; Escape closes it.
- [ ] At stop 3, enlarge the symbolic graph. Name A=golden disc, B=teal wedge, C=crimson block. Green/image and gold/text arrows are facts; purple is only the question. Click both countermodels, then Escape.
- [ ] At stops 4–5, show exact recorded No and the four-checkpoint table. CE50 is not CE+ES50. Separate malformed CE output from semantic correctness.
- [ ] At stop 6, show the conceptual layer control and historical Smol plots. Say explicitly: different model; correlations, no established neural cause for the Qwen failure. Enlarge plotted axes rather than relying on compact previews.
- [ ] At stop 7, state 18 development cases/six families/one seed; 9/18 versus 7/18, interval includes zero. No demonstrated superiority or final-holdout claim.
- [ ] Exit tour and open Reproducibility; show hashes/read-only scope. The failed screen is separate from the completed small pilot. The 9,216 generated variants are dependent, not independent samples.
- [ ] Time the complete route to five minutes using the existing [speaking schedule](../../DEMO_GUIDE.md). Practise Previous/Next, Enter/Space, enlargement and Escape; keep room for questions.
- [ ] Keep the preserved static gallery/screenshots available; announce any static fallback. Do not alter the frozen release or try live inference to recover a presentation problem.
- [ ] Ask one unfamiliar reader to explain Undetermined versus No after the first minute. Record actual feedback; automated assertions do not establish comprehension.
- [ ] Run the setup on the partner's physical machine before claiming second-machine success.

Readiness is **CONDITIONAL GO** until the room, timed speech and partner checks are performed. No new experiments are needed for this presentation route.
