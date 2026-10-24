# Publication research progress

## October 25, 2026 — controlled benchmark and isolated pilot

Starting revision: `7b932d3a1dbdac262e1515e4a47c42380e078700`; local branch and origin/main matched after fetch. The 12 pre-existing untracked historical files remain untouched. University title and immutable October 28 archive remain preserved.

**Question:** Can the task distinguish composition from conditional role shortcuts, and can the two pinned models be scored safely on identical inputs?

**Implemented:** full-factorial development generator and eight diagnostic baselines; separate-process model workers with immutable admission and a 12-forward aggregate cap; exact token-anchor alignment/no-op CPU controls; objective-gradient tests; provenance analysis; a read-only dashboard publication update.

**Raw evidence:** `artifacts/publication/20261025/benchmark/`, `pilot/` and `analysis.json`. The new benchmark contains 9,216 records in six dependent families, not 9,216 independent samples. Three families fit lookup baselines and three evaluate them, all exposed development. Nonsemantic audited feature ceilings are 1/3 balanced accuracy; each oracle-parsed unimodal role feature reaches 2/3; joint symbolic oracle reaches 1. This is an information diagnostic, not VLM capability or exhaustive leakage clearance.

**Pilot:** all 12 additional forwards completed. Smol: Yes/Yes on gold No/Undetermined; Qwen: No/Yes. Qwen scores exactly replicate the earlier six-forward record. All 657 Smol and 625 Qwen parameter comparisons pass. Each process exits; physical GPU use returns to its admissible baseline before the next worker. Smol peak allocated 3,330.529 MiB, peak sampled physical use 3,876 MiB and minimum sampled available RAM 4,748.219 MiB. Qwen corresponding values 1,575.535, 1,905 and 4,803.25 MiB. Sampling can miss transients. No optimizer, backward model pass, generated completion or intervention occurred.

**Scientific result:** both models select an unsupported definite answer on this fork; neither pair passes the prespecified opposed No/U-margin causal screen. This one-family result is exploratory and does not justify three-class BA, statistical superiority, a failure mechanism or generalization.

**Checks:** 13 new CPU tests passed after correcting one test's keyword-only API call. Initial guarded full discovery: 332 tests, ULVO 329 passes/3 restricted-data skips; ordinary Python 235 passes/97 skips. Both recorded zero restricted-open attempts. Frontend TypeScript and production build pass. Final verification logs and integration revisions are recorded with the subsequent release update.

**Next:** complete manuscript/figure/reproducibility updates; keep the spent 12-forward budget closed. Additional inference, training, GPU interventions and frozen final evaluation each retain their stated approval gates.
