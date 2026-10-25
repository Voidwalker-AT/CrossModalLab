# University submission organization and hygiene inventory

Prepared October 25, 2026 from verified starting revision `28e23b98e2ad48ff8ea63cd5a028180e186eb907`, matching origin/main after fetch. The initial tracked-file inventory has 766 entries: `artifacts/` 390, `src/` 59, `scripts/` 72, `tests/` 46, `dashboard/` 54, `datasets/` 28, `configs/` 20, `paper/` 47, `docs/` 34 and `submission/` seven, plus root/environment files. These counts describe the starting revision, not a permanently fixed file total.

Scope: complete tracked-path/size inventory, targeted source/docs/config/manifests and selected dashboard evidence. Restricted mixed/final records were not opened; their sizes/paths came from filesystem/index metadata, not content inspection. No model/checkpoint was loaded. No files were moved or deleted. [The repository map](REPOSITORY_MAP.md) is the concise navigation entry point.

## Classification and dependencies

| Class | Existing location | Decision |
|---|---|---|
| Essential project files | README, requirements, pyproject, `.ulvo/`, agent rules | Retain versions/environment identities; improve only human navigation |
| Scientific implementation | `src/`, research `scripts/`, `tests/` | Preserve exact bytes; no source/test refactor or new experiment |
| Dashboard implementation | `dashboard/frontend/src/`, `dashboard/backend/` | Preserve behavior; add a README link and actual screenshots only |
| Dataset manifests/assets | `datasets/`, `configs/`, selected `dashboard/assets/` | Keep paths/hashes; no duplicate dataset tree, ground-truth edits or closed-record access |
| Experimental outputs | `artifacts/` | Preserve raw results, partial failures and source identities |
| Historical archives | Versioned local submission folders/ZIPs and checkpoint directories | Retain everything; no cache/checkpoint/archive cleanup |
| Submission materials | Existing `submission/2026-10-28/`, paper; new `submission/university_oct28/` | New concise navigation around the same evidence, not replacement of the frozen release |
| Temporary/generated | `.cache/`, frontend `node_modules/` and `dist/`, Python bytecode, local generated packages | Leave in place and ignored; do not commit or delete them |
| Never commit | Weights, optimizer binaries, environments, credentials, global WSCTL backups | Existing ignore rules and focused staged checks; no infrastructure mutation |

The server reads `dashboard/backend/artifacts.py` → original CE/CE+ES result JSON and append-only training metrics → selected `dashboard/assets/development_cases.json` and exact image copies. It validates prompt/image identities and both symbolic algorithms. The separate publication route reads allowlisted October 25 evidence; historical Smol diagnostics have their own source manifest. Scientific code resolves paths through the existing configuration architecture. The paper/figures/registries and older package tooling reference original paths. These dependencies are why there are no cosmetic moves.

## Hygiene observations

The largest tracked file is the October 25 development `records.jsonl`, 11,648,512 bytes. Next are historical mixed benchmark records (3,950,588 and 1,275,387 bytes), inspected as metadata only. Large tracked files are data/provenance rather than new weights; they are deliberately preserved. No tracked weight/optimizer/cache file is added by this task. Detailed largest-file metadata and staged checks are in [verification](../submission/university_oct28/verification/README.md).

Multiple ignored preflight/r1/r2 review distributions and earlier October 28 releases exist locally. They are versioned historical/generated outputs, not an invitation to delete them. The professor-facing folder adds six short guides and screenshots rather than another dataset/archive copy. Existing `.gitignore` exclusions cover `.cache/`, `node_modules/`, `dist/`, bytecode, model/checkpoint binaries and generated release ZIPs; no ignore-rule change is needed.

Twelve pre-existing untracked historical reports/checkpoint README/recovery metadata files are left untouched and unstaged. Global WSCTL registry backups and external WSCTL scripts remain local/external. The registration is used as-is; no global WSCTL, ULVO, Python or CUDA configuration changes occur.

## Documentation findings and traceable corrections

- The old root badge/phase table was stale and used empty badge links. The old claim that training was design-only contradicted later stored CE and five-update CE+ES results. The root README now summarizes the verified current state and links detailed history.
- Old source-checkout instructions could be mistaken for a prebuilt release. The new quick start distinguishes building from serving and states Node/npm prerequisites explicitly.
- `submission/2026-10-28/` is a historical release instruction set, not the new current assessor entry point. It remains unmodified and linked with that distinction.
- The design specifications include planned intervention/live-instrumentation capabilities; the map labels them design/history and directs readers to the implemented dashboard documentation.
- The paper workspace README still described a two-column layout, although the current HTML builder uses a single-column review layout. Only that README description/navigation was corrected; manuscript, figures, tables and claim registries remain unchanged.
- Historical asset-license tables are not individual-image rights clearance. The new limitation guide repeats the already recorded rights caution; no acquisition manifest is rewritten or new external photo redistributed.
- Scope remains explicit: 18 dashboard cases/six families; five CE+ES updates; separate Smol measurements; failed 108-forward screen; 9,216 variants/six families/72 images. The solver is not a neural explanation. Scientific raw evidence and the manuscript are unchanged.

The new workspace's checklist leaves partner hardware, real human rehearsal, author details, university format and final upload pending. Automated browser success is not those approvals. This is an organization/readiness improvement, not proof of international publication readiness.

The all-README local-link audit also identifies two preserved acquired-metadata issues: `datasets/benchmarks/mohobench/README.md` has an unmatched fence; `datasets/benchmarks/r_bench/README.md` refers to absent `assets/r-bench.png`. These are historical upstream/readme snapshots, not the dashboard inputs or the new assessment navigation. They were documented rather than rewritten or supplemented by an unapproved asset download. The new university-facing documents and root navigation have no unresolved local-link errors. A targeted Git attribute preserves new verification-log bytes so their recorded hashes survive checkout; existing source-hash rules are unchanged.
