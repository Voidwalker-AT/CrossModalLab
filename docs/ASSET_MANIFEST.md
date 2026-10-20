# Asset Manifest — CrossModalLab

Classification of repository and external assets. Large binaries stay **outside Git** unless explicitly approved. This manifest does not download anything.

## A. Small source-controlled benchmark metadata (IN GIT)

| Asset | Path | License | Notes |
|---|---|---|---|
| Controlled benchmark JSONL | `datasets/controlled/controlled_benchmark_180.jsonl` | Project-generated | Symbolic ground truth; ~200 KB |
| Controlled manifest | `datasets/controlled/manifests/controlled_180_manifest.json` | Project-generated | Includes SHA-256 image checksums |
| Diagnostic controls manifest | `datasets/controlled/manifests/diagnostic_controls_manifest.json` | Project-generated | |
| GQA subset manifest | `datasets/real_world/gqa/manifests/gqa_30_manifest.json` | CC BY 4.0 (source) | |
| COCO subset manifest | `datasets/real_world/coco/manifests/coco_30_manifest.json` | CC BY 4.0 (source) | |
| HallusionBench subset JSON + manifest | `datasets/benchmarks/hallusionbench/` | Apache 2.0 | |
| MoHoBench subset JSON + manifest | `datasets/benchmarks/mohobench/` | MIT | |
| R-Bench metadata + LICENSE | `datasets/benchmarks/r_bench/` | Apache 2.0 | Full archive **not** stored |
| Acquisition manifests | `datasets/metadata/*.json` | Project records | Historical local paths may reference `E:\cache\...` on the primary machine |

## B. Generated controlled images (LOCAL; Gitignored by default)

| Asset | Path | In Git? |
|---|---|---|
| 180 controlled PNG images | `datasets/controlled/images/` | No (see `.gitignore`) |
| Diagnostic PNG images | `datasets/controlled/diagnostic_images/` | Small; currently tracked if present |

Regenerate with:

```bash
ulvo run . -- python scripts/generate_controlled_benchmark.py   # Mode A
python scripts/generate_controlled_benchmark.py                 # Mode B (needs Pillow)
```

## C. External / real-world image subsets (LOCAL; large files Gitignored)

| Asset | Source | License | Local path | Acquisition |
|---|---|---|---|---|
| GQA / Visual Genome 30 images | Stanford GQA | CC BY 4.0 | `datasets/real_world/gqa/images/` | `scripts/acquire_real_world_datasets.py` |
| COCO val2017 30 images | MS COCO | CC BY 4.0 | `datasets/real_world/coco/images/` | same |
| HallusionBench stimuli | tianyi-lab/HallusionBench | Apache 2.0 | `datasets/benchmarks/hallusionbench/images/` | same |
| R-Bench archive | mrwu-mac/R-Bench | Apache 2.0 | **Not stored** | Metadata only; download independently if needed |
| MoHoBench subset | yanxuzhu/MoHoBench | MIT | JSON only under `datasets/benchmarks/mohobench/` | same |

## D. Downloaded model checkpoints (OUTSIDE REPO; never commit)

### Primary research model

| Field | Value |
|---|---|
| Name | `smolvlm2-2.2b-nf4` |
| Hugging Face ID | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` |
| Revision (pinned) | `482adb537c021c86670beed01cd58990d01e72e4` |
| License | Apache 2.0 |
| Size | ~8.375 GB |
| Primary local path | `E:/cache/huggingface/models--HuggingFaceTB--SmolVLM2-2.2B-Instruct/snapshots/482adb537c021c86670beed01cd58990d01e72e4` |
| Checksums | `datasets/metadata/smolvlm2_acquisition_manifest.json` |
| Required? | **Required for GPU VLM experiments**; optional for symbolic CPU tests |
| Acquisition | `scripts/acquire_smolvlm2.py` (respects `CROSSMODAL_HF_HOME` / `HF_HOME`) |

### Legacy pilot model

| Field | Value |
|---|---|
| Hugging Face ID | `HuggingFaceTB/SmolVLM-500M-Instruct` |
| Revision | `a7da5b986cb59b408707209984f360a5f4ad7e47` |
| License | Apache 2.0 |
| Local path (primary) | `E:/cache/huggingface/hub/models--HuggingFaceTB--SmolVLM-500M-Instruct/snapshots/...` |
| Checksums | `datasets/metadata/model_acquisition_manifest.json` |
| Required? | Optional (historical pilot artifacts only) |

## E. Activation dumps (OUTSIDE GIT)

| Asset | Path | In Git? |
|---|---|---|
| Activation tensors | `artifacts/activations/` | No (ignored) |

## F. Evaluation results & reports (IN GIT when small JSON/MD)

| Asset | Path |
|---|---|
| Pilot / audit / diagnostics JSON | `artifacts/evaluations/*.json` |
| Markdown scientific reports | `artifacts/reports/*.md` |
| Hardware telemetry | `artifacts/telemetry/*.json` |

Do not overwrite another contributor’s run artifacts; write new filenames with run IDs.

## G. Publication figures (future)

Reserved under `artifacts/figures/` (create when needed). Large binaries require explicit approval before Git tracking.

## Environment variables for asset locations

| Variable | Purpose |
|---|---|
| `CROSSMODAL_HF_HOME` / `HF_HOME` | Hugging Face cache root |
| `CROSSMODAL_MODEL_SNAPSHOT_DIR` | Explicit snapshot path override |
| `CROSSMODAL_DATASETS_DIR` | Dataset root override |
| `CROSSMODAL_ARTIFACTS_DIR` | Artifact root override |

See `configs/local.env.example` and `src/config/paths.py`.

## Rules

1. Never commit `.safetensors`, `.bin`, `.pt`, multi-GB images, or HF cache trees.
2. Never redownload assets that already exist at a resolvable path.
3. Never move the primary machine’s `E:/cache` weights without explicit authorization.
4. Second developers obtain models/datasets independently using the acquisition scripts and this manifest.
5. Checksums in acquisition manifests are historical records from the primary machine download; re-verify if integrity is in doubt.
