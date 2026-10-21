# Phase 2E — Large-Scale Dataset Strategy

**Mode:** Design only — **no dataset downloads in this phase.**  
**Holdout:** existing frozen test partitions remain untouchable for training.

---

## 1. Dataset shortlist (researched)

| Dataset | Source / citation | License / redistribution | Size (typical) | Role | Relation to CrossModalLab | Feasibility (E: 43 GB free, 16 GB RAM) |
|---|---|---|---|---|---|---|
| **Visual Genome** | Krishna et al., CVPR 2017; stanfordvl | Non-commercial research terms (check current ToS); images © owners | ~100 GB+ full; scene graphs large | Tier B natural relations | Dense relations; not epistemic-abstention | Full download **infeasible**; use **GQA-style subsets** only |
| **GQA** | Hudson & Manning, EMNLP 2019; stanfordqa | CC BY 4.0 (images via Visual Genome / dataset card) | Full questions ~GBs; images large | Tier B/C compositional VQA | Strong overlap with relational composition | **Feasible** as stratified subset (already have 30-image subset) |
| **Winoground** | Thrush et al., CVPR 2022, arXiv:2204.03162; `facebook/winoground` | Research-only HF gated access; Getty images | **~366 MB**, &lt;1K groups | Tier B/C evaluation | Word-order compositionality; **no** multi-premise chains/abstention labels | **Feasible** after HF access grant; eval-only |
| **ARO** | Yuksekgonul et al., ICLR 2023, arXiv:2210.01936 | Public benchmark subsets (verify per-repo) | Moderate | Tier B eval | 1-hop relation/attribution “bag-of-words” | Feasible eval subset |
| **HallusionBench** | Guan et al., CVPR 2024; `tianyi-lab/HallusionBench` | **Apache 2.0** (per project asset manifest) | Question JSON + 19 stimuli in-repo already | Tier C | Language hallucination vs visual illusion; **paired controls** | **Already partially onboarded**; full suite optional |
| **VizWiz** | Gurari et al., CVPR 2018; vizwiz.org | Dataset ToU — typically non-commercial research; **verify before redistrib** | Large (photos + Q/A) | Tier C | Real-world unanswerable / insufficient evidence | Storage OK if sharded; license gate before train use |
| **MoHoBench** | yanxuzhu/MoHoBench, AAAI 2026 | **MIT** (in-repo) | Subset JSON present | Tier C | Unanswerable VQA honesty | **Onboarded subset** |
| **R-Bench** | mrwu-mac/R-Bench, ICML 2024 | **Apache 2.0** | Archive not stored | Tier C | Relational reasoning | Metadata-only until approved fetch |
| **Project synthetic (v1/v2)** | CrossModalLab generators | Project-generated | 720+ records in Git | Tier A | Exact chain/fork labels | **In-house; primary training source** |

### Annotation / preprocessing notes
- **Tier A:** zero external annotation cost; solver-verified; render on demand.  
- **Tier B:** GQA relationship questions need filtering scripts; Winoground needs gated HF token (no download this phase).  
- **Tier C:** HallusionBench/MoHo already structured for consistency metrics; VizWiz needs answerability label mapping.

### Leakage risks
- Never train on **contrast-v2 test** or **Phase 1C final holdout**.  
- Scene-graph images reused across splits → group by **image id / scene id**.  
- Template paraphrases must stay within family groups.

---

## 2. Storage & annotation budget

| Tier | 5k | 20k | 50k |
|---|---:|---:|---:|
| A synthetic (PNG ~2 KB + JSONL) | ~15–30 MB | ~60–120 MB | ~150–300 MB |
| B natural (≤384² JPEG ~40 KB) | ~200 MB | ~800 MB | ~2 GB |
| C realistic (mixed) | ~100 MB | ~400 MB | ~1 GB |
| **Total disk (est.)** | **~0.3–0.5 GB** | **~1.3–2 GB** | **~3.5–5 GB** |

**Annotation cost (human):** Tier A 0; Tier B/C if re-labeling: assume 5–15 s/example → 5k ≈ 7–21 person-hours; prefer **reuse official labels** (0 marginal).

**Recommended incremental scale (if approved later):**
1. **5,000** Tier A (+ existing contrast-v2) for LoRA feasibility smoke  
2. **20,000** A+B for train/val with group-disjoint natural images  
3. **50,000** only if 20k shows generalization to held-out templates/scenes  

---

## 3. Three-tier design summary

| Tier | Content | Labels | Splits |
|---|---|---|---|
| **A** | Controlled relational chains/forks; counterfactual families | Symbolic solver + oracle | Family-disjoint 70/15/15 |
| **B** | GQA/Winoground/ARO/COCO-derived spatial | Official + optional re-query | Image-disjoint |
| **C** | HallusionBench/MoHo/VizWiz-style insufficient evidence | Official consistency / answerability | Protocol-disjoint |

Existing frozen holds remain evaluation-only.

**Artifacts:** this document + `docs/phase_2e_novelty_audit.md` + `docs/phase_2e_training_feasibility.md`.
