# Selected bibliography and sentence support

Fourteen entries: thirteen selected from the prior verified library plus the specifically needed probe-control paper. Primary records for every selected entry were reopened on October 25. This is metadata/abstract or official-card verification, not a fresh full-paper audit. Prior selected full-text locators remain in the preserved `paper/prior_art_evidence_matrix.md` and original release. No third-party full paper is redistributed. Do not imply our models were evaluated on these external benchmarks.

| Key | Proposed supporting sentence / role | Primary record and depth | Limit |
|---|---|---|---|
| W1 | Compositional evaluation can require matching relations rather than isolated word recognition. | https://arxiv.org/abs/2204.03162 ; title/authors/abstract | Winoground is a different task; no borrowed experimental numbers |
| W18 | ARO evaluates attributes, relations and order in VLM representations. | https://arxiv.org/abs/2210.01936v3 ; metadata/abstract | Do not claim ARO was run locally |
| W2 | Spatial reasoning is a documented challenge for VLMs. | https://aclanthology.org/2023.emnlp-main.568/ ; official EMNLP 2023 record, DOI 10.18653/v1/2023.emnlp-main.568 | Not evidence of our particular model's mechanism |
| W19 | HallusionBench distinguishes language hallucination and visual-illusion challenges. | https://arxiv.org/abs/2310.14566v5 ; metadata/abstract | Cite preprint-version entry; do not assign an unverified proceedings field |
| W4 | Selective prediction includes an explicit reject option. | https://arxiv.org/abs/1901.09192 ; metadata/abstract | CrossModalLab has no trained/calibrated selective predictor or risk-coverage result |
| W5 | Reliable VQA studies abstaining rather than answering incorrectly. | https://arxiv.org/abs/2204.13631 ; metadata/abstract | Prior art, not a claim of novelty for abstention |
| W14 | Recent VLM work studies answerability and a representation-action gap. | https://arxiv.org/abs/2609.32653v1 ; metadata/abstract | 2026 preprint, venue status UNVERIFIED; do not claim our probe establishes that mechanism |
| W15 | Recent spatial work considers missing or misleading visual evidence. | https://arxiv.org/abs/2605.30557v2 ; metadata/abstract | 2026 preprint, venue status UNVERIFIED; distinct viewpoint/occlusion setting |
| QwenCard | The study uses the named Qwen3-VL-2B-Instruct checkpoint; repository records pin its revision. | https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct ; official model documentation | Card is not peer-reviewed evidence for our result; pinned local config establishes architecture details |
| SmolCard | SmolVLM2 combines a shape-optimized SigLIP encoder and SmolLM2 decoder. | https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct ; official card | Documentation, not a comparison experiment or causal explanation |
| LoRA | Low-rank adapters update a small parameter subset while the base is frozen. | https://arxiv.org/abs/2106.09685 ; metadata/abstract | Cite 2021 preprint version; no unverified venue normalization |
| W8 | QLoRA combines quantization with low-rank adaptation, including NF4 and double quantization. | https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract.html ; official NeurIPS 2023 record/abstract | Prior method, not our invention; local ES objective is a scoped implementation |
| ProbeControls | Probe performance must be interpreted alongside controls and probe capacity. | https://aclanthology.org/D19-1275/ ; official EMNLP-IJCNLP 2019, pp. 2733-2743, DOI 10.18653/v1/D19-1275; abstract | This linguistic probe study does not establish a VLM causal mechanism; no claim that its precise control task was reproduced |
| W7 | Activation-patching interpretation depends on metric and intervention design. | https://arxiv.org/abs/2309.16042 ; metadata/abstract, inherited selected-methods audit | Preprint-version citation; no claim of a completed causal circuit locally |

All empirical CrossModalLab numbers cite repository artifacts, not these papers. The selected BibTeX retains stable keys to ease integration. W8's selected note explicitly identifies the official NeurIPS 2023 record and arXiv identifier 2305.14314; its original library entry is preserved. Other preprint venue normalization remains intentionally outside scope. No dummy DOI or placeholder URL is supplied.

Omitted library entries are not invalidated: POPE, R-Bench, MoHoBench, R-Tuning and other broader comparisons are simply unnecessary for this bounded manuscript narrative. Add one back only if a specific final sentence needs it and its relevance/metadata are checked. Preserve the original 27-entry evidence library unchanged.
