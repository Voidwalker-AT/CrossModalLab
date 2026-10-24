# Natural-image evidence track — admission and annotation protocol

October 25, 2026. No natural-image evaluation was executed. Fresh local hash checks verify 30 COCO and 30 GQA candidate photographs against their historical manifests. The candidates remain unannotated; no new download, redistribution, fabricated incident or deployment claim occurs. Individual source IDs/URLs/hashes are retained in `artifacts/publication/20261025/analysis.json#natural_assets`.

## Rights and provenance gate

Historical dataset-level CC BY 4.0 strings do not verify individual photographs. The [official COCO terms](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm) distinguish annotation/website rights from image rights under their original Flickr terms. Do not export photographs in the publication package until each image's creator, original page, licence/version, permitted derivatives and attribution text are verified. Treat GQA/Visual Genome image rights separately too. Preserve old acquisition manifests and record corrected rights conclusions in a new registry. No legal clearance or blanket image licence is asserted here.

## Task admission

Use a small manually verified set of independent source photographs, retaining all related crops/text variants in one family. Document exact image bytes, dimensions, crop/transform coordinates and full-source identity. Identify visible objects with unambiguous referring expressions and bounding boxes; define left-of in the image coordinate plane rather than inferring 3D spatial order. Record occlusion, mirrored scenes, camera/viewpoint ambiguity and edge cases.

For genuine cross-modal composition, the image must establish only one relevant premise. If both queried objects are visibly ordered, the query may be answered directly: classify it as a visual-only control, not a cross-modal chain/fork. A textual premise about an unpictured third object must be explicitly supplied as a controlled assumption, not falsely described as an observed real-world fact. Retain all image-derived relations, including unintended cues, when checking sufficiency. Never label a visually decidable query U merely because a selected two-edge abstraction omitted visible evidence.

Two human annotators independently record visible evidence, textual assumption, query, ambiguity and justification before seeing model outputs; disagreement goes to a third adjudicator. Record original votes and agreement counts, not only the final label. Unresolved perception/rights/identity cases are excluded under the frozen rule, with reasons preserved. The symbolic oracle checks the accepted relational abstraction but cannot certify that the abstraction exhausts a photograph's evidence. AI suggestions cannot substitute for independent human verification.

## Collection and evaluation

Freeze the candidate list, inclusion decisions, complete evidence graphs, annotations, family split and hypotheses before model evaluation. Keep this natural-image track separate from synthetic geometry and historical development metrics. Obtain an explicit additional forward budget after admission; all current 12 additional forwards are spent. Report class/family counts, unsupported definite answers and inappropriate abstentions with family-level uncertainty only when sample support warrants it. This track is BLOCKED on individual rights verification and human annotation, not on software availability. New photography is an alternative only with consent and an agreed acquisition protocol.
