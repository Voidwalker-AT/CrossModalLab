"""Confound-controlled cross-modal contrast dataset (Phase 1E / v2).

All three logical labels share ONE experimental condition id:
CROSS_MODAL_TWO_PREMISE

Primary query is always: "Is A to the left of C?"

Label is determined solely by symbolic premises (visual + textual), never by
condition name, filename suffix, or prompt boilerplate that differs by class.

Visual canvas shows only two entities (never C for chain-style premises).
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw

from src.config.paths import REPO_ROOT
from src.symbolic.relations import DirectionalRelation, Modality, Premise, Query
from src.symbolic.solver import SymbolicSolver
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "cb_helpers", REPO_ROOT / "scripts" / "generate_controlled_benchmark.py"
)
_helpers = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_helpers)
ENTITY_PALETTE = _helpers.ENTITY_PALETTE
draw_shape = _helpers.draw_shape

SCHEMA_VERSION = "cross_modal_contrast_v2"
CONDITION_ID = "CROSS_MODAL_TWO_PREMISE"
DEFAULT_SEED = 20261031
LABELS = ("Yes", "No", "Undetermined")
VARIANTS = ("entailed", "contradicted", "underdetermined")

ID_ENTITIES = (
    "blue_cube",
    "red_sphere",
    "green_cylinder",
    "yellow_cone",
    "purple_pyramid",
    "orange_torus",
    "cyan_box",
    "magenta_prism",
)
HELD_OUT_ENTITIES = (
    "silver_hexagon",
    "amber_ring",
    "teal_wedge",
    "violet_star",
    "crimson_block",
    "emerald_oval",
    "golden_disc",
    "indigo_capsule",
)

# Linguistic paraphrases for the textual premise (balanced across labels via rotation)
PARAPHRASES = (
    "The {y} is to the left of the {z}.",
    "The {z} stands to the right of the {y}.",
    "Notice that the {y} appears left of the {z}.",
    "Spatially, {y} is left of {z}.",
)

# Query phrasings (same semantic query; balanced)
QUERY_FORMS = (
    "is {a} to the left of {c}?",
    "Is the {a} left of the {c}?",
    "Based on the evidence, is {a} left of {c}?",
)


@dataclass(frozen=True)
class ContrastFamily:
    family_id: str
    a: str
    b: str
    c: str
    split: str
    entity_pool: str
    paraphrase_idx: int
    query_idx: int
    layout_seed: int


def _solve(premises: List[Premise], query: Query) -> Tuple[str, str, bool]:
    res = SymbolicSolver.solve(premises, query, verify_with_oracle=True)
    return res.epistemic_status.value, res.explanation, res.algorithm_agreement


def variant_premises(variant: str, a: str, b: str, c: str) -> Tuple[List[Premise], Tuple[str, str]]:
    """Return premises and visual (left, right) entity pair.

    Never places C on the visual canvas for these constructions.
    """
    if variant == "entailed":
        # Visual R(A,B); Text R(B,C) => R(A,C) entailed
        return [Premise(a, b, DirectionalRelation.LEFT_OF, Modality.VISUAL),
                Premise(b, c, DirectionalRelation.LEFT_OF, Modality.TEXTUAL)], (a, b)
    if variant == "contradicted":
        # Visual R(B,A); Text R(C,B) => R(C,B)∧R(B,A) => R(C,A) => ¬R(A,C)
        return [Premise(b, a, DirectionalRelation.LEFT_OF, Modality.VISUAL),
                Premise(c, b, DirectionalRelation.LEFT_OF, Modality.TEXTUAL)], (b, a)
    if variant == "underdetermined":
        # Visual R(A,B); Text R(C,B) => A,C both left of B; R(A,C) underdetermined
        return [Premise(a, b, DirectionalRelation.LEFT_OF, Modality.VISUAL),
                Premise(c, b, DirectionalRelation.LEFT_OF, Modality.TEXTUAL)], (a, b)
    raise ValueError(variant)


def label_for_variant(variant: str) -> str:
    return {"entailed": "Yes", "contradicted": "No", "underdetermined": "Undetermined"}[variant]


def build_prompt(a: str, b: str, c: str, paraphrase_idx: int, query_idx: int) -> str:
    def fmt(e: str) -> str:
        return e.replace("_", " ")

    # Textual premise subjects depend on variant; prompt text is built from
    # the *textual* premise only (visual is in the image). Caller passes the
    # actual textual pair via paraphrase template arguments stored in record.
    raise NotImplementedError("Use build_prompt_with_text_pair")


def build_prompt_with_text_pair(
    a: str,
    c: str,
    text_left: str,
    text_right: str,
    paraphrase_idx: int,
    query_idx: int,
) -> str:
    def fmt(e: str) -> str:
        return e.replace("_", " ")

    tmpl = PARAPHRASES[paraphrase_idx % len(PARAPHRASES)]
    # Support both "X left of Y" and "Y right of X" phrasings
    if "{z}" in tmpl and "right of" in tmpl:
        premise = tmpl.format(y=fmt(text_right), z=fmt(text_left))
        # "Z is to the right of Y" means Y left of Z — but our template uses y left of z
        # For right-of phrasing: if text_left L text_right, say "{text_right} is to the right of {text_left}"
        # Rebuild simply:
        premise = f"The {fmt(text_right)} is to the right of the {fmt(text_left)}."
    else:
        premise = tmpl.format(y=fmt(text_left), z=fmt(text_right))

    q = QUERY_FORMS[query_idx % len(QUERY_FORMS)].format(a=fmt(a), c=fmt(c))
    return (
        f"{premise}\n\n"
        f"Question: Based on the image and statement, {q[0].upper() + q[1:] if q[0].islower() else q}\n"
        "Carefully evaluate whether the question is entailed (Yes), contradicted (No), "
        "or underdetermined from the evidence (Undetermined).\n"
        "Answer directly with: Yes, No, or Undetermined."
    )


def render_canvas(
    left: str,
    right: str,
    path: Path,
    rng: random.Random,
    *,
    bg_jitter: int = 0,
    gap_scale: float = 1.0,
) -> Dict[str, Any]:
    bg = (
        248 + rng.randint(-bg_jitter, bg_jitter),
        249 + rng.randint(-bg_jitter, bg_jitter),
        250 + rng.randint(-bg_jitter, bg_jitter),
    )
    bg = tuple(max(0, min(255, int(v))) for v in bg)
    img = Image.new("RGB", (384, 384), color=bg)
    draw = ImageDraw.Draw(img)
    obj_w = int(56 + rng.randint(0, 12))
    obj_h = int(56 + rng.randint(0, 12))
    lx1 = rng.randint(36, 110)
    ly1 = rng.randint(130, 190)
    lx2, ly2 = lx1 + obj_w, ly1 + obj_h
    gap = int((rng.randint(40, 90)) * gap_scale)
    rx1 = lx2 + gap
    ry1 = rng.randint(130, 190)
    rx2, ry2 = rx1 + obj_w, ry1 + obj_h
    if rx2 > 350:
        rx1 = 350 - obj_w
        rx2 = 350
        assert lx2 < rx1
    draw_shape(draw, ENTITY_PALETTE[left]["shape"], ENTITY_PALETTE[left]["color"], (lx1, ly1, lx2, ly2))
    draw_shape(draw, ENTITY_PALETTE[right]["shape"], ENTITY_PALETTE[right]["color"], (rx1, ry1, rx2, ry2))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "PNG")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "image_sha256": digest,
        "image_file": _rel(path),
        "entities_shown": {left: "left", right: "right"},
        "bboxes": {
            left: [lx1, ly1, lx2, ly2],
            right: [rx1, ry1, rx2, ry2],
        },
        "bg_rgb": list(bg),
        "object_size": [obj_w, obj_h],
        "gap_px": rx1 - lx2,
    }


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def plan_families(n_families: int = 240, seed: int = DEFAULT_SEED) -> List[ContrastFamily]:
    rng = random.Random(seed)
    id_list = list(ID_ENTITIES)
    ho_list = list(HELD_OUT_ENTITIES)
    n_id = int(n_families * 0.8)
    n_ho = n_families - n_id
    plans: List[ContrastFamily] = []
    used_id, used_ho = set(), set()

    def take(pool, used, i, prefix):
        while True:
            a, b, c = rng.sample(pool, 3)
            key = (a, b, c)
            if key not in used:
                used.add(key)
                return ContrastFamily(
                    family_id=f"{prefix}_{i:04d}",
                    a=a,
                    b=b,
                    c=c,
                    split="train",
                    entity_pool="in_distribution" if prefix == "xc_id" else "held_out_entities",
                    paraphrase_idx=rng.randrange(len(PARAPHRASES)),
                    query_idx=rng.randrange(len(QUERY_FORMS)),
                    layout_seed=rng.randrange(10**9),
                )

    for i in range(n_id):
        plans.append(take(id_list, used_id, i, "xc_id"))
    for i in range(n_ho):
        plans.append(take(ho_list, used_ho, i, "xc_ho"))

    order = list(range(len(plans)))
    rng.shuffle(order)
    n = len(order)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    for idx, pos in enumerate(order):
        split = "train" if idx < n_train else ("validation" if idx < n_train + n_val else "test")
        p = plans[pos]
        plans[pos] = ContrastFamily(
            family_id=p.family_id,
            a=p.a,
            b=p.b,
            c=p.c,
            split=split,
            entity_pool=p.entity_pool,
            paraphrase_idx=p.paraphrase_idx,
            query_idx=p.query_idx,
            layout_seed=p.layout_seed,
        )
    return plans


def build_family_records(plan: ContrastFamily, images_dir: Path) -> List[Dict[str, Any]]:
    records = []
    q = Query(plan.a, plan.c, DirectionalRelation.LEFT_OF)
    for vi, variant in enumerate(VARIANTS):
        premises, (v_left, v_right) = variant_premises(variant, plan.a, plan.b, plan.c)
        # Textual premise pair for prompt text
        text_p = [p for p in premises if p.modality == Modality.TEXTUAL][0]
        status, explanation, agreement = _solve(premises, q)
        expected = label_for_variant(variant)
        # Map status to expected label
        status_label = {
            "entailed": "Yes",
            "contradicted": "No",
            "underdetermined": "Undetermined",
            "inconsistent": None,
        }[status]
        if status_label != expected:
            raise AssertionError(
                f"{plan.family_id} {variant}: solver={status} expected_label={expected}"
            )
        if status == "inconsistent":
            raise AssertionError("inconsistent graphs excluded from 3-way contrast task")
        if not agreement:
            raise AssertionError(f"oracle disagreement for {plan.family_id} {variant}")

        prompt = build_prompt_with_text_pair(
            plan.a,
            plan.c,
            text_p.subject,
            text_p.object,
            plan.paraphrase_idx + vi,  # rotate paraphrases across variants but keep same format family
            plan.query_idx,
        )
        img_path = images_dir / f"{plan.family_id}__{variant}.png"
        rng = random.Random(plan.layout_seed + vi * 17)
        img_meta = render_canvas(
            v_left,
            v_right,
            img_path,
            rng,
            bg_jitter=6,
            gap_scale=1.0 + 0.15 * ((plan.layout_seed + vi) % 3),
        )

        # Single-modality insufficiency check
        only_visual = [p for p in premises if p.modality == Modality.VISUAL]
        only_text = [p for p in premises if p.modality == Modality.TEXTUAL]
        vis_status, _, _ = _solve(only_visual, q) if only_visual else ("underdetermined", "", True)
        txt_status, _, _ = _solve(only_text, q) if only_text else ("underdetermined", "", True)
        single_mod_sufficient = status in {"entailed", "contradicted"} and (
            vis_status == status or txt_status == status
        )

        rec = {
            "schema_version": SCHEMA_VERSION,
            "family_id": plan.family_id,
            "sample_id": f"{plan.family_id}::{variant}",
            "condition_id": CONDITION_ID,
            "variant_id": variant,  # evaluator-only; not used in prompt text as answer cue
            "split": plan.split,
            "entity_pool": plan.entity_pool,
            "entities": {"A": plan.a, "B": plan.b, "C": plan.c},
            "query": {"subject": plan.a, "object": plan.c, "relation": "left_of"},
            "premises": [
                {"subject": p.subject, "object": p.object, "modality": p.modality.value}
                for p in premises
            ],
            "ground_truth_label": expected,
            "symbolic_status": status,
            "solver_explanation": explanation,
            "oracle_agreement": agreement,
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "image_file": img_meta["image_file"],
            "image_sha256": img_meta["image_sha256"],
            "visual_entities_shown": sorted(img_meta["entities_shown"].keys()),
            "third_entity_hidden": plan.c not in img_meta["entities_shown"],
            "single_modality_sufficient": single_mod_sufficient,
            "requires_both_modalities": (not single_mod_sufficient) and status in {"entailed", "contradicted", "underdetermined"},
            "layout": {
                "bg_rgb": img_meta["bg_rgb"],
                "object_size": img_meta["object_size"],
                "gap_px": img_meta["gap_px"],
                "bboxes": img_meta["bboxes"],
            },
            "is_synthetic_demo": False,
        }
        if plan.c in img_meta["entities_shown"]:
            raise AssertionError("Entity C leaked onto visual canvas")
        records.append(rec)
    return records


def generate_contrast_dataset(
    *,
    n_families: int = 240,
    seed: int = DEFAULT_SEED,
    out_dir: Path,
    render_images: bool = True,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    plans = plan_families(n_families=n_families, seed=seed)
    all_records: List[Dict[str, Any]] = []
    for plan in plans:
        all_records.extend(build_family_records(plan, images_dir if render_images else images_dir))

    from collections import Counter

    label_counts = Counter(r["ground_truth_label"] for r in all_records)
    cond_counts = Counter(r["condition_id"] for r in all_records)
    # Condition-only baseline: constant condition => majority class
    maj = label_counts.most_common(1)[0][0]
    condition_baseline_ba = 1.0 / 3.0  # balanced accuracy of majority on balanced set

    # Paraphrase × label balance
    para_lab = Counter((r["prompt_sha256"][:8], r["ground_truth_label"]) for r in all_records)

    # Family split integrity
    fam_split = {}
    split_ok = True
    for r in all_records:
        if r["family_id"] in fam_split and fam_split[r["family_id"]] != r["split"]:
            split_ok = False
        fam_split[r["family_id"]] = r["split"]

    # Counterfactual consistency: each family has one of each label
    fam_labels = Counter()
    for r in all_records:
        fam_labels[(r["family_id"], r["ground_truth_label"])] += 1
    cf_ok = all(fam_labels[(f, lab)] == 1 for f in fam_split for lab in LABELS)

    jsonl_path = out_dir / "cross_modal_contrast_v2.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for r in all_records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "n_families": len(plans),
        "n_records": len(all_records),
        "condition_id": CONDITION_ID,
        "labels_per_family": list(LABELS),
        "label_counts": dict(label_counts),
        "condition_counts": dict(cond_counts),
        "split_family_counts": Counter(p.split for p in plans),
        "condition_only_majority_label": maj,
        "condition_only_balanced_accuracy": condition_baseline_ba,
        "condition_label_correlated": False,
        "family_split_ok": split_ok,
        "counterfactual_complete_families": cf_ok,
        "third_entity_leak_count": sum(1 for r in all_records if not r["third_entity_hidden"]),
        "single_modality_sufficient_count": sum(1 for r in all_records if r["single_modality_sufficient"]),
        "notes": [
            "New version; v1 preserved unchanged.",
            "Single condition_id for all labels.",
            "Condition-name baseline cannot exceed chance under class balance.",
            "C never rendered on canvas for these premises.",
            "Inconsistent graphs excluded from three-way task.",
        ],
        "jsonl": _rel(jsonl_path),
    }
    (out_dir / "cross_modal_contrast_meta_v2.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {**meta, "records": all_records}


def load_contrast_records(jsonl_path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
