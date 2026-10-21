"""Cross-modal dependency benchmark generator (Phase 1B).

Each family has three distinct entities A, B, C:
- Visual premise involves only two entities (never reveals C in chain conditions).
- Textual premise involves the remaining bridge/query entities.
- Labels are independently determined by SymbolicSolver.

Conditions (8):
1. both_premises
2. visual_removed
3. text_removed
4. visual_reversed
5. text_reversed
6. both_modified
7. underdetermined_fork
8. contradictory_evidence

This module prepares metadata + optional rendered images; it does not run the VLM.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.config.paths import REPO_ROOT
from src.symbolic.relations import DirectionalRelation, Modality, Premise, Query
from src.symbolic.solver import SymbolicSolver

from PIL import Image, ImageDraw, ImageDraw as _ImageDraw

# Import palette/draw helpers without requiring scripts to be a package.
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "controlled_benchmark_helpers",
    REPO_ROOT / "scripts" / "generate_controlled_benchmark.py",
)
_helpers = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_helpers)
BG_COLOR = _helpers.BG_COLOR
CANVAS_SIZE = _helpers.CANVAS_SIZE
ENTITY_PALETTE = _helpers.ENTITY_PALETTE
draw_shape = _helpers.draw_shape

SCHEMA_VERSION = "cross_modal_dependency_v1"
DEFAULT_SEED = 20261021
CONDITIONS = (
    "both_premises",
    "visual_removed",
    "text_removed",
    "visual_reversed",
    "text_reversed",
    "both_modified",
    "underdetermined_fork",
    "contradictory_evidence",
)

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


def _solver_status(premises: List[Premise], query: Query) -> Tuple[str, str]:
    result = SymbolicSolver.solve(premises, query)
    return result.epistemic_status.value, result.explanation


def _label_from_status(status: str) -> str:
    return {
        "entailed": "Yes",
        "contradicted": "No",
        "underdetermined": "Undetermined",
        "inconsistent": "Undetermined",  # inconsistent evidence is not a forced definite answer
    }[status]


def _P(subj: str, obj: str, modality: Modality) -> Premise:
    return Premise(subj, obj, DirectionalRelation.LEFT_OF, modality)


def _Q(subj: str, obj: str) -> Query:
    return Query(subj, obj, DirectionalRelation.LEFT_OF)


def condition_premises(condition: str, a: str, b: str, c: str) -> Tuple[List[Premise], str]:
    """Return premises for a condition and a short evidence-type label."""
    if condition == "both_premises":
        return [_P(a, b, Modality.VISUAL), _P(b, c, Modality.TEXTUAL)], "cross_modal_chain"
    if condition == "visual_removed":
        return [_P(b, c, Modality.TEXTUAL)], "text_only_insufficient"
    if condition == "text_removed":
        return [_P(a, b, Modality.VISUAL)], "visual_only_insufficient"
    if condition == "visual_reversed":
        return [_P(b, a, Modality.VISUAL), _P(b, c, Modality.TEXTUAL)], "cross_modal_modified_visual"
    if condition == "text_reversed":
        return [_P(a, b, Modality.VISUAL), _P(c, b, Modality.TEXTUAL)], "cross_modal_modified_text"
    if condition == "both_modified":
        return [_P(b, a, Modality.VISUAL), _P(c, b, Modality.TEXTUAL)], "cross_modal_both_modified"
    if condition == "underdetermined_fork":
        return [_P(a, b, Modality.VISUAL), _P(c, b, Modality.TEXTUAL)], "fork_insufficient_evidence"
    if condition == "contradictory_evidence":
        # Visual: C left of A (on A,C canvas). Text: chain A<B<C entails R(A,C).
        return [
            _P(c, a, Modality.VISUAL),
            _P(a, b, Modality.TEXTUAL),
            _P(b, c, Modality.TEXTUAL),
        ], "contradictory_multimodal"
    raise ValueError(condition)


def visual_pair_for_condition(condition: str, a: str, b: str, c: str) -> Tuple[str, str]:
    """Which two entities appear on the canvas and their left/right order (left, right)."""
    if condition == "visual_removed":
        return a, b  # blank/neutral still may render nothing; caller decides
    if condition == "contradictory_evidence":
        return c, a  # C left of A
    if condition == "visual_reversed" or condition == "both_modified":
        return b, a  # reversed visual vs base A left of B
    # base visual premise is R(A,B): A left of B
    return a, b


def _rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def render_condition_image(
    *,
    condition: str,
    a: str,
    b: str,
    c: str,
    output_path: Path,
    rng: random.Random,
) -> Dict[str, Any]:
    if condition == "visual_removed":
        img = Image.new("RGB", CANVAS_SIZE, color=BG_COLOR)
        meta = {"dimensions": list(CANVAS_SIZE), "entities": {}, "note": "no visual entities"}
    else:
        left, right = visual_pair_for_condition(condition, a, b, c)
        img = Image.new("RGB", CANVAS_SIZE, color=BG_COLOR)
        draw = ImageDraw.Draw(img)
        lx1 = rng.randint(40, 120)
        ly1 = rng.randint(140, 180)
        lx2, ly2 = lx1 + 64, ly1 + 64
        rx1 = rng.randint(220, 280)
        ry1 = rng.randint(140, 180)
        rx2, ry2 = rx1 + 64, ry1 + 64
        assert lx2 < rx1
        draw_shape(draw, ENTITY_PALETTE[left]["shape"], ENTITY_PALETTE[left]["color"], (lx1, ly1, lx2, ly2))
        draw_shape(draw, ENTITY_PALETTE[right]["shape"], ENTITY_PALETTE[right]["color"], (rx1, ry1, rx2, ry2))
        meta = {
            "dimensions": list(CANVAS_SIZE),
            "entities": {
                left: {"bbox": [lx1, ly1, lx2, ly2], "role": "left"},
                right: {"bbox": [rx1, ry1, rx2, ry2], "role": "right"},
            },
            "hidden_third_entity": c if c not in (left, right) else None,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")
    h = hashlib.sha256(output_path.read_bytes()).hexdigest()
    meta["image_sha256"] = h
    meta["image_file"] = _rel_or_abs(output_path)
    return meta


def build_prompt_for_condition(
    *,
    condition: str,
    a: str,
    b: str,
    c: str,
    linguistic_template: str = "standard",
) -> str:
    def fmt(e: str) -> str:
        return e.replace("_", " ")

    query = f"is {fmt(a)} to the left of {fmt(c)}?"
    instruction = (
        "Carefully evaluate whether the question is entailed (Yes), contradicted (No), "
        "or underdetermined from the evidence (Undetermined).\n"
        "Answer directly with: Yes, No, or Undetermined."
    )
    if condition == "visual_removed":
        premise = f"The {fmt(b)} is to the left of the {fmt(c)}."
        body = f"{premise}\n\nQuestion: Based on the statements, {query}\n{instruction}"
    elif condition == "text_removed":
        premise = "No additional statement is provided."
        body = f"{premise}\n\nQuestion: Based on the image, {query}\n{instruction}"
    elif condition == "contradictory_evidence":
        premise = (
            f"The {fmt(a)} is to the left of the {fmt(b)}.\n"
            f"The {fmt(b)} is to the left of the {fmt(c)}."
        )
        body = f"{premise}\n\nQuestion: Based on the image and statements, {query}\n{instruction}"
    elif condition == "visual_reversed":
        premise = f"The {fmt(b)} is to the left of the {fmt(c)}."
        body = f"{premise}\n\nQuestion: Based on the image and statement, {query}\n{instruction}"
    elif condition == "text_reversed":
        premise = "No additional statement is provided."  # fork uses two textual? actually visual A<B + text C<B
        # For text_reversed we still have visual A left of B and text C left of B
        premise = f"The {fmt(c)} is to the left of the {fmt(b)}."
        body = f"{premise}\n\nQuestion: Based on the image and statement, {query}\n{instruction}"
    elif condition == "both_modified":
        premise = f"The {fmt(c)} is to the left of the {fmt(b)}."
        body = f"{premise}\n\nQuestion: Based on the image and statement, {query}\n{instruction}"
    elif condition == "underdetermined_fork":
        premise = f"The {fmt(c)} is to the left of the {fmt(b)}."
        body = f"{premise}\n\nQuestion: Based on the image and statement, {query}\n{instruction}"
    else:  # both_premises
        premise = f"The {fmt(b)} is to the left of the {fmt(c)}."
        body = f"{premise}\n\nQuestion: Based on the image and statement, {query}\n{instruction}"

    if linguistic_template == "variant_b":
        body = body.replace("Carefully evaluate", "Think carefully and evaluate")
    return body


@dataclass
class FamilyPlan:
    family_id: str
    a: str
    b: str
    c: str
    split: str
    entity_pool: str
    linguistic_template: str


def plan_families(
    *,
    n_families: int = 240,
    seed: int = DEFAULT_SEED,
    id_entities: Sequence[str] = ID_ENTITIES,
    held_out_entities: Sequence[str] = HELD_OUT_ENTITIES,
) -> List[FamilyPlan]:
    rng = random.Random(seed)
    id_list = list(id_entities)
    ho_list = list(held_out_entities)
    plans: List[FamilyPlan] = []

    # 192 in-distribution + 48 held-out entity triples
    n_id = int(n_families * 0.8)
    n_ho = n_families - n_id
    used_id = set()
    used_ho = set()
    for i in range(n_id):
        while True:
            a, b, c = rng.sample(id_list, 3)
            key = (a, b, c)
            if key not in used_id:
                used_id.add(key)
                break
        plans.append(
            FamilyPlan(
                family_id=f"xm_id_{i:04d}",
                a=a,
                b=b,
                c=c,
                split="train",  # assigned later by group
                entity_pool="in_distribution",
                linguistic_template="standard" if i % 2 == 0 else "variant_b",
            )
        )
    for i in range(n_ho):
        while True:
            a, b, c = rng.sample(ho_list, 3)
            key = (a, b, c)
            if key not in used_ho:
                used_ho.add(key)
                break
        plans.append(
            FamilyPlan(
                family_id=f"xm_ho_{i:04d}",
                a=a,
                b=b,
                c=c,
                split="train",
                entity_pool="held_out_entities",
                linguistic_template="standard" if i % 2 == 0 else "variant_b",
            )
        )

    # Group-disjoint split over families: 70/15/15
    order = list(range(len(plans)))
    rng.shuffle(order)
    n = len(order)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    for idx, pos in enumerate(order):
        if idx < n_train:
            split = "train"
        elif idx < n_train + n_val:
            split = "validation"
        else:
            split = "test"
        p = plans[pos]
        plans[pos] = FamilyPlan(
            family_id=p.family_id,
            a=p.a,
            b=p.b,
            c=p.c,
            split=split,
            entity_pool=p.entity_pool,
            linguistic_template=p.linguistic_template,
        )
    return plans


def build_family_records(
    plan: FamilyPlan,
    *,
    images_dir: Path,
    rng: random.Random,
    render_images: bool = True,
) -> List[Dict[str, Any]]:
    records = []
    q = _Q(plan.a, plan.c)
    for cond in CONDITIONS:
        premises, evidence_type = condition_premises(cond, plan.a, plan.b, plan.c)
        status, explanation = _solver_status(premises, q)
        label = _label_from_status(status)
        prompt = build_prompt_for_condition(
            condition=cond,
            a=plan.a,
            b=plan.b,
            c=plan.c,
            linguistic_template=plan.linguistic_template,
        )
        # Visual leakage audit: third entity must not appear in image for non-contradictory chain-like conditions
        visual_entities = set()
        if cond != "visual_removed":
            left, right = visual_pair_for_condition(cond, plan.a, plan.b, plan.c)
            visual_entities = {left, right}
        third = plan.c if plan.c not in visual_entities else None
        # For conditions where query subject/object should not both be determined by single modality alone
        single_modality_determines = status in {"entailed", "contradicted"} and len(premises) == 1

        rec = {
            "schema_version": SCHEMA_VERSION,
            "family_id": plan.family_id,
            "sample_id": f"{plan.family_id}::{cond}",
            "condition": cond,
            "evidence_type": evidence_type,
            "split": plan.split,
            "entity_pool": plan.entity_pool,
            "linguistic_template": plan.linguistic_template,
            "entities": {"A": plan.a, "B": plan.b, "C": plan.c},
            "query": {"subject": plan.a, "object": plan.c, "relation": "left_of"},
            "premises": [
                {"subject": p.subject, "object": p.object, "modality": p.modality.value}
                for p in premises
            ],
            "ground_truth_label": label,
            "symbolic_status": status,
            "solver_explanation": explanation,
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "visual_entities_shown": sorted(visual_entities),
            "third_entity_hidden_in_image": third,
            "single_modality_sufficient": single_modality_determines,
            "requires_both_modalities": (
                status == "entailed" and cond == "both_premises"
            ),
            "is_synthetic_demo": False,
        }

        if render_images:
            img_path = images_dir / f"{plan.family_id}__{cond}.png"
            meta = render_condition_image(
                condition=cond, a=plan.a, b=plan.b, c=plan.c, output_path=img_path, rng=rng
            )
            rec["image_file"] = meta["image_file"]
            rec["image_sha256"] = meta["image_sha256"]
            rec["image_metadata"] = meta
        else:
            rec["image_file"] = None
            rec["image_sha256"] = None
            rec["image_metadata"] = None

        # Modality ablation integrity checks
        if cond == "visual_removed":
            rec["modality_ablation_ok"] = all(p.modality != Modality.VISUAL for p in premises)
        elif cond == "text_removed":
            rec["modality_ablation_ok"] = all(p.modality != Modality.TEXTUAL for p in premises)
        else:
            rec["modality_ablation_ok"] = True

        records.append(rec)
    return records


def generate_dataset(
    *,
    n_families: int = 240,
    seed: int = DEFAULT_SEED,
    out_dir: Path,
    render_images: bool = True,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    plans = plan_families(n_families=n_families, seed=seed)
    rng = random.Random(seed + 1)
    all_records: List[Dict[str, Any]] = []
    for plan in plans:
        all_records.extend(
            build_family_records(plan, images_dir=images_dir, rng=rng, render_images=render_images)
        )

    # Independent re-verification
    mismatches = []
    for rec in all_records:
        premises = [
            Premise(
                p["subject"],
                p["object"],
                DirectionalRelation.LEFT_OF,
                Modality.VISUAL if p["modality"] == "visual" else Modality.TEXTUAL,
            )
            for p in rec["premises"]
        ]
        status, _ = _solver_status(premises, _Q(rec["entities"]["A"], rec["entities"]["C"]))
        if status != rec["symbolic_status"]:
            mismatches.append(rec["sample_id"])

    # Class distribution
    from collections import Counter

    label_counts = Counter(r["ground_truth_label"] for r in all_records)
    cond_counts = Counter(r["condition"] for r in all_records)
    split_family_counts = Counter(p.split for p in plans)

    # Superficial feature audit: answer vs first entity color name token
    color_leak = Counter()
    for r in all_records:
        first_token = r["entities"]["A"].split("_")[0]
        color_leak[(first_token, r["ground_truth_label"])] += 1

    # Ensure no family crosses splits
    family_split = {}
    split_ok = True
    for r in all_records:
        fid = r["family_id"]
        if fid in family_split and family_split[fid] != r["split"]:
            split_ok = False
        family_split[fid] = r["split"]

    payload = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "n_families": len(plans),
        "n_records": len(all_records),
        "n_conditions_per_family": len(CONDITIONS),
        "label_counts": dict(label_counts),
        "condition_counts": dict(cond_counts),
        "split_family_counts": dict(split_family_counts),
        "solver_mismatches": mismatches,
        "group_disjoint_splits_ok": split_ok,
        "notes": [
            "Dataset preparation only; not GPU inference authorization.",
            "Object-vocabulary disjointness does not alone eliminate shortcut leakage.",
            "Underdetermined != False; contradictory evidence labeled separately via solver.",
        ],
        "records": all_records,
    }
    # Write JSONL + metadata
    jsonl_path = out_dir / "cross_modal_dependency_v1.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for rec in all_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    meta_path = out_dir / "cross_modal_dependency_meta_v1.json"
    meta = {k: v for k, v in payload.items() if k != "records"}
    meta["jsonl"] = _rel_or_abs(jsonl_path)
    meta["color_label_counts"] = {f"{k[0]}|{k[1]}": v for k, v in color_leak.items()}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return payload


def load_records(jsonl_path: Path) -> List[Dict[str, Any]]:
    rows = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows
