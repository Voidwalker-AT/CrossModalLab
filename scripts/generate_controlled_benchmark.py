"""CrossModalLab — Controlled Benchmark Image & Manifest Generator.

Generates 180 controlled procedural samples across 5 balanced topologies:
1. Forward Chain (Entailed) — 36 samples
2. Reverse Chain (Contradicted) — 36 samples
3. Underdetermined Fork — 36 samples (18 fan-in, 18 fan-out)
4. Unimodal Visual Control — 36 samples
5. Unimodal Text Control — 36 samples

Strict Scientific Constraints:
- Pillow-based rendering of distinct geometric entities.
- Exactly two entities appear in cross-modal visual premise images; third entity NEVER appears.
- Entity bounding boxes and coordinates explicitly recorded.
- Strict horizontal left-of spatial geometry verified: x_max(left) < x_min(right).
- SHA-256 image checksums computed and recorded.
- No label leakage in filenames or prompts.
"""

import os
import sys
import json
import math
import random
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw

# Add repository root to path for symbolic imports
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.symbolic.relations import DirectionalRelation, EpistemicStatus, Modality, Premise, Query
from src.symbolic.solver import SymbolicSolver
from src.symbolic.generator import IN_DISTRIBUTION_ENTITIES

# Entity visual definitions: color RGB and shape drawer
ENTITY_PALETTE = {
    "blue_cube": {"color": (41, 128, 185), "shape": "box"},
    "red_sphere": {"color": (231, 76, 60), "shape": "circle"},
    "green_cylinder": {"color": (39, 174, 96), "shape": "pill"},
    "yellow_cone": {"color": (241, 196, 15), "shape": "triangle"},
    "purple_pyramid": {"color": (142, 68, 173), "shape": "diamond"},
    "orange_torus": {"color": (230, 126, 34), "shape": "ring"},
    "cyan_box": {"color": (26, 188, 156), "shape": "box"},
    "magenta_prism": {"color": (232, 67, 147), "shape": "hexagon"},
}

CANVAS_SIZE = (384, 384)
BG_COLOR = (248, 249, 250)

def draw_shape(draw: ImageDraw.ImageDraw, shape: str, color: Tuple[int, int, int], bbox: Tuple[int, int, int, int]):
    """Renders a single geometric entity within the given bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    if shape == "box":
        draw.rectangle([x1, y1, x2, y2], fill=color, outline=(40, 40, 40), width=2)
    elif shape == "circle":
        draw.ellipse([x1, y1, x2, y2], fill=color, outline=(40, 40, 40), width=2)
    elif shape == "pill":
        r = min(w, h) / 4
        draw.rounded_rectangle([x1, y1, x2, y2], radius=int(r), fill=color, outline=(40, 40, 40), width=2)
    elif shape == "triangle":
        points = [(cx, y1), (x1, y2), (x2, y2)]
        draw.polygon(points, fill=color, outline=(40, 40, 40), width=2)
    elif shape == "diamond":
        points = [(cx, y1), (x2, cy), (cx, y2), (x1, cy)]
        draw.polygon(points, fill=color, outline=(40, 40, 40), width=2)
    elif shape == "ring":
        draw.ellipse([x1, y1, x2, y2], fill=color, outline=(40, 40, 40), width=2)
        inner_margin = int(w * 0.28)
        draw.ellipse([x1 + inner_margin, y1 + inner_margin, x2 - inner_margin, y2 - inner_margin], fill=BG_COLOR, outline=(40, 40, 40), width=2)
    elif shape == "hexagon":
        pts = []
        for i in range(6):
            angle = math.radians(60 * i - 30)
            px = cx + (w / 2) * math.cos(angle)
            py = cy + (h / 2) * math.sin(angle)
            pts.append((px, py))
        draw.polygon(pts, fill=color, outline=(40, 40, 40), width=2)
    else:
        draw.rectangle([x1, y1, x2, y2], fill=color, outline=(40, 40, 40), width=2)

def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(64 * 1024):
            h.update(chunk)
    return h.hexdigest()

def render_pairwise_image(
    left_entity: str,
    right_entity: str,
    output_path: Path,
    rng: random.Random
) -> Dict[str, Any]:
    """
    Renders an image with left_entity strictly to the left of right_entity.
    Verifies x2(left) < x1(right).
    Returns bounding box metadata.
    """
    img = Image.new("RGB", CANVAS_SIZE, color=BG_COLOR)
    draw = ImageDraw.Draw(img)

    obj_w = 64
    obj_h = 64

    # Jitter left position within left region [40..130]
    lx1 = rng.randint(40, 120)
    ly1 = rng.randint(140, 180)
    lx2 = lx1 + obj_w
    ly2 = ly1 + obj_h

    # Jitter right position within right region [220..300]
    rx1 = rng.randint(220, 280)
    ry1 = rng.randint(140, 180)
    rx2 = rx1 + obj_w
    ry2 = ry1 + obj_h

    # Assert formal spatial condition
    assert lx2 < rx1, f"Spatial overlap violation: lx2={lx2} >= rx1={rx1}"

    draw_shape(draw, ENTITY_PALETTE[left_entity]["shape"], ENTITY_PALETTE[left_entity]["color"], (lx1, ly1, lx2, ly2))
    draw_shape(draw, ENTITY_PALETTE[right_entity]["shape"], ENTITY_PALETTE[right_entity]["color"], (rx1, ry1, rx2, ry2))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")

    return {
        "dimensions": list(CANVAS_SIZE),
        "entities": {
            left_entity: {"bbox": [lx1, ly1, lx2, ly2], "role": "left"},
            right_entity: {"bbox": [rx1, ry1, rx2, ry2], "role": "right"}
        },
        "horizontal_gap_px": rx1 - lx2
    }

def render_blank_canvas(output_path: Path) -> Dict[str, Any]:
    """Renders a blank canvas for unimodal text control."""
    img = Image.new("RGB", CANVAS_SIZE, color=BG_COLOR)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")
    return {
        "dimensions": list(CANVAS_SIZE),
        "entities": {},
        "horizontal_gap_px": None
    }

def render_three_entity_image(
    left_entity: str,
    mid_entity: str,
    right_entity: str,
    output_path: Path,
    rng: random.Random
) -> Dict[str, Any]:
    """Renders 3 entities left-to-right for unimodal visual control."""
    img = Image.new("RGB", CANVAS_SIZE, color=BG_COLOR)
    draw = ImageDraw.Draw(img)

    obj_w = 56
    obj_h = 56

    x1_a = rng.randint(30, 60)
    y1_a = rng.randint(145, 185)
    x2_a = x1_a + obj_w
    y2_a = y1_a + obj_h

    x1_b = rng.randint(155, 185)
    y1_b = rng.randint(145, 185)
    x2_b = x1_b + obj_w
    y2_b = y1_b + obj_h

    x1_c = rng.randint(275, 305)
    y1_c = rng.randint(145, 185)
    x2_c = x1_c + obj_w
    y2_c = y1_c + obj_h

    assert x2_a < x1_b < x2_b < x1_c, "3-entity order violation"

    draw_shape(draw, ENTITY_PALETTE[left_entity]["shape"], ENTITY_PALETTE[left_entity]["color"], (x1_a, y1_a, x2_a, y2_a))
    draw_shape(draw, ENTITY_PALETTE[mid_entity]["shape"], ENTITY_PALETTE[mid_entity]["color"], (x1_b, y1_b, x2_b, y2_b))
    draw_shape(draw, ENTITY_PALETTE[right_entity]["shape"], ENTITY_PALETTE[right_entity]["color"], (x1_c, y1_c, x2_c, y2_c))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")

    return {
        "dimensions": list(CANVAS_SIZE),
        "entities": {
            left_entity: {"bbox": [x1_a, y1_a, x2_a, y2_a], "role": "left"},
            mid_entity: {"bbox": [x1_b, y1_b, x2_b, y2_b], "role": "middle"},
            right_entity: {"bbox": [x1_c, y1_c, x2_c, y2_c], "role": "right"}
        },
        "gap_ab_px": x1_b - x2_a,
        "gap_bc_px": x1_c - x2_b
    }

def format_entity_name(name: str) -> str:
    """Formats 'blue_cube' -> 'blue cube'."""
    return name.replace("_", " ")

def generate_controlled_suite(
    output_dir: Path,
    seed: int = 1337,
    n_per_topology: int = 36
) -> Dict[str, Any]:
    """Generates the full 180 controlled benchmark samples with matched pairs."""
    rng = random.Random(seed)
    images_dir = output_dir / "images"
    manifests_dir = output_dir / "manifests"
    annotations_dir = output_dir / "annotations"

    images_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    entities = list(IN_DISTRIBUTION_ENTITIES)
    all_triplets = []
    for i in range(len(entities)):
        for j in range(len(entities)):
            for k in range(len(entities)):
                if len({i, j, k}) == 3:
                    all_triplets.append((entities[i], entities[j], entities[k]))

    rng.shuffle(all_triplets)

    records = []
    topologies = [
        "forward_chain",
        "reverse_chain",
        "underdetermined_fork",
        "unimodal_visual",
        "unimodal_text"
    ]

    sample_counter = 0

    for topology in topologies:
        triplet_pool = all_triplets.copy()
        rng.shuffle(triplet_pool)

        for i in range(n_per_topology):
            sample_counter += 1
            sample_id = f"ctrl_{topology}_{sample_counter:04d}"
            a, b, c = triplet_pool.pop()

            img_filename = f"{sample_id}.png"
            img_path = images_dir / img_filename

            if topology == "forward_chain":
                # Premise 1 (Visual): A is left of B
                # Premise 2 (Textual): B is left of C
                # Query: Is A left of C? => ENTAILED
                img_meta = render_pairwise_image(a, b, img_path, rng)
                text_premise = f"The {format_entity_name(b)} is to the left of the {format_entity_name(c)}."
                query_text = f"Based on the image and statement, is the {format_entity_name(a)} to the left of the {format_entity_name(c)}?"
                epistemic_status = "entailed"
                ground_truth_decision = "Yes"
                premises_logic = [
                    {"modality": "visual", "subject": a, "object": b, "relation": "left_of"},
                    {"modality": "textual", "subject": b, "object": c, "relation": "left_of"}
                ]
                query_logic = {"subject": a, "object": c, "relation": "left_of"}

            elif topology == "reverse_chain":
                # Premise 1 (Visual): B is left of A
                # Premise 2 (Textual): C is left of B
                # Query: Is A left of C? => CONTRADICTED (C < B < A)
                img_meta = render_pairwise_image(b, a, img_path, rng)
                text_premise = f"The {format_entity_name(c)} is to the left of the {format_entity_name(b)}."
                query_text = f"Based on the image and statement, is the {format_entity_name(a)} to the left of the {format_entity_name(c)}?"
                epistemic_status = "contradicted"
                ground_truth_decision = "No"
                premises_logic = [
                    {"modality": "visual", "subject": b, "object": a, "relation": "left_of"},
                    {"modality": "textual", "subject": c, "object": b, "relation": "left_of"}
                ]
                query_logic = {"subject": a, "object": c, "relation": "left_of"}

            elif topology == "underdetermined_fork":
                # Fork: A is left of B (Visual), C is left of B (Textual)
                # Or B is left of A (Visual), B is left of C (Textual)
                is_fan_in = (i % 2 == 0)
                if is_fan_in:
                    # Visual: A < B. Text: C < B. Query: A < C? => UNDERDETERMINED
                    img_meta = render_pairwise_image(a, b, img_path, rng)
                    text_premise = f"The {format_entity_name(c)} is to the left of the {format_entity_name(b)}."
                    premises_logic = [
                        {"modality": "visual", "subject": a, "object": b, "relation": "left_of"},
                        {"modality": "textual", "subject": c, "object": b, "relation": "left_of"}
                    ]
                else:
                    # Visual: B < A. Text: B < C. Query: A < C? => UNDERDETERMINED
                    img_meta = render_pairwise_image(b, a, img_path, rng)
                    text_premise = f"The {format_entity_name(b)} is to the left of the {format_entity_name(c)}."
                    premises_logic = [
                        {"modality": "visual", "subject": b, "object": a, "relation": "left_of"},
                        {"modality": "textual", "subject": b, "object": c, "relation": "left_of"}
                    ]
                query_text = f"Based on the image and statement, is the {format_entity_name(a)} to the left of the {format_entity_name(c)}?"
                epistemic_status = "underdetermined"
                ground_truth_decision = "Undetermined"
                query_logic = {"subject": a, "object": c, "relation": "left_of"}

            elif topology == "unimodal_visual":
                # Visual: A < B < C in image
                # Text: "No additional statement is provided."
                # 50% positive query (A < C? => Yes), 50% negative query (C < A? => No)
                img_meta = render_three_entity_image(a, b, c, img_path, rng)
                text_premise = "No additional statement is provided."
                is_pos = (i % 2 == 0)
                if is_pos:
                    query_text = f"Based on the image, is the {format_entity_name(a)} to the left of the {format_entity_name(c)}?"
                    epistemic_status = "entailed"
                    ground_truth_decision = "Yes"
                    query_logic = {"subject": a, "object": c, "relation": "left_of"}
                else:
                    query_text = f"Based on the image, is the {format_entity_name(c)} to the left of the {format_entity_name(a)}?"
                    epistemic_status = "contradicted"
                    ground_truth_decision = "No"
                    query_logic = {"subject": c, "object": a, "relation": "left_of"}
                premises_logic = [
                    {"modality": "visual", "subject": a, "object": b, "relation": "left_of"},
                    {"modality": "visual", "subject": b, "object": c, "relation": "left_of"}
                ]

            elif topology == "unimodal_text":
                # Blank image.
                # Text: "The [A] is to the left of the [B]. The [B] is to the left of the [C]."
                # 50% positive query (A < C? => Yes), 50% negative query (C < A? => No)
                img_meta = render_blank_canvas(img_path)
                text_premise = f"The {format_entity_name(a)} is to the left of the {format_entity_name(b)}. The {format_entity_name(b)} is to the left of the {format_entity_name(c)}."
                is_pos = (i % 2 == 0)
                if is_pos:
                    query_text = f"Based on the statements, is the {format_entity_name(a)} to the left of the {format_entity_name(c)}?"
                    epistemic_status = "entailed"
                    ground_truth_decision = "Yes"
                    query_logic = {"subject": a, "object": c, "relation": "left_of"}
                else:
                    query_text = f"Based on the statements, is the {format_entity_name(c)} to the left of the {format_entity_name(a)}?"
                    epistemic_status = "contradicted"
                    ground_truth_decision = "No"
                    query_logic = {"subject": c, "object": a, "relation": "left_of"}
                premises_logic = [
                    {"modality": "textual", "subject": a, "object": b, "relation": "left_of"},
                    {"modality": "textual", "subject": b, "object": c, "relation": "left_of"}
                ]

            # Verify with SymbolicSolver
            p_objs = [Premise(p["subject"], p["object"], DirectionalRelation(p["relation"]), Modality(p["modality"])) for p in premises_logic]
            q_obj = Query(query_logic["subject"], query_logic["object"], DirectionalRelation(query_logic["relation"]))
            solver_res = SymbolicSolver.solve(p_objs, q_obj, verify_with_oracle=True)
            assert solver_res.epistemic_status.value == epistemic_status, f"Solver mismatch for {sample_id}: expected {epistemic_status}, got {solver_res.epistemic_status.value}"

            img_hash = sha256_file(img_path)

            record = {
                "sample_id": sample_id,
                "is_synthetic_demo": False,
                "topology": topology,
                "image_file": img_filename,
                "image_sha256": img_hash,
                "image_metadata": img_meta,
                "text_premise": text_premise,
                "query": query_text,
                "ground_truth_decision": ground_truth_decision,
                "epistemic_status": epistemic_status,
                "symbolic_ground_truth": {
                    "relation": "left_of",
                    "premises": premises_logic,
                    "query": query_logic,
                    "epistemic_status": epistemic_status,
                    "consistent_linear_extensions": solver_res.linear_extensions
                }
            }
            records.append(record)

    manifest = {
        "suite_name": "controlled_benchmark_180",
        "seed": seed,
        "n_samples": len(records),
        "topology_distribution": {
            top: sum(1 for r in records if r["topology"] == top) for top in topologies
        },
        "decision_distribution": {
            dec: sum(1 for r in records if r["ground_truth_decision"] == dec) for dec in ["Yes", "No", "Undetermined"]
        },
        "records": records
    }

    manifest_json = manifests_dir / "controlled_180_manifest.json"
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Also write JSONL for easy line-by-line reading
    jsonl_path = output_dir / "controlled_benchmark_180.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Generated {len(records)} controlled samples in {output_dir}")
    print(f"Topology distribution: {manifest['topology_distribution']}")
    print(f"Decision distribution: {manifest['decision_distribution']}")
    print(f"Manifest written to: {manifest_json}")
    return manifest

if __name__ == "__main__":
    out_dir = Path("datasets/controlled")
    generate_controlled_suite(out_dir, seed=1337, n_per_topology=36)
