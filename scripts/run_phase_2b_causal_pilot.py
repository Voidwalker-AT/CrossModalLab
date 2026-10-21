#!/usr/bin/env python3
"""Phase 2B — Bounded exploratory causal activation-patching pilot (3 families).

Engineering no-op gate first; then source→target / reverse / controls at L18 last token.
No final holdout. No VLM training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from PIL import Image

from src.causal.patch_runner import (
    capture_last_residual,
    delta_margin,
    residual_patch_context,
    run_with_residual_patch,
    score_margins_from_logprobs,
)


def _cand_key(label: str) -> str:
    return f" {label}" if not label.startswith(" ") else label


def score_pair_margins(logprobs: Dict[str, float], source_label: str, target_label: str) -> Dict[str, float]:
    return score_margins_from_logprobs(logprobs, _cand_key(source_label), _cand_key(target_label))
from src.config.experiment import get_experiment_config
from src.config.paths import apply_default_cache_environment, get_artifacts_dir
from src.evaluation.behavioral_harness import append_jsonl, load_jsonl
from src.evaluation.behavioral_metrics import parse_model_answer
from src.instrumentation.capture_integrity import atomic_write_json
from src.models.model_loader import load_pinned_smolvlm2

apply_default_cache_environment()

CANDIDATES = [" Yes", " No", " Undetermined"]
LAYER = 18
NOOP_LOGIT_TOL = 1e-4
PATCH_SEED = 20261102


def load_pairs(family_ids: List[str]) -> List[Dict[str, Any]]:
    recs = [
        json.loads(line)
        for line in (REPO_ROOT / "datasets/cross_modal_contrast_v2/cross_modal_contrast_v2.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    by = {(r["family_id"], r["variant_id"]): r for r in recs}
    pairs = []
    for fam in family_ids:
        e = by[(fam, "entailed")]
        u = by[(fam, "underdetermined")]
        c = by[(fam, "contradicted")]
        # Entailed vs underdetermined: same visual *relation* (A left of B) but
        # contrast v2 renders independent canvases (layout jitter). Treat as
        # same_visual_structure_changed_text; do not require identical image_sha256.
        same_vis = set(e.get("visual_entities_shown") or []) == set(u.get("visual_entities_shown") or [])
        assert same_vis
        pairs.append(
            {
                "family_id": fam,
                "pair_id": f"{fam}:entailed_vs_underdetermined",
                "pair_type": "same_visual_structure_changed_text",
                "source": e,
                "target": u,
                "source_label": e["ground_truth_label"],  # Yes
                "target_label": u["ground_truth_label"],  # Undetermined
            }
        )
        # Secondary: same text, changed image (contradicted source → underdetermined target)
        # text premises equal?
        tp = lambda r: [p for p in r["premises"] if p["modality"] == "textual"]
        if tp(c) == tp(u) and c["image_sha256"] != u["image_sha256"]:
            pairs.append(
                {
                    "family_id": fam,
                    "pair_id": f"{fam}:contradicted_vs_underdetermined",
                    "pair_type": "same_text_changed_image",
                    "source": c,
                    "target": u,
                    "source_label": c["ground_truth_label"],  # No
                    "target_label": u["ground_truth_label"],  # Undetermined
                }
            )
    return pairs


def build_inputs(processor, rec: Dict[str, Any], device: str = "cuda:0") -> Dict[str, Any]:
    img = Image.open(REPO_ROOT / rec["image_file"]).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": rec["prompt"]},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    return inputs.to(device)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--max-patched-forwards", type=int, default=24)
    args = p.parse_args()

    cfg = get_experiment_config()
    exec_spec = json.loads(
        (REPO_ROOT / "configs/phase_2b/causal_pilot_exec_spec_v1.json").read_text(encoding="utf-8")
    )
    families = exec_spec["families"]
    assert len(families) == 3
    pairs = load_pairs(families)
    print(f"Pairs loaded: {len(pairs)} for families {families}")

    out_dir = args.out_dir or (get_artifacts_dir() / "causal" / "phase_2b")
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "intervention_records_v1.jsonl"
    existing = load_jsonl(jsonl_path) if jsonl_path.is_file() else []
    done_ids = {r.get("intervention_id") for r in existing if r.get("status") == "ok"}

    processor, model, model_meta = load_pinned_smolvlm2(require_gpu=True)
    device = model_meta["device"]
    image_token_id = model.config.image_token_id
    run_id = time.strftime("phase_2b_%Y%m%dT%H%M%SZ", time.gmtime())
    rng = np.random.default_rng(PATCH_SEED)

    from src.evaluation.candidate_scorer import score_candidate_answers

    def score_pair_inputs(inputs, source_label, target_label):
        with torch.inference_mode():
            out = model(**inputs)
            prompt_logits = out.logits[0, -1, :].detach().float()
        cand = score_candidate_answers(
            model=model,
            processor=processor,
            base_inputs=inputs,
            candidate_strings=CANDIDATES,
            device=device,
        )
        logprobs = {k: v.total_log_likelihood for k, v in cand.candidates.items()}
        m = score_pair_margins(logprobs, source_label, target_label)
        # greedy answer
        # use prompt final argmax among candidates as generated proxy + optional generate
        top_cand = max(logprobs.items(), key=lambda kv: kv[1])[0].strip()
        return {
            "logprobs": logprobs,
            "margin": m,
            "top_candidate": top_cand,
            "prompt_logits_finite": bool(torch.isfinite(prompt_logits).all()),
            "seq_len": int(inputs["input_ids"].shape[1]),
            "n_image_tokens": int((inputs["input_ids"] == image_token_id).sum()),
        }

    # ---------- Engineering validation on first pair ----------
    eng_pair = pairs[0]
    print(f"\n=== ENGINEERING VALIDATION on {eng_pair['pair_id']} ===")
    src_in = build_inputs(processor, eng_pair["source"], device=device)
    tgt_in = build_inputs(processor, eng_pair["target"], device=device)
    T_src = int(src_in["input_ids"].shape[1])
    T_tgt = int(tgt_in["input_ids"].shape[1])
    print(f"seq_len source={T_src} target={T_tgt}")

    # Unmodified target
    tgt_base = score_pair_inputs(tgt_in, eng_pair["source_label"], eng_pair["target_label"])
    # No-op target
    noop_logits, noop_meta = run_with_residual_patch(
        model, tgt_in, layer_index=LAYER, token_position=T_tgt - 1, replacement_row=None, no_op=True
    )
    # Compare prompt-final logits: recompute unmodified for exact equality check
    with torch.inference_mode():
        unmod = model(**tgt_in).logits[0, -1, :].detach().float()
    # Need logits from noop path - run_with_residual_patch returns logits
    max_diff_noop = float((noop_logits - unmod).abs().max())
    print(f"No-op max|Δlogits|={max_diff_noop:.3e} tol={NOOP_LOGIT_TOL}")
    eng_ok = max_diff_noop <= NOOP_LOGIT_TOL and noop_meta["hook_applied"] == 0
    if not eng_ok:
        atomic_write_json(
            out_dir / "engineering_validation_v1.json",
            {"status": "FAIL_NOOP", "max_diff": max_diff_noop, "noop_meta": noop_meta},
        )
        print("ENGINEERING GATE FAILED — aborting scientific pilot")
        return 3

    # Capture source residual + one non-identity patch smoke
    src_row, src_cap = capture_last_residual(model, src_in, layer_index=LAYER, device=device)
    smoke_logits, smoke_meta = run_with_residual_patch(
        model,
        tgt_in,
        layer_index=LAYER,
        token_position=T_tgt - 1,
        replacement_row=src_row,
        no_op=False,
    )
    smoke_diff = float((smoke_logits - unmod).abs().max())
    print(f"Non-identity patch max|Δlogits|={smoke_diff:.3e} hook_applied={smoke_meta['hook_applied']}")
    atomic_write_json(
        out_dir / "engineering_validation_v1.json",
        {
            "status": "PASS",
            "pair_id": eng_pair["pair_id"],
            "layer": LAYER,
            "noop_max_abs_logits_diff": max_diff_noop,
            "noop_tolerance": NOOP_LOGIT_TOL,
            "nonidentity_max_abs_logits_diff": smoke_diff,
            "source_capture_meta": src_cap["meta"],
            "noop_meta": noop_meta,
            "patch_meta": smoke_meta,
            "token_positions": {"source_last": T_src - 1, "target_last": T_tgt - 1},
        },
    )
    print("ENGINEERING GATE PASSED")

    # ---------- Scientific pilot ----------
    patched_budget = args.max_patched_forwards
    n_patched = 0
    records = []

    directions_plan = [
        ("source_to_target", "source"),
        ("target_to_source", "target"),
        ("same_label", "source"),  # same-label: patch source residual into source (no semantic change expected)
        ("unrelated_token", "source"),
        ("random_source", "source"),
    ]

    for pair in pairs:
        intervention_prefix = f"{pair['pair_id']}"
        src_rec, tgt_rec = pair["source"], pair["target"]
        src_label, tgt_label = pair["source_label"], pair["target_label"]
        src_in = build_inputs(processor, src_rec, device=device)
        tgt_in = build_inputs(processor, tgt_rec, device=device)
        T_s, T_t = int(src_in["input_ids"].shape[1]), int(tgt_in["input_ids"].shape[1])

        print(f"\n=== PAIR {pair['pair_id']} type={pair['pair_type']} T_src={T_s} T_tgt={T_t} ===")

        # Baselines
        if f"{intervention_prefix}::baseline_source" not in done_ids:
            s_sc = score_pair_inputs(src_in, src_label, tgt_label)
            row = {
                "intervention_id": f"{intervention_prefix}::baseline_source",
                "status": "ok",
                "role": "baseline_source",
                "pair_id": pair["pair_id"],
                "family_id": pair["family_id"],
                "pair_type": pair["pair_type"],
                "source_label": src_label,
                "target_label": tgt_label,
                "seq_len": s_sc["seq_len"],
                "n_image_tokens": s_sc["n_image_tokens"],
                "logprobs": s_sc["logprobs"],
                "margin": s_sc["margin"],
                "top_candidate": s_sc["top_candidate"],
                "model": model_meta,
                "layer": LAYER,
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            append_jsonl(jsonl_path, row)
            records.append(row)
            existing.append(row)
            done_ids.add(row["intervention_id"])
            print(f"  baseline_source m={s_sc['margin']['margin_source_minus_target']:.4f}")

        if f"{intervention_prefix}::baseline_target" not in done_ids:
            t_sc = score_pair_inputs(tgt_in, src_label, tgt_label)
            row = {
                "intervention_id": f"{intervention_prefix}::baseline_target",
                "status": "ok",
                "role": "baseline_target",
                "pair_id": pair["pair_id"],
                "family_id": pair["family_id"],
                "pair_type": pair["pair_type"],
                "source_label": src_label,
                "target_label": tgt_label,
                "seq_len": t_sc["seq_len"],
                "n_image_tokens": t_sc["n_image_tokens"],
                "logprobs": t_sc["logprobs"],
                "margin": t_sc["margin"],
                "top_candidate": t_sc["top_candidate"],
                "model": model_meta,
                "layer": LAYER,
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            append_jsonl(jsonl_path, row)
            records.append(row)
            existing.append(row)
            done_ids.add(row["intervention_id"])
            target_baseline_m = t_sc["margin"]["margin_source_minus_target"]
            print(f"  baseline_target m={target_baseline_m:.4f}")
        else:
            target_baseline_m = next(
                r["margin"]["margin_source_minus_target"]
                for r in existing
                if r.get("intervention_id") == f"{intervention_prefix}::baseline_target"
            )

        # Capture residuals once
        src_row, _ = capture_last_residual(model, src_in, layer_index=LAYER, device=device)
        tgt_row, _ = capture_last_residual(model, tgt_in, layer_index=LAYER, device=device)
        # unrelated: use last non-image token row from a different family sample if available else reverse target row
        # Use mean of target sequence non-last as unrelated source (documented)
        # Simpler unrelated: source residual from a *different* layer is not available; use tgt_row[0] substitute from unrelated position:
        # Re-capture using first token as unrelated token vector
        def capture_token_row(inputs, pos):
            cap = {}

            def hook(module, args, output):
                h = output[0] if isinstance(output, (tuple, list)) else output
                cap["row"] = h[0, pos, :].detach().float().cpu().numpy()
                return output

            handle = model.model.text_model.layers[LAYER].register_forward_hook(hook)
            try:
                with torch.inference_mode():
                    model(**inputs)
            finally:
                handle.remove()
            return cap["row"]

        unrelated_row = capture_token_row(tgt_in, 0)
        random_row = rng.normal(0, float(src_row.std() + 1e-6), size=src_row.shape).astype(np.float32)

        # Patch directions
        for direction, basis in directions_plan:
            iid = f"{intervention_prefix}::{direction}"
            if iid in done_ids:
                continue
            if n_patched >= patched_budget:
                print("PATCHED FORWARD CAP REACHED")
                break

            if direction == "noop":
                row_np = None
                no_op = True
            elif direction == "source_to_target":
                row_np, no_op = src_row, False
                target_in = tgt_in
            elif direction == "target_to_source":
                row_np, no_op = tgt_row, False
                target_in = src_in
            elif direction == "same_label":
                # Patch source residual into source run (same semantic label)
                row_np, no_op = src_row, False
                target_in = src_in
            elif direction == "unrelated_token":
                row_np, no_op = unrelated_row, False
                target_in = tgt_in
            elif direction == "random_source":
                row_np, no_op = random_row, False
                target_in = tgt_in
            else:
                raise ValueError(direction)

            if direction in {"source_to_target", "unrelated_token", "random_source"}:
                run_in = tgt_in
                run_pos = T_t - 1
                baseline_m = target_baseline_m
            elif direction == "target_to_source":
                run_in = src_in
                run_pos = T_s - 1
                # baseline for reverse is source baseline margin
                baseline_m = next(
                    r["margin"]["margin_source_minus_target"]
                    for r in existing
                    if r.get("intervention_id") == f"{intervention_prefix}::baseline_source"
                )
            elif direction == "same_label":
                run_in = src_in
                run_pos = T_s - 1
                baseline_m = next(
                    r["margin"]["margin_source_minus_target"]
                    for r in existing
                    if r.get("intervention_id") == f"{intervention_prefix}::baseline_source"
                )
            else:
                run_in = tgt_in
                run_pos = T_t - 1
                baseline_m = target_baseline_m

            t0 = time.perf_counter()
            with residual_patch_context(
                model,
                layer_index=LAYER,
                token_position=run_pos,
                replacement_row=row_np,
                no_op=no_op,
            ) as ctx:
                with torch.inference_mode():
                    logits = model(**run_in).logits[0, -1, :].detach().float()
                cand = score_candidate_answers(
                    model=model,
                    processor=processor,
                    base_inputs=run_in,
                    candidate_strings=CANDIDATES,
                    device=device,
                )
                hook_applied = ctx.applied
            meta = {
                "layer_index": LAYER,
                "token_position": run_pos,
                "no_op": no_op,
                "hook_applied": hook_applied,
            }
            if (not no_op) and hook_applied < 1:
                raise RuntimeError("Patch hook did not apply during scoring")
            n_patched += 0 if no_op else 1
            logprobs = {k: v.total_log_likelihood for k, v in cand.candidates.items()}
            m = score_pair_margins(logprobs, src_label, tgt_label)
            dm = delta_margin(baseline_m, m["margin_source_minus_target"])
            top_cand = max(logprobs.items(), key=lambda kv: kv[1])[0].strip()
            elapsed = time.perf_counter() - t0

            row = {
                "intervention_id": iid,
                "status": "ok",
                "role": "intervention",
                "direction": direction,
                "pair_id": pair["pair_id"],
                "family_id": pair["family_id"],
                "pair_type": pair["pair_type"],
                "source_label": src_label,
                "target_label": tgt_label,
                "layer": LAYER,
                "token_position": run_pos,
                "seq_len": int(run_in["input_ids"].shape[1]),
                "n_image_tokens": int((run_in["input_ids"] == image_token_id).sum()),
                "logprobs": logprobs,
                "margin": m,
                "delta_m": dm,
                "top_candidate": top_cand,
                "answer_shifted_toward_source": logprobs[_cand_key(src_label)] > logprobs[_cand_key(tgt_label)],
                "hook_meta": meta,
                "model": model_meta,
                "latency_sec": round(elapsed, 4),
                "peak_vram_mib": round(torch.cuda.max_memory_allocated(0) / (1024 * 1024), 3),
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            append_jsonl(jsonl_path, row)
            records.append(row)
            done_ids.add(iid)
            print(
                f"  {direction}: Δm={dm['delta_m']:+.4f} nie={dm['normalized_delta_m']} "
                f"({dm['nie_stability']}) top={top_cand!r} vram={row['peak_vram_mib']:.0f}",
                flush=True,
            )
            torch.cuda.empty_cache()

        if n_patched >= patched_budget:
            break

    del model, processor
    torch.cuda.empty_cache()

    # ---------- Analysis ----------
    rows = [r for r in load_jsonl(jsonl_path) if r.get("status") == "ok"]
    by_dir: Dict[str, List[float]] = {}
    for r in rows:
        if r.get("role") == "intervention" and "delta_m" in r:
            by_dir.setdefault(r["direction"], []).append(r["delta_m"]["delta_m"])

    def _sum(vals):
        return {
            "n": len(vals),
            "mean_delta_m": float(np.mean(vals)) if vals else None,
            "median_delta_m": float(np.median(vals)) if vals else None,
            "min": float(np.min(vals)) if vals else None,
            "max": float(np.max(vals)) if vals else None,
            "frac_positive": float(np.mean([v > 0 for v in vals])) if vals else None,
        }

    summary = {
        "phase": "2B",
        "run_id": run_id,
        "exec_spec": "configs/phase_2b/causal_pilot_exec_spec_v1.json",
        "n_families": len(families),
        "n_pairs": len(pairs),
        "n_interventions_executed": sum(1 for r in rows if r.get("role") == "intervention"),
        "n_patched_forwards": n_patched,
        "engineering_gate": json.loads((out_dir / "engineering_validation_v1.json").read_text(encoding="utf-8"))["status"],
        "delta_m_by_direction": {k: _sum(v) for k, v in by_dir.items()},
        "final_holdout_touched": False,
        "claims": {
            "software_intervention_success": True,
            "exploratory_causal_logit_influence": any(
                (v or 0) != 0 for vals in by_dir.values() for v in vals
            ),
            "relation_specific_mechanism": "NOT_ESTABLISHED",
            "unique_circuit": "NOT_ESTABLISHED",
            "logical_reasoning_restored": "NOT_ESTABLISHED",
        },
        "interpretation_notes": [
            "N=3 families; no inferential statistics claimed.",
            "Positive delta_m means margin shifted toward source label.",
            "Random/unrelated controls measure generic sensitivity.",
            "Patch of last token is not exclusive image-to-text pathway.",
        ],
        "peak_vram_mib_max": max((r.get("peak_vram_mib") or 0) for r in rows if r.get("peak_vram_mib")) if rows else None,
    }
    atomic_write_json(out_dir / "causal_pilot_summary_v1.json", summary)
    print(json.dumps(summary["delta_m_by_direction"], indent=2))
    print(f"Summary: {out_dir / 'causal_pilot_summary_v1.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
