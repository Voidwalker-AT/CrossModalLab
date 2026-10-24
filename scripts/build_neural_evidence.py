"""Validate existing CPU artifacts and package small neural diagnostics; no forward pass.

NumPy is used only for existing local NPZ validation. Dashboard runtime is stdlib.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.instrumentation.capture_integrity import identity_from_stored_fields, verify_vector_artifact

MODEL = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
REVISION = "482adb537c021c86670beed01cd58990d01e72e4"


def build(root=ROOT):
    import numpy as np
    sources = {}
    def source(name):
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("source outside checkout")
        sources[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path
    def read(name):
        return json.loads(source(name).read_text(encoding="utf-8"))
    meta_path = "artifacts/instrumentation/phase_1a/activations_meta_v1.jsonl"
    metadata = [json.loads(line) for line in source(meta_path).read_text().splitlines()]
    captures, inventory = [], []
    ids = set()
    for row in metadata:
        if row.get("is_synthetic_demo") is True or row["status"] != "ok":
            raise ValueError("non-empirical capture cannot enter gallery")
        assert row["sample_id"] not in ids
        ids.add(row["sample_id"])
        assert row["model"]["model_id"] == MODEL and row["model"]["revision"] == REVISION
        path = source(row["vectors_file"])
        issues = verify_vector_artifact(vectors_file=path, expected_layers=row["layers"],
                                       expected_selectors=["last", "first_image", "last_image", "mean_image", "last_text_non_image"])
        if issues:
            raise ValueError(str(issues))
        raw = json.loads(path.read_text())
        assert raw["logits_identical"] and raw["logits_max_abs_diff"] == 0 and not raw["has_nonfinite"]
        points = []
        for layer, bundle in raw["layers"].items():
            for selector, token in bundle["tokens"].items():
                vector = token["vector"]
                assert token["shape"] == [2048] and len(vector) == 2048 and all(math.isfinite(v) for v in vector)
                calculated = math.sqrt(math.fsum(v*v for v in vector))
                # Source norms are rounded to six decimals after float32 norm.
                tolerance = 8 * (2**-23) * max(1, calculated) + .5e-6
                assert abs(calculated-token["l2_norm"]) <= tolerance, (row["sample_id"], layer, selector)
                index = token["token_index"]
                assert (selector == "mean_image" and index is None) or (isinstance(index, int) and 0 <= index < raw["sequence_length"])
                points.append({"layer": int(layer), "selector": selector, "token_index": index,
                               "l2_norm": token["l2_norm"], "recomputed_l2": calculated,
                               "shape": token["shape"], "dtype": token["native_dtype"]})
        relative = path.relative_to(root).as_posix()
        captures.append({"sample_id": row["sample_id"], "family_id": row["matched_group_id"],
                         "role": row["role"], "ground_truth": row["ground_truth_label"],
                         "historical_parsed_answer": row["parsed_answer"],
                         "historical_correct": row["is_correct_0d2"],
                         "behavior_alignment": "Historical Phase 0D2 answer; not a fresh generation during this passive capture",
                         "matched_group_id": row["matched_group_id"] if row["genuinely_paired"] else None,
                         "run_id": row["run_id"], "condition": raw["input_condition"],
                         "image_sha256": row["image_sha256"], "sequence_length": raw["sequence_length"],
                         "points": points, "source": relative, "sha256": sources[relative],
                         "classification": "CORRELATED", "validation": "vector shape/finite/norm and stored zero logit drift verified on CPU"})
        inventory.append({"source": relative, "model": MODEL, "revision": REVISION,
                          "sample_id": row["sample_id"], "layers": row["layers"], "tensor_shape": [2048],
                          "token_selectors": sorted(raw["layers"]["3"]["tokens"]), "dtype": "float32",
                          "method": "passive decoder residual-output sparse rows; L2 norm",
                          "condition": raw["input_condition"], "sha256": sources[relative],
                          "classification": "CORRELATED", "validation_status": "CPU_REVALIDATED"})
    probe_path = "artifacts/probes/phase_1f/probe_reliability_summary_v1.json"
    probe = read(probe_path)
    assert probe["final_holdout_touched"] is False
    feature_meta = [json.loads(line) for line in source("artifacts/probes/phase_1f/features_meta_v1.jsonl").read_text().splitlines()]
    for row in feature_meta:
        assert row["split_role"] in {"train", "validation"}, row["split_role"]
        assert row["model"]["model_id"] == MODEL and row["model"]["revision"] == REVISION
        identity = identity_from_stored_fields(row)
        assert identity and identity.identity_hash() == row["capture_identity"]["identity_hash"]
        path = source(row["vectors_file"])
        with np.load(path, allow_pickle=False) as arrays:
            assert arrays.files == ["vector"]
            vector = arrays["vector"]
            assert vector.shape == (24576,) and str(vector.dtype) == "float32" and bool(np.isfinite(vector).all())
    probe_points = []
    for result in probe["probe_results"]:
        spec = result["spec"]
        assert result["n_train"] == 81 and result["n_val"] == 27
        assert result["group_bootstrap_ci"]["n_groups"] == 9
        if spec["use_pca"]:
            continue
        cm = result["confusion_matrix"]
        calculated = sum(cm[label][label] / sum(cm[label].values()) for label in ("Yes", "No", "Undetermined")) / 3
        assert abs(calculated-result["balanced_accuracy"]) <= .5e-6
        probe_points.append({"layer": spec["layer"], "selector": spec["token_selector"],
                             "balanced_accuracy": result["balanced_accuracy"], "ci": result["group_bootstrap_ci"],
                             "l2_regularization": result["best_l2"], "shape": [result["feature_dim_raw"]]})
    inventory.append({"source": probe_path, "sha256": sources[probe_path], "model": MODEL, "revision": REVISION,
                      "layers": [3,12,18,23], "tensor_shape": [2048], "token_selectors": ["last","last_text_non_image","mean_image"],
                      "dtype": "stored metrics / float32 input features", "method": "ridge linear probe, development-selected L2/layer/selector",
                      "classification": "CORRELATED", "validation_status": "108 feature identities/shapes/finite and stored confusion metrics checked"})
    causal_path = "artifacts/causal/phase_2b/recovery_audit_v1.json"
    causal = read(causal_path)
    records_path = "artifacts/causal/phase_2b/intervention_records_v1.jsonl"
    records = [json.loads(line) for line in source(records_path).read_text().splitlines()]
    assert len(records) == causal["evidence_file"]["n_valid"] == 16
    inventory.append({"source": records_path, "sha256": sources[records_path], "model": MODEL, "revision": REVISION,
                      "layers": [18], "tensor_shape": [2048], "token_selectors": ["last"],
                      "dtype": "float32 capture; stored margin deltas", "method": "historical activation patching with negative controls",
                      "classification": "UNVERIFIED", "validation_status": "16 surviving rows; controls invalidate relation-specific causal attribution"})
    payload = {"schema": "crossmodallab_neural_v1", "is_synthetic_demo": False,
               "model": {"id": MODEL, "revision": REVISION, "checkpoint": "historical frozen NF4 Smol; not Qwen adapters"},
               "captures": captures, "probe": {"points": probe_points, "train_cases": 81, "validation_cases": 27,
                   "train_families": 27, "validation_families": 9, "source": probe_path,
                   "family_ids": probe["val_families"], "classification": "CORRELATED",
                   "limitation": "Layer/selector/L2 chosen on development results; probe accessibility does not show causal usage."},
               "causal_audit": {"source": causal_path, "records": 16, "interventions": 10,
                   "missing_planned": 26, "targeted_abs_delta": causal["effect_audit"]["targeted_abs_mean"],
                   "unrelated_token_abs_delta": causal["effect_audit"]["max_control_abs_mean"],
                   "classification": "UNVERIFIED", "causal_mechanism_established": False,
                   "limitation": "Negative controls exceed targeted effects; incomplete/post-hoc recovery. No confirmatory causal heatmap."},
               "missing": ["Qwen per-case activation capture", "validated intermediate candidate-answer readout", "attention-head tensors", "separate MLP outputs", "individual neuron evidence", "confirmatory relation-specific causal circuit"],
               "alignment": "Match only Phase 1A genuinely_paired group + identical image hash/model/condition. Compare same token selector at its recorded position; text and sequence lengths may differ. Not an isolated failure intervention.",
               "inventory": inventory, "sources": [{"path": k, "sha256": v} for k,v in sorted(sources.items())],
               "validation": {"captures": len(captures), "activation_rows": sum(len(c["points"]) for c in captures),
                              "feature_files": len(feature_meta), "probe_points": len(probe_points),
                              "new_gpu_forwards": 0, "new_optimizer_steps": 0}}
    return payload


if __name__ == "__main__":
    target = ROOT / "dashboard/assets/neural_evidence_v1.json"
    if target.exists():
        raise SystemExit("Existing curated bundle preserved; review/version it explicitly before regeneration")
    payload = build()
    target.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload["validation"]))
