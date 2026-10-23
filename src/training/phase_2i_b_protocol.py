"""CPU-safe frozen CE+ES experiment planning; no model loading or optimization."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform

from src.config.paths import REPO_ROOT
from src.models.qwen3vl2_loader import QWEN_MODEL_ID, QWEN_REVISION
from src.training.answer_parser import PARSER_VERSION
from src.training.run_state import digest, file_hash
from src.training.sampling_and_masking import maximize_family_coverage_schedule

CANDIDATES = (" Yes", " No", " Undetermined")
LABEL_INDEX = {"Yes": 0, "No": 1, "Undetermined": 2}


def validate_objective(es):
    expected = {"mode": "margin", "margin": 1.0, "mu_es": 0.25, "ce_weight": 1.0,
        "class_scores": "summed_candidate_sequence_log_likelihood", "length_normalized": False,
        "candidates": list(CANDIDATES), "abstain_index": 2,
        "sufficiency_from_gold": "0 if Undetermined else 1", "anti_over_abstention": True}
    if es != expected:
        raise ValueError("frozen ES objective mismatch; no silent tuning or CE substitution")


def compose_objective(ce, model, prompt_inputs, candidate_ids, label, es_config):
    """L = CE + 0.25 ES; full summed candidate LLs, no detach in the graph."""
    import torch
    from src.training.differentiable_scoring import (
        candidate_class_loglikelihoods, evidence_sufficiency_from_class_scores, total_loss,
    )
    validate_objective(es_config)
    gold = LABEL_INDEX[label]
    scores = candidate_class_loglikelihoods(model, prompt_inputs, candidate_ids, candidates=CANDIDATES)
    es = evidence_sufficiency_from_class_scores(scores, torch.tensor([gold], device=scores.device),
        sufficiency=torch.tensor([0 if gold == 2 else 1], device=scores.device), margin=es_config["margin"])
    return total_loss(ce, es, mu_es=es_config["mu_es"], ce_weight=es_config["ce_weight"]), es, scores


def build_plan(*, root=REPO_ROOT, dataset=None):
    """Hash the manifest; select/use only train and validation records and images."""
    root = root.resolve()
    cfg_path = root / "configs/phase_2i_b/es_ablation_config_v1.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    base_path = (root / cfg["base_protocol"]).resolve()
    if not base_path.is_relative_to(root):
        raise ValueError("protocol path outside workspace")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    validate_objective(cfg["es_objective"])
    if base["quantization"] != {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                                 "bnb_4bit_compute_dtype": "bfloat16", "bnb_4bit_use_double_quant": True}:
        raise ValueError("pinned NF4 quantization mismatch")
    matched, training = cfg["matched_to_2i_a"], base["training"]
    if (base["model_id"] != QWEN_MODEL_ID or base["revision"] != QWEN_REVISION
            or matched["holdout_allowed"] is not False or base["data"]["holdout_allowed"] is not False
            or cfg["no_tuning_on_val"] is not True or matched["parser"] != PARSER_VERSION):
        raise ValueError("model, parser, or holdout policy mismatch")
    for key in ("seed", "max_steps", "sanity_steps", "optimizer", "learning_rate", "sampling", "masking"):
        if matched[key] != training[key]:
            raise ValueError(f"CE control protocol mismatch: {key}")
    if (matched["lora_r"] != base["lora"]["r"] or matched["lora_alpha"] != base["lora"]["lora_alpha"]
            or training["batch_size"] != 1 or training["grad_accum"] != 1
            or not training["gradient_checkpointing"] or training["use_cache"]):
        raise ValueError("LoRA/training execution mismatch")
    dataset = (dataset or (root / base["data"]["dataset"])).resolve()
    if not dataset.is_relative_to(root):
        raise ValueError("dataset outside workspace")
    records = []
    with dataset.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["split"] in {"train", "validation"}:
                records.append(row)
    train = [r for r in records if r["split"] == "train"]
    val = [r for r in records if r["split"] == "validation"]
    if len({r["sample_id"] for r in records}) != len(records):
        raise ValueError("duplicate development sample IDs")
    if {r["family_id"] for r in train} & {r["family_id"] for r in val}:
        raise ValueError("train/validation family leakage")
    families = sorted({r["family_id"] for r in val})[:6]
    if len(families) != 6:
        raise ValueError("six validation families required")
    validation = sorted((r for r in val if r["family_id"] in families and r["variant_id"] in {
        "entailed", "contradicted", "underdetermined"}), key=lambda r: (r["family_id"], r["variant_id"]))
    if any(sorted(r["ground_truth_label"] for r in validation if r["family_id"] == fid)
           != sorted(LABEL_INDEX) for fid in families):
        raise ValueError("validation families must contain exactly three primary cases")
    schedule = maximize_family_coverage_schedule(train, max_steps=matched["max_steps"], seed=matched["seed"])
    def identify(row):
        image = (root / row["image_file"]).resolve()
        if not image.is_relative_to(root):
            raise ValueError("image outside workspace")
        if (hashlib.sha256(row["prompt"].encode()).hexdigest() != row["prompt_sha256"]
                or file_hash(image) != row["image_sha256"]):
            raise ValueError("prompt/image identity mismatch")
        if row["ground_truth_label"] not in LABEL_INDEX or row.get("is_synthetic_demo") is not False:
            raise ValueError("invalid label or demo entering empirical run")
        return {k: row[k] for k in ("sample_id", "family_id", "ground_truth_label", "prompt_sha256", "image_sha256")}
    selected = [identify(r) for r in schedule]
    held_dev = [identify(r) for r in validation]
    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None
    source_names = ("scripts/run_phase_2i_b_es_ablation.py", "src/training/run_state.py",
        "src/training/phase_2i_b_protocol.py", "src/training/differentiable_scoring.py",
        "src/training/sampling_and_masking.py", "src/training/answer_parser.py", "src/training/lora_config.py",
        "src/models/qwen3vl2_loader.py")
    identity = {"model_id": QWEN_MODEL_ID, "revision": QWEN_REVISION, "quantization": base["quantization"],
        "lora": base["lora"], "config": cfg, "base_protocol": base,
        "config_sha256": file_hash(cfg_path), "base_protocol_sha256": file_hash(base_path),
        "dataset_sha256": file_hash(dataset), "dataset": str(dataset.relative_to(root)),
        "schedule": selected, "schedule_sha256": digest(selected), "validation": held_dev,
        "prompt_configuration": {"training_abstention_prompt": False, "generation_max_tokens": 16,
                                 "do_sample": False, "parser": PARSER_VERSION,
                                 "masking_implementation": "gold_answer_span_exact"},
        "optimizer": {"class": "AdamW", "learning_rate": matched["learning_rate"], "betas": [0.9, 0.999],
                      "eps": 1e-8, "weight_decay": 0.01, "amsgrad": False, "scheduler": None},
        "source_sha256": {name: file_hash(REPO_ROOT / name) for name in source_names},
        "runtime": {"python": platform.python_version(), **{name: version(name) for name in (
            "torch", "transformers", "peft", "bitsandbytes", "numpy", "safetensors")}},
        "holdout_used": False}
    return {"identity": identity, "config": cfg, "schedule": schedule, "validation": validation}
