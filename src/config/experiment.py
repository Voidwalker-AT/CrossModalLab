"""Shared experiment configuration for CrossModalLab.

Loads the repository-pinned model, quantization, dataset, seed, prompt, and
scoring configuration from ``configs/experiment_config.json``. Both ULVO and
standard-Python contributors must observe the same values for comparable runs.

This module intentionally does not import torch/transformers at import time so
that symbolic CPU-only workflows remain dependency-free.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from src.config.paths import REPO_ROOT

CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.json"


@dataclass(frozen=True)
class QuantizationConfig:
    load_in_4bit: bool
    bnb_4bit_quant_type: str
    bnb_4bit_compute_dtype: str
    bnb_4bit_use_double_quant: bool


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    revision: str
    local_files_only_default: bool


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    seed: int
    model: ModelConfig
    quantization: QuantizationConfig
    prompt_template: Dict[str, Any]
    candidate_answers: Dict[str, Any]
    scoring: Dict[str, Any]
    datasets: Dict[str, Any]
    artifacts: Dict[str, Any]
    generation: Dict[str, Any]

    @property
    def candidate_answer_strings(self) -> list[str]:
        return list(self.candidate_answers["strings"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "model": {
                "model_id": self.model.model_id,
                "revision": self.model.revision,
                "local_files_only_default": self.model.local_files_only_default,
            },
            "quantization": {
                "load_in_4bit": self.quantization.load_in_4bit,
                "bnb_4bit_quant_type": self.quantization.bnb_4bit_quant_type,
                "bnb_4bit_compute_dtype": self.quantization.bnb_4bit_compute_dtype,
                "bnb_4bit_use_double_quant": self.quantization.bnb_4bit_use_double_quant,
            },
            "prompt_template": self.prompt_template,
            "candidate_answers": self.candidate_answers,
            "scoring": self.scoring,
            "datasets": self.datasets,
            "artifacts": self.artifacts,
            "generation": self.generation,
        }


@lru_cache(maxsize=1)
def get_experiment_config() -> ExperimentConfig:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"Experiment config not found: {CONFIG_PATH}. "
            "Clone the full repository; do not invent substitute model revisions."
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    model = raw["model"]
    quant = raw["quantization"]
    return ExperimentConfig(
        schema_version=int(raw["schema_version"]),
        seed=int(raw["seed"]),
        model=ModelConfig(
            model_id=model["model_id"],
            revision=model["revision"],
            local_files_only_default=bool(model.get("local_files_only_default", True)),
        ),
        quantization=QuantizationConfig(
            load_in_4bit=bool(quant["load_in_4bit"]),
            bnb_4bit_quant_type=str(quant["bnb_4bit_quant_type"]),
            bnb_4bit_compute_dtype=str(quant["bnb_4bit_compute_dtype"]),
            bnb_4bit_use_double_quant=bool(quant["bnb_4bit_use_double_quant"]),
        ),
        prompt_template=raw["prompt_template"],
        candidate_answers=raw["candidate_answers"],
        scoring=raw["scoring"],
        datasets=raw["datasets"],
        artifacts=raw["artifacts"],
        generation=raw["generation"],
    )


class CapabilityError(RuntimeError):
    """Raised when local hardware/software cannot support the pinned GPU config."""


def require_gpu_capabilities(
    *,
    allow_cpu_fallback: bool = False,
    require_bitsandbytes: bool = True,
    require_bf16: bool = True,
) -> Dict[str, Any]:
    """Check whether this machine can run the pinned GPU experiment configuration.

    Never silently substitutes a different model or quantization scheme.
    Raises ``CapabilityError`` with an explicit remediation message on failure
    unless ``allow_cpu_fallback`` is True (then returns a status dict only for
    non-quantized CPU paths that the caller must handle explicitly).
    """
    cfg = get_experiment_config()
    status: Dict[str, Any] = {
        "model_id": cfg.model.model_id,
        "revision": cfg.model.revision,
        "quantization": cfg.quantization.bnb_4bit_quant_type if cfg.quantization.load_in_4bit else "none",
        "torch_available": False,
        "cuda_available": False,
        "bitsandbytes_available": False,
        "bf16_supported": False,
        "device_name": None,
        "supported": False,
        "reasons": [],
    }

    try:
        import torch  # type: ignore
    except ImportError as exc:
        status["reasons"].append(f"torch is not installed: {exc}")
        if allow_cpu_fallback:
            return status
        raise CapabilityError(
            "PyTorch is not installed. Install GPU dependencies from requirements.lock "
            "(Mode A / ULVO) or follow docs/DEPENDENCIES.md for a portable install. "
            "Do not expect quantized SmolVLM2 experiments to run without torch."
        ) from exc

    status["torch_available"] = True
    status["torch_version"] = getattr(torch, "__version__", "unknown")
    status["cuda_available"] = bool(torch.cuda.is_available())

    if require_bitsandbytes:
        try:
            import bitsandbytes  # noqa: F401

            status["bitsandbytes_available"] = True
        except Exception as exc:  # bitsandbytes may fail import for ABI reasons
            status["reasons"].append(f"bitsandbytes unavailable: {exc}")

    if status["cuda_available"]:
        props = torch.cuda.get_device_properties(0)
        status["device_name"] = props.name
        status["vram_mib"] = round(props.total_memory / (1024 * 1024), 2)
        if require_bf16:
            status["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    else:
        status["reasons"].append("CUDA is not available")

    problems = []
    if not status["cuda_available"]:
        problems.append("CUDA device required for pinned 4-bit NF4 quantization")
    if require_bitsandbytes and not status["bitsandbytes_available"]:
        problems.append("bitsandbytes required for NF4 quantization")
    if status["cuda_available"] and require_bf16 and not status["bf16_supported"]:
        problems.append("bfloat16 compute required by pinned quantization config")

    status["supported"] = not problems
    if problems and not allow_cpu_fallback:
        raise CapabilityError(
            "This machine cannot run the pinned GPU experiment configuration "
            f"({cfg.model.model_id} @ {cfg.model.revision}, {cfg.quantization.bnb_4bit_quant_type} 4-bit). "
            "Details: " + "; ".join(problems + status["reasons"]) + ". "
            "CrossModalLab does not silently switch models or quantization methods. "
            "For CPU-only symbolic workflows, run: python -m unittest discover -s tests. "
            "See docs/DEPENDENCIES.md and docs/ASSET_MANIFEST.md."
        )

    return status
