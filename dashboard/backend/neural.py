"""Load a CPU-validated, small recorded-neural bundle without ML dependencies."""
import json
import math

MODEL = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
REVISION = "482adb537c021c86670beed01cd58990d01e72e4"


def validate(bundle):
    if bundle.get("schema") != "crossmodallab_neural_v1" or bundle.get("is_synthetic_demo") is not False:
        raise ValueError("Only curated empirical neural evidence is supported")
    if bundle["model"]["id"] != MODEL or bundle["model"]["revision"] != REVISION:
        raise ValueError("Neural model identity mismatch; never attribute Smol captures to Qwen")
    seen = set()
    for capture in bundle["captures"]:
        if capture["sample_id"] in seen or capture["classification"] != "CORRELATED":
            raise ValueError("duplicate capture or unsupported interpretation")
        seen.add(capture["sample_id"])
        for point in capture["points"]:
            if point["layer"] not in (3,12,18,23) or point["shape"] != [2048]:
                raise ValueError("unsupported neural shape/layer")
            if not math.isfinite(point["l2_norm"]) or point["l2_norm"] < 0:
                raise ValueError("invalid measured norm")
    if bundle["causal_audit"]["causal_mechanism_established"] is not False:
        raise ValueError("Unsupported causal-mechanism claim")
    return {**bundle, "available": True}


def load(root):
    path = root / "dashboard/assets/neural_evidence_v1.json"
    if not path.is_file():
        return {"available": False, "status": "NOT MEASURED", "reason": "Curated neural bundle unavailable"}
    return validate(json.loads(path.read_text(encoding="utf-8")))
