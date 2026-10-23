"""Read-only saved-state verification; no model load or optimizer step."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
from safetensors.torch import load_file
from src.training.run_state import RunStore, file_hash, validate_optimizer, validate_rng, write_once
from src.training.phase_2i_b_protocol import build_plan


def main():
    evidence = Path(__file__).resolve().parent
    preflight = json.loads((evidence / "preflight_and_plan.json").read_text())
    identity = preflight["identity"]
    assert build_plan()["identity"] == identity, "executed source/runtime/plan drift"
    run = ROOT / "artifacts/training/phase_2i_b/runs/56e277e421e442b08d33b107fc934494"
    checkpoints = []
    with RunStore(run, identity) as store:
        inspection = store.inspect()
        assert store.step == 5 and len(store.rows) == 5
        for step, checkpoint in enumerate(store.checkpoints):
            blob = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
            params = [torch.nn.Parameter(torch.empty(x["shape"], dtype=getattr(torch, x["dtype"].split(".")[-1])))
                      for x in blob["layout"]]
            optimizer = torch.optim.AdamW(params, lr=identity["optimizer"]["learning_rate"])
            validate_optimizer(blob, optimizer, blob["layout"], step)
            rng = torch.load(checkpoint / "rng.pt", map_location="cpu", weights_only=True)
            validate_rng(rng)
            assert len(rng["cuda"]) == 1 and rng["numpy"] is not None
            adapter = load_file(str(checkpoint / "adapter_model.safetensors"))
            assert len(adapter) == 224 and all(torch.isfinite(v).all() for v in adapter.values())
            checkpoints.append({"step": step, "optimizer_states": len(blob["state"]["state"]),
                                "rng_valid": True, "cuda_rng_devices": len(rng["cuda"]),
                                "adapter_tensors": len(adapter), "hash_chain_valid": True})
        for row in store.rows:
            assert row["parameter_acceptance"]["passed"] and row["grad_l2"] > 0
            assert abs(row["total_loss"] - row["ce_loss"] - .25 * row["es_loss"]) < 2e-6
    execution = json.loads(next(run.glob("execution_*.json")).read_text())
    samples = execution["resource_samples"]
    def agg(key, op):
        return op(s[key] for s in samples if s.get(key) is not None)
    telemetry = {"elapsed_sec": execution["elapsed_sec"], "sample_count": len(samples),
                 "maximum_sampled_nvidia_mib": agg("nvidia_smi_used_mib", max),
                 "minimum_sampled_host_available_mib": agg("host_available_mib", min),
                 "maximum_sampled_commit_bytes": agg("host_commit_bytes", max),
                 "max_step_allocated_mib": max(r["memory"]["torch_max_allocated_mib"] for r in store.rows),
                 "max_step_reserved_mib": max(r["memory"]["torch_max_reserved_mib"] for r in store.rows)}
    assert execution["status"] == "PAUSED" and execution["completed_step"] == 5
    assert not execution["sampling_errors"] and telemetry["minimum_sampled_host_available_mib"] >= 1500
    assert telemetry["maximum_sampled_nvidia_mib"] < 4096
    unchanged = all(file_hash(ROOT / p) == h for p, h in preflight["history_sha256"].items())
    assert unchanged, "historical evidence changed"
    ce_rows = [json.loads(x) for x in (ROOT / "artifacts/training/phase_2i_a/train_metrics_append.jsonl").read_text().splitlines()]
    assert [r["family_id"] for r in ce_rows[:5]] == [r["family_id"] for r in store.rows]
    assert [r["label"] for r in ce_rows[:5]] == [r["label"] for r in store.rows]
    report = {"status": "PASSED", "actual_optimizer_steps": 5, "inspection": inspection,
              "checkpoints": checkpoints, "telemetry": telemetry,
              "historical_files_unchanged": len(preflight["history_sha256"]),
              "holdout_evaluated": False, "ce_first_five_order_and_labels_match": True,
              "note": "CPU deserialization and state validation; no real-model resume or extra optimizer steps"}
    destination = evidence / "acceptance.json"
    if destination.exists():
        assert json.loads(destination.read_text()) == report, "saved acceptance evidence differs"
    else:
        write_once(destination, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
