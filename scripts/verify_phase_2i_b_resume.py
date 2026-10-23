"""Read-only existing-run CPU/GPU restoration audit; never calls optimizer.step.

Writes a new report outside the run; no events/checkpoints/journal are changed.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.training.phase_2i_b_protocol import build_plan
from src.training.run_state import RunStore, file_hash, write_once


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true", help="Load pinned local model and prove actual restoration; zero steps")
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    run, report_dir = args.run.resolve(), args.report_dir.resolve()
    assert run.is_relative_to(ROOT) and report_dir.is_relative_to(ROOT)
    plan = build_plan()
    before = {str(p.relative_to(run)): file_hash(p) for p in run.rglob("*") if p.is_file() and p.name != ".run.lock"}
    report = {"run_id": run.name, "holdout_evaluated": False, "optimizer_steps": 0,
              "gpu_requested": args.gpu, "identity": plan["identity"]}
    try:
        import torch
        from src.training.run_state import capture_rng, validate_rng, validate_optimizer, optimizer_layout
        with RunStore(run, plan["identity"]) as store:
            report["inspection"] = store.inspect()
            assert store.step == 5, "expected historical step-five continuation only"
            assert run.name == "56e277e421e442b08d33b107fc934494"
            report["validated_checkpoint_steps"] = list(range(store.step + 1))
            for step, checkpoint in enumerate(store.checkpoints):
                blob = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
                params = [torch.nn.Parameter(torch.empty(x["shape"], dtype=getattr(torch, x["dtype"].split(".")[-1]))) for x in blob["layout"]]
                opt = torch.optim.AdamW(params, lr=plan["identity"]["optimizer"]["learning_rate"])
                validate_optimizer(blob, opt, blob["layout"], step)
                validate_rng(torch.load(checkpoint / "rng.pt", map_location="cpu", weights_only=True))
            if args.gpu:
                import psutil, shutil
                from src.config.paths import apply_default_cache_environment
                from src.models.qwen3vl2_loader import load_qwen3vl2_nf4
                from src.training.scorer_verification import ParameterAudit, run_observed
                from src.training.run_state import canonical
                available = psutil.virtual_memory().available
                report["available_ram_bytes_before_load"] = available
                assert available >= 4 * 1024**3, "RAM_GATE_FAIL: require 4 GiB physical available"
                assert torch.cuda.is_available(), "CUDA required"
                # Reserve measured remaining checkpoint growth plus audit-snapshot
                # storage, bounded by the pinned model's 2B-scale loaded storage.
                required = 4 * 1024**3 + 45 * sum(p.stat().st_size for p in store.checkpoints[-1].iterdir() if p.is_file())
                report["required_disk_bytes"] = required
                assert shutil.disk_usage(ROOT).free >= required, "insufficient disk reserve"
                apply_default_cache_environment()
                os.environ["HF_HUB_OFFLINE"] = "1"
                spec = importlib.util.spec_from_file_location("frozen_es_runner", ROOT / "scripts/run_phase_2i_b_es_ablation.py")
                runner = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(runner)
                loaded = {}
                def action():
                    processor, base, meta = load_qwen3vl2_nf4(local_files_only=True)
                    model, _ = runner.attach_lora(base, r=2)
                    loaded.update(processor=processor, model=model)
                    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
                    loaded["optimizer"] = optimizer
                    with ParameterAudit(ROOT / ".cache/recovery-temp") as audit:
                        audit.capture(model)
                        store.restore(model, optimizer, *runner.adapter_restorers(model, plan["identity"]["lora"]))
                        actual_rng = capture_rng()  # Immediately after restoration.
                        expected_rng = torch.load(store.checkpoints[-1] / "rng.pt", map_location="cpu", weights_only=True)
                        assert actual_rng["python"] == expected_rng["python"]
                        assert actual_rng["numpy"] == expected_rng["numpy"]
                        assert torch.equal(actual_rng["torch_cpu"], expected_rng["torch_cpu"])
                        assert len(actual_rng["cuda"]) == len(expected_rng["cuda"]) == 1
                        assert all(torch.equal(a, b) for a, b in zip(actual_rng["cuda"], expected_rng["cuda"]))
                        comparison = audit.compare(model)
                        assert all(r["passed"] for r in comparison["parameters"] if not r["requires_grad_before"])
                        report["restoration_parameter_comparison"] = comparison
                    blob = torch.load(store.checkpoints[-1] / "optimizer.pt", map_location="cpu", weights_only=True)
                    validate_optimizer({"class": "AdamW", "layout": optimizer_layout(model, optimizer), "state": optimizer.state_dict()}, optimizer, blob["layout"], 5)
                    for key, expected in blob["state"]["state"].items():
                        actual = optimizer.state_dict()["state"][key]
                        assert all(torch.equal(actual[k].cpu(), v) for k, v in expected.items())
                    report.update(adapter_exact_reload=True, optimizer_moments_exact_reload=True,
                                  rng_exact_reload=True, restored_step=store.step, load_meta=meta)
                def cleanup():
                    loaded.clear()
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                assert run_observed(report, action, cleanup) == 0, report.get("error_message")
            report["status"] = "PASSED"
    except Exception as exc:
        import traceback
        report.update(status="FAILED", error=str(exc), traceback=traceback.format_exc())
    report["historical_run_files_unchanged"] = all((run / p).is_file() and file_hash(run / p) == h for p, h in before.items())
    assert report["historical_run_files_unchanged"]
    destination = report_dir / f"resume_validation_{uuid4().hex}.json"
    write_once(destination, report)
    print(json.dumps({"report": str(destination), **{k: v for k, v in report.items() if k not in ("identity", "restoration_parameter_comparison", "telemetry")}}, indent=2))
    return 0 if report["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
