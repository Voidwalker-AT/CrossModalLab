"""SYNTHETIC DEMO: CPU-only persistence/loss fixtures, never empirical results."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout, redirect_stderr
import io

from src.config.paths import REPO_ROOT
from src.training.run_state import RunStore, capture_rng, digest, file_hash, restore_rng
from src.training.phase_2i_b_protocol import build_plan, validate_objective

try:
    import torch
    from safetensors.torch import load_file, save_file
except ImportError:
    torch = None


def runner_module():
    spec = importlib.util.spec_from_file_location("es_runner_under_test", REPO_ROOT / "scripts/run_phase_2i_b_es_ablation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_identity():
    return {"is_synthetic_demo": True, "watermark": "SYNTHETIC DEMO", "revision": "fixture-revision",
            "dataset_sha256": hashlib.sha256(b"SYNTHETIC DEMO").hexdigest(), "mu_es": 0.25,
            "schedule": [{"sample_id": "demo-one"}, {"sample_id": "demo-two"}]}


@unittest.skipIf(torch is None, "optional CPU torch/safetensors unavailable")
class TestRunState(unittest.TestCase):
    is_synthetic_demo = True

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.identity = synthetic_identity()
        self.store = RunStore.fresh(Path(self.tmp.name), self.identity)
        self.store.__enter__()
        self.addCleanup(self.store.__exit__, None, None, None)
        self.model = torch.nn.Module()
        self.model.register_parameter("lora_weight", torch.nn.Parameter(torch.tensor([1., 2.])))
        self.model.register_parameter("frozen", torch.nn.Parameter(torch.tensor([3.]), requires_grad=False))
        self.optimizer = torch.optim.AdamW([self.model.lora_weight], lr=1e-4)
        # Even synthetic tests must not perform optimizer steps.
        self.guard = patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("optimizer steps forbidden"))
        self.guard.start()
        self.addCleanup(self.guard.stop)

    def save_adapter(self, path):
        save_file({"lora_weight": self.model.lora_weight.detach().contiguous()}, str(path / "adapter_model.safetensors"))
        (path / "adapter_config.json").write_text(json.dumps({"is_synthetic_demo": True, "watermark": "SYNTHETIC DEMO"}))

    def validate_adapter(self, path):
        state = load_file(str(path / "adapter_model.safetensors"))
        if set(state) != {"lora_weight"} or state["lora_weight"].shape != self.model.lora_weight.shape:
            raise ValueError("adapter mismatch")

    def load_adapter(self, path):
        self.validate_adapter(path)
        with torch.no_grad():
            self.model.lora_weight.copy_(load_file(str(path / "adapter_model.safetensors"))["lora_weight"])

    def commit(self, step):
        if step:
            # Hand-constructed SYNTHETIC DEMO state, no optimizer operation.
            self.optimizer.state[self.model.lora_weight] = {"step": torch.tensor(float(step)),
                "exp_avg": torch.zeros_like(self.model.lora_weight), "exp_avg_sq": torch.ones_like(self.model.lora_weight)}
        row = None if not step else {"step": step, "sample_id": self.identity["schedule"][step - 1]["sample_id"],
                                    "is_synthetic_demo": True, "watermark": "SYNTHETIC DEMO"}
        self.store.commit(step, row, self.model, self.optimizer, self.save_adapter)

    def rewrite_component(self, name, mutate):
        checkpoint = self.store.checkpoints[-1]
        path = checkpoint / name
        blob = torch.load(path, weights_only=True)
        mutate(blob)
        torch.save(blob, path)
        meta_path = checkpoint / "checkpoint_meta.json"
        meta = json.loads(meta_path.read_text())
        meta["files"][name] = file_hash(path)
        meta_path.write_text(json.dumps(meta))

    def test_fresh_run_unique_and_identity_immutable(self):
        other = RunStore.fresh(Path(self.tmp.name), self.identity)
        self.assertNotEqual(self.store.run_id, other.run_id)
        self.assertEqual(self.store.manifest["identity_hash"], digest(self.identity))
        self.assertEqual(self.store.step, -1)

    def test_valid_resume_restores_adapter_optimizer_rng(self):
        self.commit(0)
        self.commit(1)
        before_tensor = self.model.lora_weight.detach().clone()
        expected = (random.random(), torch.rand(3))
        with torch.no_grad():
            self.model.lora_weight.add_(100)
        self.optimizer.state.clear()
        random.seed(42)
        torch.manual_seed(42)
        self.store.inspect()
        self.assertEqual(self.store.restore(self.model, self.optimizer, self.validate_adapter, self.load_adapter), 1)
        torch.testing.assert_close(self.model.lora_weight, before_tensor, rtol=0, atol=0)
        self.assertEqual(self.optimizer.state[self.model.lora_weight]["step"].item(), 1)
        self.assertEqual(random.random(), expected[0])
        torch.testing.assert_close(torch.rand(3), expected[1], rtol=0, atol=0)

    def test_missing_checkpoint_preserves_metrics(self):
        journal = self.store.path / "train_metrics_append.jsonl"
        journal.write_text('{"step":1}\n')
        before = journal.read_bytes()
        with self.assertRaisesRegex(ValueError, "missing initial checkpoint"):
            self.store.inspect(recover_metrics=True)
        self.assertEqual(journal.read_bytes(), before)

    def test_corrupted_checkpoint_metadata(self):
        self.commit(0)
        (self.store.checkpoints[0] / "checkpoint_meta.json").write_text("{")
        with self.assertRaises(ValueError):
            self.store.inspect()

    def test_wrong_revision_dataset_and_coefficient(self):
        for key, value in (("revision", "different"), ("dataset_sha256", "different"), ("mu_es", 1.0)):
            identity = deepcopy(self.identity)
            identity[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "identity mismatch"):
                RunStore(self.store.path, identity)

    def test_missing_optimizer_or_rng_component(self):
        self.commit(0)
        for name in ("optimizer.pt", "rng.pt"):
            path = self.store.checkpoints[0] / name
            saved = path.read_bytes()
            path.unlink()  # Owned synthetic fixture, never historical data.
            with self.assertRaises(ValueError):
                self.store.inspect()
            path.write_bytes(saved)

    def test_wrong_optimizer_state_and_missing_rng_keys(self):
        self.commit(0)
        self.commit(1)
        self.rewrite_component("optimizer.pt", lambda blob: blob["state"]["state"].clear())
        self.store.inspect()
        with self.assertRaisesRegex(ValueError, "optimizer state"):
            self.store.restore(self.model, self.optimizer, self.validate_adapter, self.load_adapter)

    def test_wrong_optimizer_step(self):
        self.commit(0)
        self.commit(1)
        self.rewrite_component("optimizer.pt", lambda blob: next(iter(blob["state"]["state"].values())).update(step=torch.tensor(99.)))
        self.store.inspect()
        with self.assertRaisesRegex(ValueError, "optimizer step"):
            self.store.restore(self.model, self.optimizer, self.validate_adapter, self.load_adapter)

    def test_missing_rng_state(self):
        self.commit(0)
        self.rewrite_component("rng.pt", lambda blob: blob.pop("torch_cpu"))
        self.store.inspect()
        with self.assertRaisesRegex(ValueError, "RNG"):
            self.store.restore(self.model, self.optimizer, self.validate_adapter, self.load_adapter)

    def test_duplicate_and_mixed_metrics_rejected(self):
        self.commit(0)
        self.commit(1)
        journal = self.store.path / "train_metrics_append.jsonl"
        with journal.open("ab") as stream:
            stream.write(journal.read_bytes())
        before = journal.read_bytes()
        with self.assertRaisesRegex(ValueError, "duplicate, mixed"):
            self.store.inspect(recover_metrics=True)
        self.assertEqual(journal.read_bytes(), before)

    def test_duplicate_commit_preserves_history(self):
        self.commit(0)
        self.commit(1)
        checkpoint = self.store.checkpoints[-1] / "checkpoint_meta.json"
        before = checkpoint.read_bytes()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.commit(1)
        self.assertEqual(checkpoint.read_bytes(), before)

    def test_interrupted_checkpoint_is_not_committed_and_is_preserved(self):
        self.commit(0)
        self.optimizer.state[self.model.lora_weight] = {"step": torch.tensor(1.),
            "exp_avg": torch.zeros_like(self.model.lora_weight), "exp_avg_sq": torch.ones_like(self.model.lora_weight)}
        def broken(path):
            (path / "partial.txt").write_text("SYNTHETIC DEMO")
            raise OSError("injected save interruption")
        with self.assertRaises(OSError):
            self.store.commit(1, {"step": 1, "sample_id": "demo-one"}, self.model, self.optimizer, broken)
        self.assertEqual(self.store.inspect()["step"], 0)
        self.assertFalse((self.store.path / "checkpoints/step_0001").exists())
        self.assertTrue(list((self.store.path / "checkpoints").glob(".incomplete-*")))

    def test_publish_then_append_failure_explicit_recovery(self):
        self.commit(0)
        import src.training.run_state as module
        real_append = module.append_bytes
        def broken_append(path, content):
            if Path(path).name == "train_metrics_append.jsonl":
                real_append(path, content[:15])
                raise OSError("injected torn journal")
            return real_append(path, content)
        with patch.object(module, "append_bytes", side_effect=broken_append), self.assertRaises(OSError):
            self.commit(1)
        journal = self.store.path / "train_metrics_append.jsonl"
        prefix = journal.read_bytes()
        with self.assertRaisesRegex(ValueError, "recover-metrics"):
            self.store.inspect()
        self.assertEqual(self.store.inspect(recover_metrics=True)["step"], 1)
        self.assertTrue(journal.read_bytes().startswith(prefix))
        self.assertEqual(len(journal.read_text().splitlines()), 1)
        self.store.inspect()  # No duplicate append on subsequent resume.

    def test_writer_lock_excludes_second_writer(self):
        with self.assertRaisesRegex(RuntimeError, "writer"):
            with RunStore(self.store.path, self.identity):
                pass

    def test_historical_checkpoint_bytes_unchanged(self):
        self.commit(0)
        history = {p: file_hash(p) for p in self.store.checkpoints[0].iterdir()}
        self.commit(1)
        self.store.inspect()
        self.assertEqual({p: file_hash(p) for p in history}, history)

    def test_corrupted_adapter_hash_is_rejected(self):
        self.commit(0)
        path = self.store.checkpoints[0] / "adapter_model.safetensors"
        path.write_bytes(path.read_bytes() + b"corrupt SYNTHETIC DEMO")
        with self.assertRaisesRegex(ValueError, "corrupted checkpoint"):
            self.store.inspect()

    def test_mixed_identity_metric_rejected_without_changes(self):
        self.commit(0)
        self.commit(1)
        path = self.store.path / "train_metrics_append.jsonl"
        row = json.loads(path.read_text())
        row["run_id"] = "another-run"
        path.write_text(json.dumps(row) + "\n")
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "mixed"):
            self.store.inspect(recover_metrics=True)
        self.assertEqual(path.read_bytes(), before)

    def test_duplicate_metric_steps_rejected_by_runner(self):
        runner = runner_module()
        path = Path(self.tmp.name) / "duplicate.jsonl"
        path.write_text('{"step":1}\n{"step":1}\n')
        with self.assertRaises(ValueError):
            runner.completed_steps(path)


@unittest.skipIf(torch is None, "optional CPU torch unavailable")
class TestObjective(unittest.TestCase):
    is_synthetic_demo = True

    def test_training_parameter_acceptance_detects_updates_and_frozen_changes(self):
        from src.training.scorer_verification import ParameterAudit, assess_training_change
        model = torch.nn.Module()
        model.register_parameter("lora_weight", torch.nn.Parameter(torch.tensor([1.])))
        model.register_parameter("frozen", torch.nn.Parameter(torch.tensor([2.]), requires_grad=False))
        with tempfile.TemporaryDirectory() as tmp, ParameterAudit(Path(tmp)) as audit:
            audit.capture(model)
            self.assertFalse(assess_training_change(audit.compare(model))["passed"])
            with torch.no_grad():
                model.lora_weight.add_(0.1)  # SYNTHETIC DEMO mutation, no optimizer step.
            self.assertTrue(assess_training_change(audit.compare(model))["passed"])
            with torch.no_grad():
                model.frozen.add_(0.1)
            self.assertFalse(assess_training_change(audit.compare(model))["passed"])

    def test_nonfinite_adapter_update_is_rejected(self):
        from src.training.scorer_verification import ParameterAudit, assess_training_change
        model = torch.nn.Module()
        model.register_parameter("lora_weight", torch.nn.Parameter(torch.tensor([1.])))
        model.register_parameter("frozen", torch.nn.Parameter(torch.tensor([2.]), requires_grad=False))
        with tempfile.TemporaryDirectory() as tmp, ParameterAudit(Path(tmp)) as audit:
            audit.capture(model)
            with torch.no_grad():
                model.lora_weight.fill_(float("nan"))
            self.assertFalse(assess_training_change(audit.compare(model))["passed"])

    def test_differentiable_composition_no_ce_substitution(self):
        from src.training.phase_2i_b_protocol import compose_objective
        cfg = json.loads((REPO_ROOT / "configs/phase_2i_b/es_ablation_config_v1.json").read_text())["es_objective"]
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.zeros(16))
            def forward(self, input_ids, **kwargs):
                return type("Output", (), {"logits": self.logits.expand(1, input_ids.shape[1], -1)})()
        for label in ("Yes", "No", "Undetermined"):
            model = Tiny()
            ce = torch.tensor(2., requires_grad=True)
            loss, es, scores = compose_objective(ce, model, {"input_ids": torch.tensor([[1, 2]])},
                                                [[3], [4], [5, 6]], label, cfg)
            torch.testing.assert_close(loss, ce + 0.25 * es)
            expected = torch.autograd.grad(ce + 0.25 * es, (ce, model.logits), retain_graph=True)
            actual = torch.autograd.grad(loss, (ce, model.logits))
            torch.testing.assert_close(actual[0], torch.tensor(1.))
            torch.testing.assert_close(actual[1], expected[1])
            self.assertTrue(scores.requires_grad)
            if label == "Undetermined":
                self.assertGreater(actual[1].abs().sum().item(), 0)
                self.assertGreater(es.item(), 0)

    def test_es_is_not_identical_to_ce(self):
        from src.training.differentiable_scoring import evidence_sufficiency_from_class_scores
        # Identical gold abstain probability and CE; different best definite competitor.
        scores = torch.tensor([[.2, .4, .4], [.3, .3, .4]]).log()
        es = [evidence_sufficiency_from_class_scores(s[None], torch.tensor([2]), sufficiency=torch.tensor([0])) for s in scores]
        self.assertEqual(scores[0, 2], scores[1, 2])
        self.assertNotEqual(es[0].item(), es[1].item())


class TestProtocol(unittest.TestCase):
    is_synthetic_demo = True

    def test_frozen_objective_rejects_ce_only_and_reordered_candidates(self):
        es = json.loads((REPO_ROOT / "configs/phase_2i_b/es_ablation_config_v1.json").read_text())["es_objective"]
        for key, value in (("mu_es", 0), ("margin", 0), ("candidates", list(reversed(es["candidates"]))), ("length_normalized", True)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_objective({**es, key: value})

    def test_cpu_only_plan_never_loads_model_or_initializes_run(self):
        runner = runner_module()
        plan = {"identity": synthetic_identity(), "config": {"matched_to_2i_a": {"max_steps": 50}},
                "schedule": [None] * 50, "validation": [None] * 18}
        with patch.object(runner, "build_plan", return_value=plan), patch.object(runner, "load_qwen3vl2_nf4") as loader:
            with patch.object(runner.RunStore, "fresh") as fresh, redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(["--plan"]), 0)
                loader.assert_not_called()
                fresh.assert_not_called()

    def test_execution_requires_explicit_mode_and_flag(self):
        runner = runner_module()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            runner.main([])
        self.assertEqual(error.exception.code, 2)

    def test_holdout_rejected_before_model_or_image_access(self):
        runner = runner_module()
        # The record is a synthetic sentinel; no actual holdout sample is opened.
        for operation in (lambda: runner.eval_one(None, None, {"split": "test"}, abstention_prompt=False),
                          lambda: runner.build_prompt_only_inputs(None, {"split": "test"})):
            with self.assertRaises(ValueError):
                operation()


@unittest.skipIf(torch is None, "optional CPU torch/safetensors unavailable")
class TestActualPeftSerialization(unittest.TestCase):
    is_synthetic_demo = True

    def test_existing_adapter_is_restored_without_replacing_optimizer_parameters(self):
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            self.skipTest("optional PEFT unavailable")
        runner = runner_module()
        base = torch.nn.Sequential(torch.nn.Linear(3, 3))
        model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=["0"], bias="none"))
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=1e-4)
        before = [p.detach().clone() for p in parameters]
        with tempfile.TemporaryDirectory() as tmp, RunStore.fresh(Path(tmp), synthetic_identity()) as store:
            with patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("optimizer step forbidden")):
                store.commit(0, None, model, optimizer, lambda path: model.save_pretrained(
                    str(path), safe_serialization=True, save_embedding_layers=False))
                with torch.no_grad():
                    for p in parameters:
                        p.add_(7)
                store.inspect()
                store.restore(model, optimizer, *runner.adapter_restorers(model,
                    {"r": 2, "lora_alpha": 4, "bias": "none", "target_modules": ["0"]}))
        self.assertEqual([id(p) for p in optimizer.param_groups[0]["params"]], [id(p) for p in parameters])
        for p, expected in zip(parameters, before):
            torch.testing.assert_close(p, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
