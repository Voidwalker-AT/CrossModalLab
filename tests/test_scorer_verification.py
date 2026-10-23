"""SYNTHETIC DEMO fixtures, quarantined from empirical model results."""
import hashlib
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from src.training.scorer_verification import ParameterAudit, run_observed


@unittest.skipIf(torch is None, "optional torch unavailable")
class TestParameterAudit(unittest.TestCase):
    is_synthetic_demo = True

    def model(self, data, trainable=False):
        model = torch.nn.Module()
        model.register_parameter("bias", torch.nn.Parameter(data, requires_grad=trainable))
        return model

    def audit(self, before, after, *, allow=False):
        model = self.model(before)
        with tempfile.TemporaryDirectory() as parent:
            with ParameterAudit(Path(parent), chunk_elements=2,
                                allowed_rounding={"bias": (torch.float32, torch.bfloat16)} if allow else {}) as audit:
                audit.capture(model)
                model.bias.data = after
                result = audit.compare(model)
        return result, result["parameters"][0]

    def test_reversible_bfloat16_float32(self):
        before = torch.tensor([1.125, -0.25, 0.0], dtype=torch.bfloat16)
        result, row = self.audit(before, before.float())
        self.assertTrue(result["passed"])
        self.assertEqual(row["classification"], "DTYPE_CHANGED_NUMERICALLY_EQUIVALENT")
        self.assertTrue(row["raw_byte_hash_changed"])
        self.assertEqual(row["max_abs_difference"], 0)

    def test_expected_rounding_requires_explicit_policy(self):
        before = torch.tensor([1.001, -0.12345, 2.2])
        result, row = self.audit(before, before.bfloat16(), allow=True)
        self.assertTrue(result["passed"])
        self.assertEqual(row["classification"], "DTYPE_CHANGED_WITH_EXPECTED_ROUNDING")
        self.assertGreater(row["max_abs_difference"], 0)
        self.assertFalse(row["numerically_equal"])
        result, row = self.audit(before, before.bfloat16())
        self.assertFalse(result["passed"])
        self.assertEqual(row["classification"], "NUMERICALLY_CHANGED")

    def test_rounding_policy_does_not_hide_mutation(self):
        before = torch.tensor([1.001, -0.12345, 2.2])
        after = before.bfloat16()
        after[-1] += 1
        result, row = self.audit(before, after, allow=True)
        self.assertFalse(result["passed"])
        self.assertEqual(row["classification"], "NUMERICALLY_CHANGED")

    def test_same_dtype_mutation_even_tiny_fails(self):
        before = torch.ones(5)
        after = before.clone()
        after[-1] = torch.nextafter(after[-1], torch.tensor(2.0))
        result, row = self.audit(before, after)
        self.assertFalse(result["passed"])
        self.assertGreater(row["max_relative_difference"], 0)

    def test_nan_inf_fail(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            result, row = self.audit(torch.ones(3), torch.tensor([1., value, 1.]))
            self.assertFalse(result["passed"])
            self.assertFalse(row["finite"])

    def test_nonfinite_baseline_unverifiable(self):
        result, row = self.audit(torch.tensor([float("nan")]), torch.ones(1))
        self.assertFalse(result["passed"])
        self.assertEqual(row["classification"], "UNVERIFIABLE")

    def test_frozen_grad_fails_and_lora_grad_without_update_passes(self):
        for trainable in (False, True):
            model = self.model(torch.ones(4), trainable)
            with tempfile.TemporaryDirectory() as parent, ParameterAudit(Path(parent), chunk_elements=2) as audit:
                audit.capture(model)
                if trainable:
                    (model.bias.square().sum()).backward()
                else:
                    model.bias.grad = torch.ones_like(model.bias)
                result = audit.compare(model)
                self.assertEqual(result["passed"], trainable)
                self.assertEqual(result["parameters"][0]["unexpected_frozen_gradient"], not trainable)

    def test_integer_precision_is_not_lost(self):
        before = torch.tensor([2**60, 2**60 + 1], dtype=torch.int64)
        result, row = self.audit(before, before + 1)
        self.assertFalse(result["passed"])
        self.assertEqual(row["max_abs_difference"], 1)

    def test_removed_parameter_and_shape_change_fail(self):
        model = self.model(torch.ones(3))
        with tempfile.TemporaryDirectory() as parent, ParameterAudit(Path(parent)) as audit:
            audit.capture(model)
            model.bias.data = torch.ones(4)
            self.assertFalse(audit.compare(model)["passed"])
            del model.bias
            self.assertFalse(audit.compare(model)["passed"])

    def test_cleanup_and_historical_artifact_immutability(self):
        with tempfile.TemporaryDirectory() as parent:
            root = Path(parent)
            history = root / "historical.json"
            history.write_text('{"is_synthetic_demo": true, "watermark": "SYNTHETIC DEMO"}')
            digest = hashlib.sha256(history.read_bytes()).hexdigest()
            with self.assertRaisesRegex(RuntimeError, "injected"):
                with ParameterAudit(root, chunk_elements=2) as audit:
                    audit.capture(self.model(torch.ones(9)))
                    directory = audit.directory
                    self.assertGreater(len(list(directory.iterdir())), 0)
                    raise RuntimeError("injected")
            self.assertFalse(directory.exists())
            self.assertEqual(hashlib.sha256(history.read_bytes()).hexdigest(), digest)


@unittest.skipIf(torch is None, "optional torch unavailable")
class TestFailureTelemetry(unittest.TestCase):
    is_synthetic_demo = True

    def test_assertion_records_telemetry_and_cleanup(self):
        report, cleaned = {}, []
        def fail():
            raise AssertionError("injected failure")
        def sample():
            return {"host_available_mib": 2048., "nvidia_smi_used_mib": 100.,
                    "torch_max_allocated_mib": 12., "torch_max_reserved_mib": 16.}
        code = run_observed(report, fail, lambda: cleaned.append(True), sampler=sample)
        self.assertEqual(code, 1)
        self.assertEqual(report["error_type"], "AssertionError")
        self.assertEqual(cleaned, [True])
        self.assertEqual(report["cleanup_status"], "PASSED")
        self.assertEqual(report["telemetry"]["peak_torch_allocated_mib"], 12.)
        self.assertEqual(report["telemetry"]["minimum_sampled_host_available_mib"], 2048.)
        self.assertGreaterEqual(len(report["telemetry"]["samples"]), 3)

    def test_cleanup_and_sampler_errors_do_not_mask_original(self):
        def fail():
            raise ValueError("original")
        def cleanup():
            raise RuntimeError("cleanup")
        def sample():
            raise RuntimeError("sampler")
        report = {}
        self.assertEqual(run_observed(report, fail, cleanup, sampler=sample), 1)
        self.assertEqual(report["error_type"], "ValueError")
        self.assertEqual(report["cleanup_status"], "FAILED")
        self.assertTrue(report["telemetry"]["sampling_errors"])


if __name__ == "__main__":
    unittest.main()
