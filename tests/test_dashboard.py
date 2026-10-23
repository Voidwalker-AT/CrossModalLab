"""Dashboard boundaries and real stored development evidence; no GPU imports."""
import json
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from dashboard.backend.artifacts import ArtifactStore, metrics, sha
from dashboard.backend.server import make_server


class TestDashboard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = ArtifactStore()
        cls.server = make_server(0, store=cls.store)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def get(self, path):
        with urlopen(self.url + path) as response:
            return response, response.read()

    def test_metrics_separate_semantic_and_exact_format(self):
        payload = self.store.workspace()
        values = {c["id"]: c["metrics"] for c in payload["conditions"]}
        self.assertAlmostEqual(values["es5"]["balanced_accuracy"], .5)
        self.assertAlmostEqual(values["ce50"]["balanced_accuracy"], 2/3)
        self.assertEqual(values["ce50"]["exact_compliance"], 0)
        self.assertEqual(values["ce50"]["invalid_output_rate"], 0)
        self.assertAlmostEqual(values["ce5"]["invalid_output_rate"], 1/3)
        self.assertEqual(values["es5"]["confusion_matrix"]["Undetermined"]["Undetermined"], 2)

    def test_only_real_validation_cases_and_dual_oracle(self):
        payload = self.store.workspace()
        self.assertEqual(len(payload["cases"]), 18)
        for case in payload["cases"]:
            self.assertEqual(case["split"], "validation")
            self.assertIs(case["is_synthetic_demo"], False)
            self.assertTrue(case["solver"]["algorithm_agreement"])
            self.assertEqual(sha(self.store.image(case["sample_id"])), case["image_sha256"])

    def test_synthetic_demo_cannot_enter_analytics(self):
        with self.assertRaisesRegex(ValueError, "synthetic demonstrations"):
            metrics([{"is_synthetic_demo": True}])  # SYNTHETIC DEMO rejection sentinel.

    def test_partner_image_fallback_without_original_dataset_cache(self):
        original = Path.is_file
        def without_cache(path):
            if "datasets/cross_modal_contrast_v2/images/" in path.as_posix():
                return False
            return original(path)
        sample = next(iter(self.store.cases))
        with patch.object(Path, "is_file", autospec=True, side_effect=without_cache):
            image = self.store.image(sample)
            self.assertIn("dashboard/assets/images/", image.as_posix())
            self.assertEqual(sha(image), self.store.cases[sample]["image_sha256"])

    def test_api_is_readonly_and_contained(self):
        for method in ("POST", "PUT", "PATCH", "DELETE"):
            with self.subTest(method=method), self.assertRaises(HTTPError) as error:
                urlopen(Request(self.url + "/api/workspace", method=method, data=b"{}"))
            self.assertEqual(error.exception.code, 405)
        for path in ("/api/image/not-allowlisted", "/api/image/..%2f..%2fAGENTS.md", "/..%2f..%2fAGENTS.md", "/api/train"):
            with self.subTest(path=path), self.assertRaises(HTTPError) as error:
                self.get(path)
            self.assertEqual(error.exception.code, 404)

    def test_host_restriction_and_headers(self):
        response, raw = self.get("/api/health")
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
        self.assertIn("frame-ancestors 'none'", response.headers["Content-Security-Policy"])
        self.assertFalse(json.loads(raw)["gpu_loaded"])
        with self.assertRaises(HTTPError) as error:
            urlopen(Request(self.url + "/api/health", headers={"Host": "unrelated.invalid"}))
        self.assertEqual(error.exception.code, 403)

    def test_workspace_repeats_without_mutating_evidence(self):
        before = dict(self.store.sources)
        response, raw = self.get("/api/workspace")
        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(raw)["completed_step"], 5)
        self.get("/api/workspace")
        self.assertTrue(all(sha(self.store.root / p) == h for p, h in before.items()))


if __name__ == "__main__":
    unittest.main()
