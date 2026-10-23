"""Archive integrity checks; no model, GPU or dependency installation."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from submission.verify_package import verify
from src.config.paths import get_temp_dir


class SubmissionIntegrityTests(unittest.TestCase):
    def setUp(self):
        scratch = get_temp_dir()
        scratch.mkdir(parents=True, exist_ok=True)
        self.scratch = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        (self.root / "readme.txt").write_bytes(b"recorded evidence")
        self.manifest = {"schema": "crossmodallab_submission_v1", "final_holdout_included": False,
                         "source_commit": "test-only", "files": {"readme.txt": {
                         "bytes": 17, "sha256": hashlib.sha256(b"recorded evidence").hexdigest()}}}
        self.save()

    def save(self):
        (self.root / "PACKAGE_MANIFEST.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_valid_payload_and_partner_status(self):
        self.assertEqual(verify(self.root)["partner_machine"], "PENDING")

    def test_tamper_or_missing_payload_rejected(self):
        (self.root / "readme.txt").write_bytes(b"altered")
        with self.assertRaisesRegex(ValueError, "integrity"):
            verify(self.root)
        (self.root / "readme.txt").unlink()
        with self.assertRaisesRegex(ValueError, "inventory"):
            verify(self.root)

    def test_unexpected_payload_rejected(self):
        (self.root / "extra.txt").write_bytes(b"unexpected")
        with self.assertRaisesRegex(ValueError, "inventory"):
            verify(self.root)

    def test_holdout_flag_rejected(self):
        self.manifest["final_holdout_included"] = True
        self.save()
        with self.assertRaisesRegex(ValueError, "unsafe"):
            verify(self.root)


if __name__ == "__main__":
    unittest.main()
