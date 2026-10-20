"""Tests for portable path and experiment configuration (CPU-only, no torch required)."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.config.experiment import get_experiment_config
from src.config.paths import (
    PRIMARY_MODEL_ID,
    PRIMARY_MODEL_REVISION,
    REPO_ROOT,
    apply_default_cache_environment,
    get_hf_home,
    get_smolvlm2_snapshot_dir,
)


class TestPaths(unittest.TestCase):
    def test_repo_root_is_stable(self) -> None:
        self.assertTrue((REPO_ROOT / "src").is_dir())
        self.assertTrue((REPO_ROOT / "configs" / "experiment_config.json").is_file())

    def test_env_override_hf_home(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"CROSSMODAL_HF_HOME": tmp}, clear=False):
                # Ensure HF_HOME does not win over CROSSMODAL_HF_HOME
                env = os.environ.copy()
                env["CROSSMODAL_HF_HOME"] = tmp
                env.pop("HF_HOME", None)
                with mock.patch.dict(os.environ, env, clear=True):
                    resolved = get_hf_home()
                    self.assertEqual(resolved, Path(tmp))

    def test_snapshot_resolution_uses_explicit_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = os.environ.copy()
            env["CROSSMODAL_MODEL_SNAPSHOT_DIR"] = tmp
            env.pop("HF_HOME", None)
            env.pop("CROSSMODAL_HF_HOME", None)
            with mock.patch.dict(os.environ, env, clear=True):
                self.assertEqual(get_smolvlm2_snapshot_dir(), Path(tmp))

    def test_apply_cache_env_does_not_overwrite_existing_hf_home(self) -> None:
        sentinel = "/tmp/custom-hf-should-not-be-clobbered"
        env = os.environ.copy()
        env["HF_HOME"] = sentinel
        env.pop("CROSSMODAL_HF_HOME", None)
        with mock.patch.dict(os.environ, env, clear=True):
            apply_default_cache_environment()
            self.assertEqual(os.environ["HF_HOME"], sentinel)

    def test_legacy_e_cache_not_forced_when_disabled(self) -> None:
        env = os.environ.copy()
        env["CROSSMODAL_LEGACY_E_CACHE"] = "0"
        env.pop("CROSSMODAL_HF_HOME", None)
        env.pop("HF_HOME", None)
        with mock.patch.dict(os.environ, env, clear=True):
            home = get_hf_home()
            self.assertFalse(str(home).startswith("E:/cache"))


class TestExperimentConfig(unittest.TestCase):
    def test_pinned_model_identity(self) -> None:
        cfg = get_experiment_config()
        self.assertEqual(cfg.model.model_id, PRIMARY_MODEL_ID)
        self.assertEqual(cfg.model.revision, PRIMARY_MODEL_REVISION)
        self.assertEqual(cfg.quantization.bnb_4bit_quant_type, "nf4")
        self.assertTrue(cfg.quantization.load_in_4bit)
        self.assertEqual(cfg.candidate_answer_strings, [" Yes", " No", " Undetermined"])

    def test_config_file_is_valid_json(self) -> None:
        path = REPO_ROOT / "configs" / "experiment_config.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["model"]["revision"], PRIMARY_MODEL_REVISION)


if __name__ == "__main__":
    unittest.main()
