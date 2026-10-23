"""Immutable run identity, per-step commits, and explicit checkpoint recovery.

No optimizer steps. Checkpoints are published by same-filesystem directory
rename, then mirrored into an append-only metric journal. A crash in that
window requires explicit journal recovery from the committed checkpoints.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
from uuid import uuid4

SCHEMA = "phase_2i_b_run_v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def append_bytes(path, content):
    with Path(path).open("ab") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def write_once(path, payload):
    """Publish an immutable JSON file; retain incomplete files on interruption."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".incomplete-{path.name}-{uuid4().hex}")
    with temp.open("xb") as stream:
        stream.write(canonical(payload) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        raise FileExistsError(path)
    temp.rename(path)


def capture_rng():
    import torch
    state = {"python": random.getstate(), "torch_cpu": torch.get_rng_state(),
             "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [], "numpy": None}
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        n = np.random.get_state()
        state["numpy"] = (n[0], n[1].tolist(), n[2], n[3], n[4])
    return state


def validate_rng(state):
    import torch
    if not isinstance(state, dict) or set(state) != {"python", "torch_cpu", "cuda", "numpy"}:
        raise ValueError("missing or invalid RNG state")
    # Validate with private generators without consuming global RNG state.
    random.Random().setstate(state["python"])
    torch.Generator(device="cpu").set_state(state["torch_cpu"])
    if not isinstance(state["cuda"], list) or any(
        not torch.is_tensor(s) or s.dtype != torch.uint8 or s.ndim != 1 for s in state["cuda"]
    ):
        raise ValueError("invalid CUDA RNG state")
    if state["numpy"] is not None:
        import numpy as np
        n = state["numpy"]
        np.random.RandomState().set_state((n[0], np.asarray(n[1], dtype="uint32"), *n[2:]))


def restore_rng(state):
    import torch
    validate_rng(state)
    if ((state["cuda"] and not torch.cuda.is_initialized()) or
            (torch.cuda.is_initialized() and len(state["cuda"]) != torch.cuda.device_count())):
        raise ValueError("CUDA RNG topology mismatch")
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    if state["numpy"] is not None:
        import numpy as np
        n = state["numpy"]
        np.random.set_state((n[0], np.asarray(n[1], dtype="uint32"), *n[2:]))


@contextmanager
def isolated_rng():
    state = capture_rng()
    try:
        yield
    finally:
        restore_rng(state)


def optimizer_layout(model, optimizer):
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    selected = [p for g in optimizer.param_groups for p in g["params"]]
    if [id(p) for _, p in named] != [id(p) for p in selected] or not named:
        raise ValueError("optimizer must bind all and only trainable parameters in named order")
    return [{"name": n, "shape": list(p.shape), "dtype": str(p.dtype)} for n, p in named]


def validate_optimizer(blob, optimizer, layout, step):
    import torch
    if blob.get("class") != "AdamW" or type(optimizer).__name__ != "AdamW" or blob.get("layout") != layout:
        raise ValueError("optimizer class or model binding mismatch")
    saved, current = blob["state"], optimizer.state_dict()
    if len(saved["param_groups"]) != len(current["param_groups"]):
        raise ValueError("optimizer group mismatch")
    ids = []
    for a, b in zip(saved["param_groups"], current["param_groups"]):
        if {k: v for k, v in a.items() if k != "params"} != {k: v for k, v in b.items() if k != "params"}:
            raise ValueError("optimizer hyperparameters mismatch")
        if a["params"] != b["params"]:
            raise ValueError("optimizer parameter count mismatch")
        ids.extend(a["params"])
    if len(set(ids)) != len(layout) or set(saved["state"]) != (set(ids) if step else set()):
        raise ValueError("missing, duplicate, or unexpected optimizer state")
    for pid, parameter in zip(ids, layout):
        if step == 0:
            continue
        slot = saved["state"][pid]
        if set(slot) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("incomplete AdamW moments")
        count = slot["step"]
        if (not torch.is_tensor(count) or count.ndim != 0 or not count.is_floating_point()
                or not torch.isfinite(count).all() or count.item() != step):
            raise ValueError("optimizer step does not match checkpoint step")
        for key in ("exp_avg", "exp_avg_sq"):
            value = slot[key]
            if (not torch.is_tensor(value) or list(value.shape) != parameter["shape"]
                    or str(value.dtype) != parameter["dtype"] or not torch.isfinite(value).all()):
                raise ValueError("optimizer moment shape/dtype/value mismatch")
        if (slot["exp_avg_sq"] < 0).any():
            raise ValueError("negative AdamW second moment")


class RunStore:
    """Single-writer store. OS locks release on process death, without deleting history."""
    def __init__(self, path, identity):
        self.path = Path(path).resolve()
        self.manifest = json.loads((self.path / "run.json").read_text(encoding="utf-8"))
        if (self.manifest.get("schema") != SCHEMA or self.manifest.get("identity") != identity
                or self.manifest.get("identity_hash") != digest(identity)):
            raise ValueError("experiment identity mismatch")
        if not re.fullmatch(r"[0-9a-f]{32}", self.manifest.get("run_id", "")):
            raise ValueError("invalid run identity")
        self.run_id = self.manifest["run_id"]
        self.identity_hash = self.manifest["identity_hash"]
        self.rows, self.checkpoints, self.lock = [], [], None

    @classmethod
    def fresh(cls, parent, identity):
        parent = Path(parent).resolve()
        parent.mkdir(parents=True, exist_ok=True)
        run_id = uuid4().hex
        path = parent / run_id
        path.mkdir(exist_ok=False)
        (path / "checkpoints").mkdir()
        write_once(path / "run.json", {"schema": SCHEMA, "run_id": run_id, "identity": identity,
            "identity_hash": digest(identity), "initial_status": "PREPARED",
            "created_utc": datetime.now(timezone.utc).isoformat()})
        return cls(path, identity)

    def __enter__(self):
        lockpath = self.path / ".run.lock"
        try:
            with lockpath.open("xb") as stream:
                stream.write(b"0")
        except FileExistsError:
            pass
        self.lock = lockpath.open("r+b")
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except Exception:
            self.lock.close()
            self.lock = None
            raise RuntimeError("run already has a writer") from None
        return self

    def __exit__(self, *_):
        if self.lock is not None:
            self.lock.close()  # Releases the operating-system lock on both platforms.
            self.lock = None

    def _locked(self):
        if self.lock is None:
            raise RuntimeError("run lock required")

    def event(self, status, **details):
        self._locked()
        now, event_id = datetime.now(timezone.utc), uuid4().hex
        write_once(self.path / "events" / f"{now.strftime('%Y%m%dT%H%M%S%f')}-{event_id}.json",
                   {"run_id": self.run_id, "event_id": event_id, "utc": now.isoformat(),
                    "status": status, **details})

    @property
    def step(self):
        return len(self.checkpoints) - 1

    def _expected_metric(self, step, row):
        if step == 0:
            if row is not None:
                raise ValueError("step zero cannot have training metrics")
            return None
        if not isinstance(row, dict) or row.get("step") != step:
            raise ValueError("metric step mismatch")
        sample = self.manifest["identity"]["schedule"][step - 1]
        if row.get("sample_id") != sample["sample_id"]:
            raise ValueError("metric schedule mismatch")
        if "ground_truth_label" in sample and row.get("label") != sample["ground_truth_label"]:
            raise ValueError("metric label mismatch")
        if row.get("run_id", self.run_id) != self.run_id or row.get("identity_hash", self.identity_hash) != self.identity_hash:
            raise ValueError("mixed metric identity")
        return {**row, "run_id": self.run_id, "identity_hash": self.identity_hash}

    def inspect(self, *, recover_metrics=False):
        """Validate committed bytes before any model load. Never infer state from metrics."""
        self._locked()
        root = self.path / "checkpoints"
        committed = sorted(p for p in root.iterdir() if p.name.startswith("step_"))
        if not committed:
            raise ValueError("missing initial checkpoint; explicit new run required")
        rows, checkpoints = [], []
        previous = None
        for step, directory in enumerate(committed):
            if directory.name != f"step_{step:04d}" or directory.is_symlink():
                raise ValueError("missing, invalid, or out-of-order checkpoint")
            meta = json.loads((directory / "checkpoint_meta.json").read_text(encoding="utf-8"))
            if (meta.get("schema") != SCHEMA or meta.get("step") != step or meta.get("run_id") != self.run_id
                    or meta.get("identity_hash") != self.identity_hash or meta.get("previous_meta_hash") != previous
                    or meta.get("scheduler") is not None):
                raise ValueError("corrupted checkpoint metadata or identity")
            files = meta.get("files", {})
            required = {"adapter_config.json", "adapter_model.safetensors", "optimizer.pt", "rng.pt"}
            if set(files) != required:
                raise ValueError("missing checkpoint component")
            for name, sha in files.items():
                file = directory / name
                if file.is_symlink() or not file.is_file() or file_hash(file) != sha:
                    raise ValueError(f"corrupted checkpoint component: {name}")
            row = self._expected_metric(step, meta.get("metric"))
            if row is not None:
                rows.append(row)
            if meta.get("metric_prefix_hash") != digest(rows):
                raise ValueError("checkpoint metric prefix mismatch")
            checkpoints.append(directory)
            previous = file_hash(directory / "checkpoint_meta.json")
        expected = b"".join(canonical(r) + b"\n" for r in rows)
        journal = self.path / "train_metrics_append.jsonl"
        observed = journal.read_bytes() if journal.exists() else b""
        if observed != expected:
            # Covers a missing append and a torn append without discarding a byte.
            if not expected.startswith(observed):
                raise ValueError("duplicate, mixed, or corrupted metric journal")
            if not recover_metrics:
                raise ValueError("incomplete metric journal; explicit --recover-metrics required")
            append_bytes(journal, expected[len(observed):])
            self.event("METRICS_RECOVERED", appended_bytes=len(expected) - len(observed))
        self.rows, self.checkpoints = rows, checkpoints
        return {"step": self.step, "incomplete_checkpoints": sorted(p.name for p in root.glob(".incomplete-*"))}

    def commit(self, step, row, model, optimizer, save_adapter):
        import torch
        self._locked()
        if step != self.step + 1 or not 0 <= step <= len(self.manifest["identity"]["schedule"]):
            raise ValueError("duplicate or out-of-order checkpoint step")
        metric = self._expected_metric(step, row)
        journal = self.path / "train_metrics_append.jsonl"
        expected = b"".join(canonical(r) + b"\n" for r in self.rows)
        if (journal.read_bytes() if journal.exists() else b"") != expected:
            raise ValueError("metric journal changed; refusing commit")
        final = self.path / "checkpoints" / f"step_{step:04d}"
        if final.exists():
            raise FileExistsError("checkpoint already committed; resume explicitly")
        staging = final.with_name(f".incomplete-{final.name}-{uuid4().hex}")
        staging.mkdir(exist_ok=False)
        layout = optimizer_layout(model, optimizer)
        blob = {"class": type(optimizer).__name__, "layout": layout, "state": optimizer.state_dict()}
        validate_optimizer(blob, optimizer, layout, step)
        save_adapter(staging)
        torch.save(blob, staging / "optimizer.pt")
        rng = capture_rng()
        validate_rng(rng)
        torch.save(rng, staging / "rng.pt")
        files = {}
        for name in ("adapter_config.json", "adapter_model.safetensors", "optimizer.pt", "rng.pt"):
            path = staging / name
            if path.is_symlink() or not path.is_file() or not path.stat().st_size:
                raise ValueError(f"missing saved checkpoint component: {name}")
            # Windows FlushFileBuffers requires a writable descriptor.
            with path.open("r+b") as stream:
                os.fsync(stream.fileno())
            files[name] = file_hash(path)
        rows = self.rows + ([metric] if metric is not None else [])
        write_once(staging / "checkpoint_meta.json", {"schema": SCHEMA, "run_id": self.run_id,
            "identity_hash": self.identity_hash, "step": step, "metric": metric, "files": files,
            "scheduler": None, "metric_prefix_hash": digest(rows),
            "previous_meta_hash": file_hash(self.checkpoints[-1] / "checkpoint_meta.json") if self.checkpoints else None})
        staging.rename(final)
        self.rows, self.checkpoints = rows, self.checkpoints + [final]
        # If this append fails, the checkpoint remains authoritative, with its row.
        if metric is not None:
            append_bytes(journal, canonical(metric) + b"\n")

    def restore(self, model, optimizer, validate_adapter, load_adapter):
        import torch
        self._locked()
        if self.step < 0:
            raise ValueError("inspect checkpoints before restoration")
        checkpoint = self.checkpoints[-1]
        blob = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
        rng = torch.load(checkpoint / "rng.pt", map_location="cpu", weights_only=True)
        validate_optimizer(blob, optimizer, optimizer_layout(model, optimizer), self.step)
        validate_rng(rng)
        if ((rng["cuda"] and not torch.cuda.is_initialized()) or
                (torch.cuda.is_initialized() and len(rng["cuda"]) != torch.cuda.device_count())):
            raise ValueError("CUDA RNG topology mismatch")
        validate_adapter(checkpoint)  # All component checks precede model mutation.
        load_adapter(checkpoint)
        optimizer.load_state_dict(blob["state"])
        restore_rng(rng)  # Last: no RNG-consuming work between this and next step.
        return self.step
