"""Bounded parameter comparisons and failure-safe telemetry for scorer checks.

No optimizers or training operations. Disk snapshots are ephemeral and private
to one audit. Rounding is accepted only for an explicit name/dtype policy and
exact equality to the expected cast, never a universal numerical tolerance.
"""
from __future__ import annotations

from collections import Counter
import gc
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
import traceback

import torch


class ParameterAudit:
    def __init__(self, parent: Path, *, chunk_elements=262144, allowed_rounding=None):
        if chunk_elements <= 0:
            raise ValueError("chunk_elements must be positive")
        self.parent = Path(parent).resolve()
        self.parent.mkdir(parents=True, exist_ok=True)
        self._temp = tempfile.TemporaryDirectory(prefix="parameter-audit-", dir=self.parent)
        self.directory = Path(self._temp.name).resolve()
        self.chunk_elements = chunk_elements
        self.allowed_rounding = allowed_rounding or {}
        self.records = {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        # Only delete the exclusive temporary tree created by this audit.
        if self.directory.parent != self.parent or not self.directory.name.startswith("parameter-audit-"):
            raise RuntimeError("snapshot cleanup boundary violation")
        self._temp.cleanup()

    def _chunks(self, tensor):
        if tensor.layout != torch.strided or not tensor.is_contiguous():
            raise ValueError("noncontiguous or unsupported parameter storage")
        flat = tensor.detach().view(-1)
        for start in range(0, flat.numel(), self.chunk_elements):
            yield flat[start:start + self.chunk_elements].cpu()

    @staticmethod
    def _bytes(chunk):
        return chunk.contiguous().view(torch.uint8).numpy().tobytes()

    def capture(self, model):
        if self.records:
            raise RuntimeError("baseline already captured")
        needed = sum(p.numel() * p.element_size() for p in model.parameters())
        if shutil.disk_usage(self.directory).free < needed + 512 * 1024**2:
            raise RuntimeError("insufficient disk for bounded parameter snapshots")
        for index, (name, parameter) in enumerate(model.named_parameters()):
            path = self.directory / f"{index}.raw"
            digest = hashlib.sha256()
            finite = True
            with path.open("xb") as stream:
                for chunk in self._chunks(parameter):
                    finite = finite and bool(torch.isfinite(chunk).all())
                    raw = self._bytes(chunk)
                    stream.write(raw)
                    digest.update(raw)
            self.records[name] = {"path": path, "dtype": parameter.dtype,
                                  "shape": list(parameter.shape), "requires_grad": parameter.requires_grad,
                                  "raw_hash": digest.hexdigest(), "finite": finite}

    def compare(self, model):
        parameters = dict(model.named_parameters())
        rows = []
        for name in self.records.keys() | parameters.keys():
            old = self.records.get(name)
            current = parameters.get(name)
            row = {"name": name, "classification": "UNVERIFIABLE", "passed": False,
                   "original_dtype": str(old["dtype"]) if old else None,
                   "final_dtype": str(current.dtype) if current is not None else None,
                   "shape": old["shape"] if old else None, "raw_byte_hash_changed": None,
                   "numerically_equal": None, "max_abs_difference": None,
                   "max_relative_difference": None, "finite": None,
                   "unexpected_frozen_gradient": False}
            rows.append(row)
            if old is None or current is None or list(current.shape) != old["shape"]:
                row["reason"] = "parameter set or shape changed"
                continue
            row["requires_grad_before"] = old["requires_grad"]
            row["requires_grad_after"] = current.requires_grad
            row["unexpected_frozen_gradient"] = not old["requires_grad"] and current.grad is not None
            same_dtype = current.dtype == old["dtype"]
            floating = current.is_floating_point() and old["dtype"].is_floating_point
            if current.is_complex() or (not same_dtype and not floating):
                row["reason"] = "unsupported precision transition"
                continue
            digest = hashlib.sha256()
            equal, projected_equal, finite = True, True, old["finite"]
            max_abs = max_rel = 0.0
            zero_reference_changed = False
            try:
                with old["path"].open("rb") as stream:
                    for chunk in self._chunks(current):
                        raw = stream.read(chunk.numel() * torch.empty((), dtype=old["dtype"]).element_size())
                        before = torch.frombuffer(bytearray(raw), dtype=old["dtype"])
                        if before.numel() != chunk.numel():
                            raise ValueError("truncated snapshot")
                        digest.update(self._bytes(chunk))
                        finite = finite and bool(torch.isfinite(chunk).all())
                        # Float64 exactly represents FP16/BF16/FP32 values. Keep
                        # integer equality native to avoid losing int64 low bits.
                        a, b = before.to(torch.float64), chunk.to(torch.float64)
                        chunk_equal = torch.equal(a, b) if floating else torch.equal(before, chunk)
                        equal = equal and chunk_equal
                        projected_equal = projected_equal and torch.equal(before.to(chunk.dtype), chunk)
                        if current.dtype == torch.int64 and not chunk_equal:
                            # Python integers avoid int64 subtraction overflow.
                            for left, right in zip(before.tolist(), chunk.tolist()):
                                difference = abs(left - right)
                                max_abs = max(max_abs, difference)
                                if left:
                                    max_rel = max(max_rel, difference / abs(left))
                                else:
                                    zero_reference_changed |= difference != 0
                            continue
                        delta = (a - b).abs()
                        if bool(torch.isfinite(delta).all()):
                            max_abs = max(max_abs, delta.max().item())
                            nonzero = a != 0
                            if bool(nonzero.any()):
                                max_rel = max(max_rel, (delta[nonzero] / a[nonzero].abs()).max().item())
                            zero_reference_changed |= bool(((a == 0) & (b != 0)).any())
                    if stream.read(1):
                        raise ValueError("snapshot size mismatch")
            except Exception as exc:
                row["reason"] = f"snapshot comparison failed: {exc}"
                continue
            row.update(raw_byte_hash_changed=digest.hexdigest() != old["raw_hash"],
                       raw_hash_before=old["raw_hash"], raw_hash_after=digest.hexdigest(),
                       numerically_equal=equal and finite, finite=finite,
                       max_abs_difference=max_abs if finite else None,
                       max_relative_difference=max_rel if finite else None,
                       relative_difference_definition="abs(after-before)/abs(before), nonzero references only",
                       common_comparison_precision="float64 for real floating tensors; exact integers otherwise",
                       zero_reference_changed=zero_reference_changed)
            permitted = self.allowed_rounding.get(name) == (old["dtype"], current.dtype)
            if not old["finite"]:
                row["reason"] = "nonfinite baseline"
            elif not finite:
                row["classification"] = "NUMERICALLY_CHANGED"
                row["reason"] = "nonfinite final value"
            elif equal:
                row["classification"] = ("EXACT_NUMERICALLY_UNCHANGED" if same_dtype else
                                         "DTYPE_CHANGED_NUMERICALLY_EQUIVALENT")
            elif not same_dtype and permitted and projected_equal:
                row["classification"] = "DTYPE_CHANGED_WITH_EXPECTED_ROUNDING"
                row["reason"] = "exact equality to explicitly permitted dtype cast; no slack tolerance"
            else:
                row["classification"] = "NUMERICALLY_CHANGED"
            row["explained_by_dtype_conversion"] = row["classification"] in {
                "DTYPE_CHANGED_NUMERICALLY_EQUIVALENT", "DTYPE_CHANGED_WITH_EXPECTED_ROUNDING"}
            row["passed"] = (row["classification"] in {
                "EXACT_NUMERICALLY_UNCHANGED", "DTYPE_CHANGED_NUMERICALLY_EQUIVALENT",
                "DTYPE_CHANGED_WITH_EXPECTED_ROUNDING"} and not row["unexpected_frozen_gradient"]
                and current.requires_grad == old["requires_grad"])
        rows.sort(key=lambda row: row["name"])
        return {"passed": bool(rows) and all(r["passed"] for r in rows), "parameters": rows,
                "summary": dict(Counter(r["classification"] for r in rows)),
                "comparison_policy": "exact numerical equality; explicit named dtype-cast projection only",
                "chunk_elements": self.chunk_elements}


def _host_commit():
    if os.name != "nt":
        return {"host_commit_bytes": None, "host_commit_note": "Windows API unavailable"}
    import ctypes
    from ctypes import wintypes
    class PerformanceInfo(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in (
            "CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal", "PhysicalAvailable",
            "SystemCache", "KernelTotal", "KernelPaged", "KernelNonpaged", "PageSize")
        ] + [(name, wintypes.DWORD) for name in ("HandleCount", "ProcessCount", "ThreadCount")]
    info = PerformanceInfo()
    info.cb = ctypes.sizeof(info)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(info), info.cb):
        raise OSError("GetPerformanceInfo failed")
    return {"host_commit_bytes": info.CommitTotal * info.PageSize,
            "host_commit_limit_bytes": info.CommitLimit * info.PageSize}


def sample_resources():
    from src.causal.telemetry import snapshot_memory
    sample = snapshot_memory().to_dict()
    try:
        sample.update(_host_commit())
    except Exception as exc:
        sample.update(host_commit_bytes=None, host_commit_note=str(exc))
    return sample


def run_observed(report, action, cleanup, *, sampler=None, interval=1.0):
    """Preserve the original exception and telemetry even if cleanup also fails."""
    sampler = sampler or sample_resources
    samples, sampling_errors = [], []
    stop = threading.Event()
    started = time.perf_counter()
    def sample():
        try:
            samples.append({"elapsed_sec": time.perf_counter() - started, **sampler()})
        except Exception as exc:
            sampling_errors.append(repr(exc))
    def monitor():
        while not stop.wait(interval):
            sample()
    sample()
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        action()
        report["status"] = "PASSED"
    except Exception as exc:
        report.update(status="FAILED", error_type=type(exc).__name__, error_message=str(exc),
                      traceback=traceback.format_exc())
    finally:
        stop.set()
        thread.join(timeout=5)
        if thread.is_alive():
            sampling_errors.append("monitor did not stop within 5 seconds")
        sample()  # Before cleanup: includes allocator peaks on the failure path.
        try:
            cleanup()
            gc.collect()
            report["cleanup_status"] = "PASSED"
        except Exception:
            report.update(status="FAILED", cleanup_status="FAILED", cleanup_traceback=traceback.format_exc())
        sample()
        def aggregate(key, operation):
            values = [s[key] for s in samples if s.get(key) is not None]
            return operation(values) if values else None
        report["telemetry"] = {
            "samples": samples, "sampling_errors": sampling_errors, "sampling_interval_sec": interval,
            "peak_torch_allocated_mib": aggregate("torch_max_allocated_mib", max),
            "peak_torch_reserved_mib": aggregate("torch_max_reserved_mib", max),
            "maximum_sampled_nvidia_used_mib": aggregate("nvidia_smi_used_mib", max),
            "minimum_sampled_host_available_mib": aggregate("host_available_mib", min),
            "maximum_sampled_host_commit_bytes": aggregate("host_commit_bytes", max),
            "nvidia_sampling_note": "sampled physical usage; not a measured peak"}
        if sampling_errors:
            report["status"] = "FAILED"
        report["elapsed_sec"] = time.perf_counter() - started
    return 0 if report["status"] == "PASSED" else 1
