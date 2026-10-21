"""Per-intervention GPU/host memory telemetry (CPU-safe schema + optional GPU read)."""

from __future__ import annotations

import os
import subprocess
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class MemorySnapshot:
    torch_allocated_mib: Optional[float]
    torch_reserved_mib: Optional[float]
    torch_max_allocated_mib: Optional[float]
    torch_max_reserved_mib: Optional[float]
    nvidia_smi_used_mib: Optional[float]
    nvidia_smi_total_mib: Optional[float]
    host_available_mib: Optional[float]
    warnings: list

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _mib(n_bytes: float) -> float:
    return round(n_bytes / (1024 * 1024), 3)


def reset_torch_peaks() -> None:
    """Documented boundary: call before each scored intervention forward."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(0)
            torch.cuda.synchronize(0)
    except Exception:
        pass


def snapshot_memory(*, dedicated_total_mib: float = 4096.0) -> MemorySnapshot:
    warnings = []
    torch_alloc = torch_res = torch_max_a = torch_max_r = None
    try:
        import torch

        if torch.cuda.is_available():
            torch_alloc = _mib(torch.cuda.memory_allocated(0))
            torch_res = _mib(torch.cuda.memory_reserved(0))
            torch_max_a = _mib(torch.cuda.max_memory_allocated(0))
            torch_max_r = _mib(torch.cuda.max_memory_reserved(0))
            if torch_max_a is not None and torch_max_a > dedicated_total_mib + 50:
                warnings.append(
                    "torch_max_allocated_exceeds_dedicated_vram;"
                    "likely_WDDM_shared_or_accounting_not_fabrication"
                )
    except Exception as exc:
        warnings.append(f"torch_memory_unavailable:{exc}")

    nvidia_used = nvidia_total = None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2,
        ).strip().splitlines()[0]
        used, total = [float(x.strip()) for x in out.split(",")]
        nvidia_used, nvidia_total = used, total
    except Exception:
        warnings.append("nvidia_smi_unavailable")

    host_free = None
    try:
        import psutil

        host_free = _mib(psutil.virtual_memory().available)
    except Exception:
        pass

    return MemorySnapshot(
        torch_allocated_mib=torch_alloc,
        torch_reserved_mib=torch_res,
        torch_max_allocated_mib=torch_max_a,
        torch_max_reserved_mib=torch_max_r,
        nvidia_smi_used_mib=nvidia_used,
        nvidia_smi_total_mib=nvidia_total,
        host_available_mib=host_free,
        warnings=warnings,
    )


def should_stop_on_memory(
    snap: MemorySnapshot,
    *,
    max_torch_allocated_mib: float = 3800.0,
    min_host_available_mib: float = 1500.0,
) -> Dict[str, Any]:
    """Stop conditions based on measured pressure (not speculative)."""
    stop = False
    reasons = []
    if snap.torch_allocated_mib is not None and snap.torch_allocated_mib > max_torch_allocated_mib:
        stop = True
        reasons.append(f"torch_allocated {snap.torch_allocated_mib} > {max_torch_allocated_mib}")
    if snap.host_available_mib is not None and snap.host_available_mib < min_host_available_mib:
        stop = True
        reasons.append(f"host_available {snap.host_available_mib} < {min_host_available_mib}")
    return {"stop": stop, "reasons": reasons, "snapshot": snap.to_dict()}
