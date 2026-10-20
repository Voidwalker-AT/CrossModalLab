"""CrossModalLab — Phase 0B Hardware and PyTorch Preflight Verification Script.

Performs bounded, non-destructive GPU allocation and PyTorch/CUDA verification
under strict RTX 2050 (4 GB VRAM ceiling) constraints.
"""

import sys
import os
import json
import time

def run_preflight():
    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "packages": {},
        "cuda_available": False,
        "device_info": {},
        "bounded_smoke_test": {}
    }

    # 1. Package Import & Version Check
    try:
        import torch
        results["packages"]["torch"] = torch.__version__
        results["packages"]["torch_cuda_version"] = torch.version.cuda
    except ImportError as e:
        results["packages"]["torch"] = f"ERROR: {e}"

    try:
        import torchvision
        results["packages"]["torchvision"] = torchvision.__version__
    except ImportError as e:
        results["packages"]["torchvision"] = f"ERROR: {e}"

    try:
        import transformers
        results["packages"]["transformers"] = transformers.__version__
    except ImportError as e:
        results["packages"]["transformers"] = f"ERROR: {e}"

    try:
        import accelerate
        results["packages"]["accelerate"] = accelerate.__version__
    except ImportError as e:
        results["packages"]["accelerate"] = f"ERROR: {e}"

    try:
        import jsonschema
        results["packages"]["jsonschema"] = jsonschema.__version__
    except ImportError as e:
        results["packages"]["jsonschema"] = f"ERROR: {e}"

    # 2. CUDA Hardware Introspection
    if "torch" in sys.modules and torch.cuda.is_available():
        results["cuda_available"] = True
        device_id = 0
        props = torch.cuda.get_device_properties(device_id)
        
        results["device_info"] = {
            "device_id": device_id,
            "device_name": props.name,
            "device_count": torch.cuda.device_count(),
            "compute_capability": f"{props.major}.{props.minor}",
            "total_vram_bytes": props.total_memory,
            "total_vram_mib": round(props.total_memory / (1024 * 1024), 2),
            "multi_processor_count": props.multi_processor_count,
            "is_bf16_supported": torch.cuda.is_bf16_supported()
        }

        # 3. Bounded CUDA Computation Smoke Test
        # Strictly bounded: small 1024x1024 float32 matrix mul (~4 MB footprint)
        torch.cuda.reset_peak_memory_stats(device_id)
        init_allocated = torch.cuda.memory_allocated(device_id)
        init_reserved = torch.cuda.memory_reserved(device_id)
        
        t0 = time.perf_counter()
        a = torch.randn(1024, 1024, device=f"cuda:{device_id}", dtype=torch.float32)
        b = torch.randn(1024, 1024, device=f"cuda:{device_id}", dtype=torch.float32)
        c = torch.matmul(a, b)
        torch.cuda.synchronize(device_id)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        peak_allocated = torch.cuda.max_memory_allocated(device_id)
        peak_reserved = torch.cuda.max_memory_reserved(device_id)
        
        del a, b, c
        torch.cuda.empty_cache()
        final_allocated = torch.cuda.memory_allocated(device_id)

        results["bounded_smoke_test"] = {
            "matrix_shape": [1024, 1024],
            "dtype": "float32",
            "execution_duration_ms": round(elapsed_ms, 3),
            "initial_allocated_mib": round(init_allocated / (1024 * 1024), 3),
            "initial_reserved_mib": round(init_reserved / (1024 * 1024), 3),
            "peak_allocated_mib": round(peak_allocated / (1024 * 1024), 3),
            "peak_reserved_mib": round(peak_reserved / (1024 * 1024), 3),
            "post_cleanup_allocated_mib": round(final_allocated / (1024 * 1024), 3),
            "status": "PASS" if peak_allocated > 0 else "FAIL"
        }
    else:
        results["cuda_available"] = False
        results["device_info"]["error"] = "CUDA not available in PyTorch"

    return results

if __name__ == "__main__":
    report = run_preflight()
    print(json.dumps(report, indent=2))
