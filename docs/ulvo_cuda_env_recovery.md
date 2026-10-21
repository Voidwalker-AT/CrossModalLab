# ULVO CUDA Environment Recovery — 2026-10-21

**Mode:** Environment recovery only. **Phase 2F training not resumed.**

## A. Root cause
`uv pip install -U peft` inside the ULVO vault environment resolved an unbounded upgrade set and replaced:

- `torch 2.6.0+cu124` → `torch 2.14.1+cpu` (CUDA **disabled**)
- `transformers 5.20.1` → `5.20.2` (and transient torchvision/torch mismatch)
- `sympy` → `1.14.0` (off lock)

PEFT itself is **not** required for recovery; the damage was the unbounded torch upgrade.

## B. Environment path
`E:\Workspace\Active\ULVO\vault\environments\python\env_crossmodallab_260321b07610`

## C. Restoration method
1. No concurrent installs; no ULVO env delete.  
2. ULVO has **no** dedicated “restore from lock” command; connector already points at `requirements.lock`.  
3. Downloaded **cached-source** CUDA wheel via PyTorch index:  
   `torch-2.6.0+cu124-cp311-cp311-win_amd64.whl` (~2.53 GB) → `E:\cache\wheels\`  
4. `uv pip install --no-deps` **local wheel** (avoided CPU substitution).  
5. Pinned `sympy==1.13.1` and `transformers==5.20.1` with `--no-deps`.  
6. Did **not** re-run `-U peft` or full lock reinstall (timeouts).

## D. Packages changed (final)

| Package | Before | After |
|---|---|---|
| torch | 2.14.1+cpu | **2.6.0+cu124** |
| torchvision | 0.21.0+cu124 | 0.21.0+cu124 (unchanged) |
| transformers | 5.20.2 | **5.20.1** |
| sympy | 1.14.0 | **1.13.1** |
| bitsandbytes | 0.50.2 | 0.50.2 |
| peft | 0.21.2 | 0.21.2 (kept; not verified for training this recovery) |

## E–F. Verification
```
torch 2.6.0+cu124
cuda True
device NVIDIA GeForce RTX 2050
capability (8, 6)
matmul_ok (64, 64)
```
`ulvo project doctor` PASSED · portability PASSED · **183 tests OK**  
Checkpoints: SmolVLM2-2.2B **True**, Qwen3-VL-2B **True**

## G. ULVO health
Doctor PASSED; env RECLAIMABLE/READY; no process leases during recovery.

## H. Remaining risks
- Full `requirements.lock` reinstall still long (CUDA wheel large); prefer **local wheel + --no-deps pins**.  
- peft 0.21.2 present but **Phase 2F training remains halted** until re-approved.  
- Future `-U` on peft/torch can re-break CUDA.

## I. Safe PEFT strategy (future, authorized only)
1. Never `uv pip install -U peft` bare.  
2. Dry-run first; freeze torch/torchvision versions.  
3. Prefer `uv pip install peft==<ver> --no-deps` then add only missing deps.  
4. Re-check `torch.cuda.is_available()` after every install.  
5. Document plan in a resolution note before execution.

**Do not auto-resume QLoRA training.**

---

## Storage cleanup (same session, after CUDA verification)

| Item | Action | Reclaim |
|---|---|---:|
| `E:\cache\wheels\torch-2.6.0+cu124-...whl` | Removed (torch already installed & verified) | **2,532,350,702 bytes (~2.42 GB)** on E: |
| uv cache (`C:\...\uv\cache`) | `uv cache prune` (targeted, not full wipe) | **~20.4 GiB** on C: |
| ULVO vault / environments | **Not deleted** (active authority) | — |
| HF SmolVLM2-2.2B / Qwen3-VL-2B | **Preserved** | — |
| Scientific datasets/artifacts | **Preserved** | — |

**Free space after cleanup:** C: **~25.5 GB** · E: **~36.0 GB**  
(Pre-cleanup measured: C: 5.36 GB · E: 33.65 GB)

`ulvo cache prune --dry-run` was inspected first; vault packages not manually deleted.
