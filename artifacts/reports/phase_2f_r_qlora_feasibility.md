# Phase 2F-R Report — Qwen QLoRA Single-Step Feasibility

**Hardware feasibility only. No scientific fine-tuning claim. Adapter discarded.**

---

## 1. PEFT compatibility
`peft 0.21.2` imports cleanly with restored stack:
- `import peft` / `LoraConfig` / `get_peft_model` / `prepare_model_for_kbit_training` — **all OK**
- torch **2.6.0+cu124**, transformers **5.20.1**, bnb **0.50.2**

No package installs or upgrades this phase.

## 2. Source-code fixes
- LoRA targets exclude vision/visual paths  
- `prepare_model_for_kbit_training` + `enable_input_require_grads`  
- `use_cache=False` during training  
- Default **r=2** (frozen config `configs/phase_2f/qlora_hw_config_v1.json`)

## 3. LoRA configuration
| Field | Value |
|---|---|
| r / α | **2 / 4** |
| Targets | `q_proj,k_proj,v_proj,o_proj` (language backbone) |
| Vision / projector | **Frozen** |
| Base quant | **NF4 preserved** |
| Batch / accum | 1 / 1 |
| Optimizer steps | **1 text + 1 multimodal** |
| Trainable params | **802,816** / 1,222,365,184 total |

## 4–6. Training outcomes

### Text-only (synthetic fixture)
| Metric | Value |
|---|---|
| Loss | **4.891** (finite) |
| Grad tensors | **224 / 224** |
| Grad L2 | 24.25 |
| Duration | 3.65 s (fwd 1.77 / bwd 1.42 / opt 0.15) |
| Adapter weights changed | **Yes (224/224)** |
| Frozen base sample | **Unchanged** |

### Multimodal (development image + text premise)
| Metric | Value |
|---|---|
| Image tokens | **144** (genuinely multimodal) |
| Seq len | 191 |
| Loss | **5.189** (finite) |
| Grad tensors | **224** |
| Duration | 3.50 s |
| Peak torch alloc (max) | **2626 MiB** |
| Peak nvidia-smi | **~3236 MiB** |

## 7. Peak GPU / host memory
- Text step max torch alloc **2228 MiB** · nvidia-smi **~3229 MiB**  
- Multimodal max torch alloc **2626 MiB** · nvidia-smi **~3236 MiB**  
- Host free during steps ≥ ~1.2 GB; no OOM  

## 8. Runtime
Load ~26 s · each optimizer step **~3.5 s**

## 9. CUDA / ULVO health
CUDA **True** on RTX 2050 · doctor **PASSED** · checkpoints intact · no env changes

## 10. Unit tests
`tests/test_phase_2f_qlora.py` — **5 OK** (config, invalid r, frozen hw spec)

## 11. Git
See push SHA.

## 12. Recommendation for fine-tuning
**CONDITIONAL GO** for a **short, monitored** QLoRA run (tiny synthetic/dev subset only) with:
- r=2, batch=1, gradient checkpointing, max steps ≤ small pre-registered N  
- Stop if torch allocated >3800 MiB or host free <1.5 GB  
- **NO-GO** for multi-epoch full contrast-v2 training without new memory measurements  

**Computational feasibility of one QLoRA step on RTX 2050 is ESTABLISHED.**

**Stopped after Phase 2F-R.**
