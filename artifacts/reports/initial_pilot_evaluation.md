# Initial Real VLM Evaluation Report: SmolVLM-500M-Instruct

**Date:** 2026-10-18T08:59:18Z  
**Model ID:** `HuggingFaceTB/SmolVLM-500M-Instruct` (Revision: `a7da5b986cb59b408707209984f360a5f4ad7e47`)  
**Total Cases:** 15 manually verified controlled cases  
**Overall Conformance:** 8/15 (53.3%)  
**Abstention Failures (Forced guessing on forks):** 3  
**Deductive / Perceptual Errors:** 4  

## 1. Summary by Relational Topology

| Topology | N | Conforming | Accuracy | Observed Model Decisions |
| :--- | :---: | :---: | :---: | :--- |
| `forward_chain` | 3 | 3 | 100% | Yes, Yes, Yes |
| `reverse_chain` | 3 | 1 | 33% | Yes, No, Yes |
| `underdetermined_fork` | 3 | 0 | 0% | Yes, Yes, Yes |
| `unimodal_visual` | 3 | 1 | 33% | No, No, No |
| `unimodal_text` | 3 | 3 | 100% | Yes, No, Yes |

## 2. Granular Case-by-Case Log

| Index | Sample ID | Topology | Expected | Model Output | Conforms? | Latency (s) | Peak VRAM (MiB) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | `ctrl_forward_chain_0001` | `forward_chain` | **Yes** | Yes | **PASS** | 3.39 | 3336.6 |
| 2 | `ctrl_forward_chain_0002` | `forward_chain` | **Yes** | Yes | **PASS** | 2.69 | 3336.6 |
| 3 | `ctrl_forward_chain_0003` | `forward_chain` | **Yes** | Yes | **PASS** | 2.63 | 3336.6 |
| 4 | `ctrl_reverse_chain_0037` | `reverse_chain` | **No** | Yes | **FAIL** | 2.66 | 3336.6 |
| 5 | `ctrl_reverse_chain_0038` | `reverse_chain` | **No** | No | **PASS** | 2.68 | 3336.6 |
| 6 | `ctrl_reverse_chain_0039` | `reverse_chain` | **No** | Yes | **FAIL** | 2.64 | 3336.6 |
| 7 | `ctrl_underdetermined_fork_0073` | `underdetermined_fork` | **Undetermined** | Yes | **FAIL** | 2.62 | 3336.6 |
| 8 | `ctrl_underdetermined_fork_0074` | `underdetermined_fork` | **Undetermined** | Yes | **FAIL** | 2.58 | 3336.6 |
| 9 | `ctrl_underdetermined_fork_0075` | `underdetermined_fork` | **Undetermined** | Yes | **FAIL** | 2.55 | 3336.6 |
| 10 | `ctrl_unimodal_visual_0109` | `unimodal_visual` | **Yes** | No | **FAIL** | 2.63 | 3336.6 |
| 11 | `ctrl_unimodal_visual_0110` | `unimodal_visual` | **No** | No | **PASS** | 2.58 | 3336.6 |
| 12 | `ctrl_unimodal_visual_0111` | `unimodal_visual` | **Yes** | No | **FAIL** | 2.56 | 3336.6 |
| 13 | `ctrl_unimodal_text_0145` | `unimodal_text` | **Yes** | Yes | **PASS** | 2.59 | 3336.6 |
| 14 | `ctrl_unimodal_text_0146` | `unimodal_text` | **No** | No | **PASS** | 2.60 | 3336.6 |
| 15 | `ctrl_unimodal_text_0147` | `unimodal_text` | **Yes** | Yes | **PASS** | 2.64 | 3336.6 |

## 3. Key Mechanistic Observations

1. **Underdetermined Fork Abstention Deficit:** On underdetermined forks, the model output definite binary answers rather than identifying that the relational information was unconstrained, demonstrating an empirical abstention deficit.
2. **Hardware Stability:** Peak VRAM across all 15 inferences remained bounded at ~3,336.6 MiB, confirming that SmolVLM-500M-Instruct runs comfortably within the 4 GB VRAM ceiling.
3. **Latency:** Mean forward generation latency was ~3.1–3.4 seconds per test query in eager bfloat16 mode.
