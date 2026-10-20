# Mechanistic Interpretability Methodology: Diagnostic Taxonomy & Causal Standards

**Document ID:** `METH-0001`  
**Phase:** Phase 0A / D1 (Foundation & Specifications)  
**Status:** Living Standard  

---

## 1. The Epistemic Hierarchy of Mechanistic Evidence

To ensure scientific validity, CrossModalLab establishes a strict hierarchy of mechanistic evidence, explicitly separating correlational observations from causal interventions.

```text
[Level 1: Behavioral Metrics]
   └── Model outputs and error rates under controlled inputs (Black-box observation)
        ↓
[Level 2: Correlational Probing]
   └── Linear probes / classifiers trained on hidden activations (Measures decodability)
        ↓
[Level 3: Latent Unembedding / Projections]
   └── Logit-lens, tuned-lens, attention heatmaps (Correlational vocabulary / routing readout)
        ↓
[Level 4: Causal Activation Interventions]
   └── Activation patching, causal tracing, interchange interventions (Proves causal mediation)
        ↓
[Level 5: Circuit Identification & Path Patching]
   └── End-to-end sub-network isolation ruling out parallel bypasses
```

---

## 2. Probing vs. Causal Interventions: Fundamental Distinctions

### A. Linear Probing & Representation Decoding
* **Definition:** A diagnostic classifier (e.g., Logistic Regression or Ridge Classifier) trained to predict a target attribute $y$ from the intermediate hidden activations $h_l(x)$ at layer $l$.
* **Epistemic Invariant 1 (Decodability $\neq$ Causal Usage):**  
  > *High probe accuracy proves only that the feature is linearly decodable from the hidden state by an external observer. It provides **zero proof** that downstream layers of the Transformer actually attend to, extract, or utilize that feature during token generation.*
* **Epistemic Invariant 2 (Probe Failure $\neq$ Feature Absence):**  
  > *Low probe accuracy does **not** prove that the feature is absent from the network. The feature may be encoded non-linearly, distributed across orthogonal subspaces, or localized to a different token position.*
* **Capacity Controls:**
  * Only regularized linear probes (with cross-validated penalty $\alpha$) are permitted as primary decodability validators.
  * Expressive non-linear probes (e.g. multi-layer perceptrons) are **prohibited** as evidence of model representation because a high-capacity probe can compute the relational composition itself rather than measuring whether the model has computed it.
  * All probe evaluations must compare against task-appropriate chance baselines and report 95% bootstrap confidence intervals.

### B. Latent Projections (The Logit Lens)
* **Definition:** Projecting intermediate residual stream hidden states $h_l$ directly into vocabulary logit space using the model's final unembedding matrix $W_U$:
$$\text{logits}_l = \text{Softmax}(W_U \cdot \text{LayerNorm}(h_l))$$
* **Epistemic Invariant 3 (Logit Lens is Correlational, NOT Causal):**  
  > *The logit lens inspects what token the residual stream is closest to at layer $l$ under the final unembedding projection. It is a **correlational projection**, not a causal intervention. A high logit for a target token at layer $l$ does not prove that subsequent layers use that state to produce the final output.*

### C. Causal Activation Interventions & Their Non-Uniqueness
1. **Activation Patching (Causal Tracing):**
   * Given a *Clean Run* (Forward Chain) and a *Corrupted Run* (Premise altered or noisy):
   * Replace the activation of a candidate component $c$ (layer $l$, head $h$, or MLP block) in the Corrupted Run with the activation from the Clean Run:
$$h_l^{\text{patched}} = h_l^{\text{corrupted}} \quad \text{with} \quad c \leftarrow c^{\text{clean}}$$
2. **Epistemic Invariant 4 (Patching $\neq$ Unique Mechanism Proof):**
   > *Activation patching demonstrates **causal mediation along the evaluated computational path**. It does **not** prove that the identified component is the unique necessary and sufficient mechanism:*
   * A component can appear "necessary" because ablating it introduces an out-of-distribution (OOD) perturbation that destructively corrupts downstream layers, even if the component does not carry the semantic feature.
   * A component can appear "sufficient" because restoring it restores a coarse baseline state, while parallel redundant circuits were also capable of the computation.
   * Proving a **unique circuit** requires: (1) path patching to trace direct component-to-component causal edges, (2) multi-component ablation to check for redundant bypasses, and (3) variance-matched negative controls (shuffled activations, Gaussian noise).

---

## 3. Ontological Distinctions in Mechanistic Interpretability

To prevent conceptual ambiguity, CrossModalLab strictly distinguishes four levels of internal neural representation:

```text
[Neuron] ───────────────> [Activation Dimension] ──────> [Sparse Feature] ─────────> [Causal Circuit]
Scalar coordinate in      Arbitrary basis coordinate     Monosemantic direction      Causally verified
MLP non-linear act.       in residual stream R^D         learned via SAE dictionary  multi-module subgraph
```

1. **Individual Neuron:** A single scalar coordinate within an MLP intermediate layer following the non-linear activation (e.g. the $i$-th element of $\text{SiLU}(W_{\text{gate}} x)$). Individual neurons in polysemantic models often fire on unrelated mixtures of concepts.
2. **Activation Dimension:** An arbitrary basis vector coordinate in the residual stream $h_l \in \mathbb{R}^{d_{\text{model}}}$. Because the residual stream exhibits rotational symmetry under orthogonal transformations, individual raw dimensions rarely correspond to cleanly interpretable features.
3. **Sparse Learned Feature:** A reconstructed feature vector $f_j = W_{\text{enc}} x$ extracted via Sparse Autoencoders (SAEs) or dictionary learning, which aligns with monosemantic, interpretable concepts.
4. **Causal Circuit:** A minimal directed acyclic subgraph of attention heads, MLP blocks, and residual stream channels that is causally necessary and sufficient to mediate a specific reasoning task.

---

## 4. Hardware Constraints & Memory Safety (RTX 2050 4 GB)

Executing mechanistic interpretability on an NVIDIA RTX 2050 Laptop GPU (4096 MiB VRAM) on Windows 11 requires strict computational safety:

### Analytical Memory Derivations
For batch size $B=1$, total tokens $S = S_{\text{vis}} + S_{\text{text}} = 260$, hidden dimension $d_{\text{model}}$, intermediate MLP dimension $d_{\text{mlp}}$, heads $H$, and 16-bit precision (2 bytes/element):
* **Residual Stream Slice:** $1 \times 260 \times d_{\text{model}} \times 2\text{ bytes}$
  - SmolVLM-500M ($d_{\text{model}}=1024$): **$0.53\text{ MB}$** per layer
  - SmolVLM-2B ($d_{\text{model}}=2048$): **$1.06\text{ MB}$** per layer
* **Attention Weight Matrix:** $1 \times H \times 260 \times 260 \times 2\text{ bytes}$
  - SmolVLM-500M ($H=16$): **$2.16\text{ MB}$** per layer
  - SmolVLM-2B ($H=32$): **$4.32\text{ MB}$** per layer
* **MLP Activation Tensor:** $1 \times 260 \times d_{\text{mlp}} \times 2\text{ bytes}$
  - SmolVLM-500M ($d_{\text{mlp}}=3584$): **$1.86\text{ MB}$** per layer
  - SmolVLM-2B ($d_{\text{mlp}}=8192$): **$4.26\text{ MB}$** per layer

### Realistic Hardware Budget Accounting
Analytical calculations must account for system overheads:
1. **Windows WDDM Display Overhead:** Windows desktop composition and UI applications consume $\sim 500\text{ MB}$ of VRAM continuously. Available VRAM is approximately **$3,500\text{ MB}$**.
2. **PyTorch Allocator Overhead:** The CUDA caching allocator introduces memory fragmentation. Calling `torch.cuda.set_per_process_memory_fraction(0.85)` is a protective allocator ceiling, **not** an absolute physical guarantee against driver-level out-of-memory spikes.
3. **KV Cache & Runtime Activations:** For $S=260$, KV cache is negligible ($< 5\text{ MB}$), but forward activation buffers in attention layers require $\sim 200\text{ MB}$.
4. **Candidate Model Allocation:**
   - **SmolVLM-500M-Instruct (Preferred First Candidate):** Weights in bfloat16 require $\sim 1,000\text{ MB}$. Peak VRAM under forward-hook extraction is estimated at $\sim 1,800\text{ MB}$, well within the $3,500\text{ MB}$ limit.
   - **SmolVLM-2B (Secondary Candidate):** Weights in bfloat16 require $\sim 4,200\text{ MB}$ (exceeds VRAM). Requires 4-bit BitsAndBytes quantization ($\sim 1,400\text{ MB}$ weights) or CPU offloading to run.

### Safe OOM Handling Protocol
All model execution and activation interception must wrap in defensive exception handlers:
```python
import gc
import torch

def safe_forward_execution(model, inputs):
    try:
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs
    except torch.cuda.OutOfMemoryError as e:
        # 1. Clear PyTorch CUDA allocator cache
        torch.cuda.empty_cache()
        # 2. Force host garbage collection
        gc.collect()
        raise RuntimeError("GPU Out-of-Memory intercepted. VRAM cache cleared.") from e
```

---

## 5. Architecture-Specific Hook Adapters

Heterogeneous vision-language architectures return varying data structures. We enforce the Adapter Pattern:

```python
from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional
import torch

class BaseActivationAdapter(ABC):
    @abstractmethod
    def extract_hidden_state(self, hook_output: Any) -> torch.Tensor:
        """Extracts (B, S, D) residual state, detached to CPU float16."""
        pass

    @abstractmethod
    def extract_attention_weights(self, hook_output: Any) -> Optional[torch.Tensor]:
        """Extracts (B, H, S, S) attention weights, detached to CPU float16."""
        pass

class HuggingFaceDecoderLayerAdapter(BaseActivationAdapter):
    def extract_hidden_state(self, hook_output: Any) -> torch.Tensor:
        if isinstance(hook_output, tuple):
            return hook_output[0].detach().cpu().to(torch.float16)
        return hook_output.detach().cpu().to(torch.float16)

    def extract_attention_weights(self, hook_output: Any) -> Optional[torch.Tensor]:
        if isinstance(hook_output, tuple) and len(hook_output) > 1:
            attn = hook_output[1]
            if isinstance(attn, torch.Tensor):
                return attn.detach().cpu().to(torch.float16)
        return None

class SiglipVisionEncoderAdapter(BaseActivationAdapter):
    def extract_hidden_state(self, hook_output: Any) -> torch.Tensor:
        if hasattr(hook_output, "last_hidden_state"):
            return hook_output.last_hidden_state.detach().cpu().to(torch.float16)
        elif isinstance(hook_output, tuple):
            return hook_output[0].detach().cpu().to(torch.float16)
        return hook_output.detach().cpu().to(torch.float16)

    def extract_attention_weights(self, hook_output: Any) -> Optional[torch.Tensor]:
        if hasattr(hook_output, "attentions") and hook_output.attentions is not None:
            return hook_output.attentions[-1].detach().cpu().to(torch.float16)
        return None
```
