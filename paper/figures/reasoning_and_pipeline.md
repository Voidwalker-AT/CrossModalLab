# Conceptual schematics

These are logical and workflow schematics, not experimental observations or benchmark records. No synthetic demonstration data enters the empirical tables.

## Figure 1: epistemic states for strict left-of

```mermaid
flowchart LR
  A["R(A,B) and R(B,C)"] --> E["Entailed: R(A,C)"]
  B["R(B,A) and R(C,B)"] --> C["Contradicted: not R(A,C)"]
  D["R(A,B) and R(C,B)"] --> U["Underdetermined: both query directions remain possible"]
```

The two premises may be distributed across image and text. In the fork, the evidence leaves the ordering of A and C open. Underdetermined does not mean false. Inconsistent premises require a separate policy and are outside this three-state example.

## Figure 2: experiment and audit pipeline

```mermaid
flowchart TD
  M["Versioned manifest, split and asset hashes"] --> P["Frozen development schedule and prompts"]
  P --> V["Image encoder and multimodal projector: frozen"]
  P --> T["Text tokens"]
  V --> L["Pinned multimodal decoder with language LoRA"]
  T --> L
  L --> CE["Gold answer-span CE"]
  L --> ES["Full candidate LLs and evidence margin"]
  CE --> O["CE + 0.25 ES"]
  ES --> O
  O --> G["Finite gradients, memory and parameter checks"]
  G --> K["Atomic adapter, optimizer, RNG checkpoint and journal"]
  K --> D["Development generation and explicit parser accounting"]
```

This diagram indicates functional roles. It does not claim that a particular hidden layer implements logical composition. The diagram omits patching because no new intervention is authorized in this mission.
