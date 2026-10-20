# Experiment Artifact Schemas & Synthetic Data Governance

**Document ID:** `SCHEMA-0001`  
**Phase:** Phase D1 (Dashboard Specification & Schemas)  
**Status:** Living Standard  

> [!CAUTION]
> **Synthetic Demonstration Data Rule:**
> The `is_synthetic_demo` flag is reserved strictly for **fabricated illustrative mockups** (e.g. mock test fixtures, UI tutorial records, fabricated logits or intervention effects). 
> Actual evaluation benchmark samples generated algorithmically by the symbolic ground-truth engine are **genuine evaluation artifacts** and must have `is_synthetic_demo: false`, even though their visual and textual stimuli are synthetically constructed. Fabricated illustrative examples must **NEVER** be cited as empirical results.

---

## 1. Synthetic Demo Quarantine & Validation Rules

To prevent contamination of empirical benchmarks:
1. **Mandatory Demo Flag:** Every experiment artifact, sample record, and intervention log includes a top-level boolean field `is_synthetic_demo`.
2. **Analytics Ingestion Gate:** All backend analytics endpoints and statistical aggregation functions must actively reject records where `is_synthetic_demo: true` with an explicit validation error / HTTP 422, rather than merely hiding them on the frontend:
   ```python
   # Invariant: Reject synthetic demonstration mockups from empirical analytics
   if record.get("is_synthetic_demo") is True:
       raise ValueError("Demo records (is_synthetic_demo=True) are quarantined from empirical analytics.")
   ```
3. **UI Visual Watermark:** If `is_synthetic_demo: true`, the frontend dashboard displays a persistent red badge: `[SYNTHETIC DEMO — EXCLUDED FROM BENCHMARK METRICS]`.
4. **No Fabricated Provenance:** Illustrative examples must not use realistic-looking fake commit hashes as verified truth; mock hashes are explicitly labeled `"demo_sha256_unverified"`.

---

## 2. Validated Schema Definitions

The architecture defines eight separate Pydantic / JSON Schema structures:

### Schema 1: Run Metadata (`run_meta.json`)
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "RunMetadata",
  "type": "object",
  "required": [
    "run_id",
    "timestamp",
    "is_synthetic_demo",
    "provenance",
    "metrics_summary"
  ],
  "properties": {
    "run_id": {"type": "string", "pattern": "^run_[0-9]{8}_[a-zA-Z0-9_]+$"},
    "timestamp": {"type": "string", "format": "date-time"},
    "is_synthetic_demo": {"type": "boolean"},
    "provenance": {
      "type": "object",
      "required": ["git_commit", "model_name", "dataset_id", "seeds", "environment"],
      "properties": {
        "git_commit": {"type": "string"},
        "git_branch": {"type": "string"},
        "dirty_repo": {"type": "boolean"},
        "model_name": {"type": "string"},
        "model_revision": {"type": "string"},
        "model_weights_sha256": {"type": "string"},
        "dataset_id": {"type": "string"},
        "dataset_sha256": {"type": "string"},
        "seeds": {
          "type": "object",
          "required": ["model_seed", "data_seed", "probe_seed"],
          "properties": {
            "model_seed": {"type": "integer"},
            "data_seed": {"type": "integer"},
            "probe_seed": {"type": "integer"}
          }
        },
        "environment": {
          "type": "object",
          "required": ["python_version", "torch_version", "os", "gpu", "vram_limit_mb"],
          "properties": {
            "python_version": {"type": "string"},
            "torch_version": {"type": "string"},
            "os": {"type": "string"},
            "gpu": {"type": "string"},
            "vram_limit_mb": {"type": "integer"}
          }
        }
      }
    },
    "metrics_summary": {
      "type": "object",
      "required": ["n_total", "accuracy_forward_chain", "accuracy_reverse_chain", "abstention_rate_fork"],
      "properties": {
        "n_total": {"type": "integer", "minimum": 1},
        "accuracy_forward_chain": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "accuracy_reverse_chain": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "abstention_rate_fork": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "forced_guess_rate_fork": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "ci_95": {
          "type": "object",
          "properties": {
            "forward_chain": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
            "reverse_chain": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
            "abstention_fork": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}
          }
        }
      }
    }
  }
}
```

---

### Schema 2: Dataset & Sample Provenance (`dataset_sample.json`)
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "DatasetSample",
  "type": "object",
  "required": [
    "sample_id",
    "is_synthetic_demo",
    "topology",
    "symbolic_ground_truth"
  ],
  "properties": {
    "sample_id": {"type": "string"},
    "is_synthetic_demo": {"type": "boolean"},
    "topology": {
      "type": "string",
      "enum": ["forward_chain", "reverse_chain", "underdetermined_fork", "unimodal_visual", "unimodal_text"]
    },
    "symbolic_ground_truth": {
      "type": "object",
      "required": ["relation", "premises", "query", "epistemic_status"],
      "properties": {
        "relation": {"type": "string", "enum": ["left_of"]},
        "premises": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["modality", "subject", "object", "relation"],
            "properties": {
              "modality": {"type": "string", "enum": ["visual", "textual"]},
              "subject": {"type": "string"},
              "object": {"type": "string"},
              "relation": {"type": "string"}
            }
          }
        },
        "query": {
          "type": "object",
          "required": ["subject", "object", "relation"],
          "properties": {
            "subject": {"type": "string"},
            "object": {"type": "string"},
            "relation": {"type": "string"}
          }
        },
        "epistemic_status": {
          "type": "string",
          "enum": ["entailed", "contradicted", "underdetermined"]
        },
        "consistent_linear_extensions": {
          "type": "array",
          "items": {"type": "array", "items": {"type": "string"}}
        }
      }
    }
  }
}
```

---

### Schema 3: Model Prediction Record (`model_prediction.json`)
*Separates raw observed outputs from diagnostic interpretations:*
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "ModelPrediction",
  "type": "object",
  "required": [
    "sample_id",
    "is_synthetic_demo",
    "raw_tokens",
    "generated_text",
    "candidate_scoring",
    "epistemic_behavior"
  ],
  "properties": {
    "sample_id": {"type": "string"},
    "is_synthetic_demo": {"type": "boolean"},
    "raw_tokens": {"type": "array", "items": {"type": "string"}},
    "generated_text": {"type": "string"},
    "candidate_scoring": {
      "type": "object",
      "required": ["length_penalty_alpha", "candidate_log_likelihoods", "normalized_candidate_likelihoods"],
      "properties": {
        "length_penalty_alpha": {"type": "number"},
        "candidate_log_likelihoods": {
          "type": "object",
          "required": ["Yes", "No", "Undetermined"],
          "properties": {
            "Yes": {"type": "number"},
            "No": {"type": "number"},
            "Undetermined": {"type": "number"}
          }
        },
        "normalized_candidate_likelihoods": {
          "type": "object",
          "required": ["Yes", "No", "Undetermined"],
          "properties": {
            "Yes": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "No": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "Undetermined": {"type": "number", "minimum": 0.0, "maximum": 1.0}
          }
        }
      }
    },
    "epistemic_behavior": {
      "type": "object",
      "required": ["decision", "is_abstention", "conforms_to_ground_truth"],
      "properties": {
        "decision": {"type": "string", "enum": ["Yes", "No", "Undetermined", "Unparseable"]},
        "is_abstention": {"type": "boolean"},
        "conforms_to_ground_truth": {"type": "boolean"},
        "failure_category": {
          "type": "string",
          "enum": ["none", "deductive_error", "abstention_failure", "perceptual_failure", "text_comprehension_failure"]
        }
      }
    }
  }
}
```

---

### Schema 4: Probe Result Record (`probe_result.json`)
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "ProbeResult",
  "type": "object",
  "required": [
    "sample_id",
    "is_synthetic_demo",
    "evaluated_layers",
    "premise_visual_trajectory",
    "premise_textual_trajectory",
    "composed_relation_trajectory",
    "topological_state_trajectory"
  ],
  "properties": {
    "sample_id": {"type": "string"},
    "is_synthetic_demo": {"type": "boolean"},
    "evaluated_layers": {"type": "array", "items": {"type": "integer"}},
    "premise_visual_trajectory": {"type": "array", "items": {"type": "number"}},
    "premise_textual_trajectory": {"type": "array", "items": {"type": "number"}},
    "composed_relation_trajectory": {"type": "array", "items": {"type": "number"}},
    "topological_state_trajectory": {"type": "array", "items": {"type": "number"}}
  }
}
```

---

### Schema 5: Activation Reference Record (`activation_ref.json`)
*Points to sparse, memory-mapped tensor slices on drive `E:` without in-memory tensor storage:*
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "ActivationReference",
  "type": "object",
  "required": [
    "sample_id",
    "is_synthetic_demo",
    "storage_path",
    "tensor_format",
    "tensor_shapes",
    "cached_layers"
  ],
  "properties": {
    "sample_id": {"type": "string"},
    "is_synthetic_demo": {"type": "boolean"},
    "storage_path": {"type": "string", "pattern": "^E:\\\\Workspace\\\\Active\\\\CrossModalLab\\\\.cache_activations/.*"},
    "tensor_format": {"type": "string", "enum": ["safetensors", "npy_mmap"]},
    "tensor_shapes": {
      "type": "object",
      "properties": {
        "residual_stream": {"type": "array", "items": {"type": "integer"}},
        "attention_weights": {"type": "array", "items": {"type": "integer"}},
        "mlp_activation": {"type": "array", "items": {"type": "integer"}}
      }
    },
    "cached_layers": {"type": "array", "items": {"type": "integer"}}
  }
}
```

---

### Schema 6: Causal Intervention Record (`causal_intervention.json`)
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "CausalIntervention",
  "type": "object",
  "required": [
    "intervention_id",
    "sample_id",
    "is_synthetic_demo",
    "clean_sample_id",
    "corrupted_sample_id",
    "target_component",
    "intervention_type",
    "observed_effect",
    "negative_controls"
  ],
  "properties": {
    "intervention_id": {"type": "string"},
    "sample_id": {"type": "string"},
    "is_synthetic_demo": {"type": "boolean"},
    "clean_sample_id": {"type": "string"},
    "corrupted_sample_id": {"type": "string"},
    "target_component": {
      "type": "object",
      "required": ["layer", "sub_block"],
      "properties": {
        "layer": {"type": "integer"},
        "sub_block": {"type": "string", "enum": ["residual", "attn_output", "mlp_output", "head"]},
        "head_index": {"type": ["integer", "null"]}
      }
    },
    "intervention_type": {
      "type": "string",
      "enum": ["activation_patch", "zero_ablation", "mean_ablation", "subspace_steering"]
    },
    "observed_effect": {
      "type": "object",
      "required": ["clean_target_logit", "corrupted_target_logit", "patched_target_logit", "normalized_indirect_effect"],
      "properties": {
        "clean_target_logit": {"type": "number"},
        "corrupted_target_logit": {"type": "number"},
        "patched_target_logit": {"type": "number"},
        "normalized_indirect_effect": {"type": "number"},
        "epistemic_verdict": {
          "type": "string",
          "enum": ["CAUSALLY_SUPPORTED", "NO_CAUSAL_EFFECT", "DESTRUCTIVE_ARTIFACT"]
        }
      }
    },
    "negative_controls": {
      "type": "object",
      "required": ["shuffled_noise_nie", "gaussian_noise_nie", "passed_controls"],
      "properties": {
        "shuffled_noise_nie": {"type": "number"},
        "gaussian_noise_nie": {"type": "number"},
        "passed_controls": {"type": "boolean"}
      }
    }
  }
}
```

---

### Schema 7: Resource Telemetry Record (`telemetry.json`)
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "ResourceTelemetry",
  "type": "object",
  "required": [
    "sample_id",
    "is_synthetic_demo",
    "vram_allocated_mb",
    "vram_reserved_mb",
    "host_ram_used_gb",
    "latency_forward_ms"
  ],
  "properties": {
    "sample_id": {"type": "string"},
    "is_synthetic_demo": {"type": "boolean"},
    "vram_allocated_mb": {"type": "number"},
    "vram_reserved_mb": {"type": "number"},
    "host_ram_used_gb": {"type": "number"},
    "latency_forward_ms": {"type": "number"},
    "oom_intercepted": {"type": "boolean", "default": false}
  }
}
```

---

### Schema 8: Scientific Evidence Classification (`evidence_classification.json`)
*Enforces strict categorization of all diagnostic assertions:*
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "EvidenceClassification",
  "type": "object",
  "required": [
    "claim_id",
    "claim_statement",
    "epistemic_status",
    "supporting_evidence_ids"
  ],
  "properties": {
    "claim_id": {"type": "string"},
    "claim_statement": {"type": "string"},
    "epistemic_status": {
      "type": "string",
      "enum": [
        "OBSERVED_BEHAVIOR",
        "CORRELATED_PROBE_FEATURE",
        "CAUSALLY_SUPPORTED_CIRCUIT",
        "UNVERIFIED_HYPOTHESIS"
      ]
    },
    "supporting_evidence_ids": {
      "type": "array",
      "items": {"type": "string"}
    }
  }
}
```

---

## 3. Illustrative Synthetic Demo Record

```json
{
  "run_id": "run_20261018_synthetic_demo_001",
  "timestamp": "2026-10-18T00:00:00Z",
  "is_synthetic_demo": true,
  "provenance": {
    "git_commit": "demo_commit_unverified",
    "git_branch": "demo_branch",
    "dirty_repo": false,
    "model_name": "SmolVLM-500M-Instruct (MOCK DEMO)",
    "model_revision": "demo_revision_unverified",
    "model_weights_sha256": "demo_sha256_unverified",
    "dataset_id": "demo_relational_split_v1",
    "dataset_sha256": "demo_sha256_unverified",
    "seeds": {"model_seed": 42, "data_seed": 1337, "probe_seed": 2026},
    "environment": {
      "python_version": "3.11.16",
      "torch_version": "2.4.0+cu118",
      "os": "Windows 11 Home 10.0.26200",
      "gpu": "NVIDIA GeForce RTX 2050 Laptop GPU",
      "vram_limit_mb": 4096
    }
  },
  "metrics_summary": {
    "n_total": 100,
    "accuracy_forward_chain": 0.85,
    "accuracy_reverse_chain": 0.82,
    "abstention_rate_fork": 0.10,
    "forced_guess_rate_fork": 0.90
  }
}
```
*Note: The above mock record has `is_synthetic_demo: true` and will be excluded by all empirical analytics filters.*
