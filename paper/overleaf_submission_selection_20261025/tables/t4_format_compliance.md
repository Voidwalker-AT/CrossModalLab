# Prefix-parser semantic correctness, invalid labels and exact-label formatting are separate endpoints.

Source: audit/dual_model_audit.json#qwen.metrics

| Condition | Correct | Invalid | Exact |
|---|---|---|---|
| frozen | 7/18 | 0/18 | 18/18 |
| prompt | 8/18 | 0/18 | 18/18 |
| ce5 | 7/18 | 6/18 | 0/18 |
| es5 | 9/18 | 0/18 | 18/18 |
| ce50 | 12/18 | 0/18 | 0/18 |
| es5prompt | 8/18 | 0/18 | 18/18 |
