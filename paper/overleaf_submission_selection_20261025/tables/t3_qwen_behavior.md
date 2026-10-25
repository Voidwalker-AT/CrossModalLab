# Historical Qwen generations on six development families; descriptive, not a superiority test.

Source: audit/dual_model_audit.json#qwen.metrics; original audit/raw_predictions.json

| Condition | Correct | BA | Macro F1 | U recall |
|---|---|---|---|---|
| frozen | 7/18 | 0.388889 | 0.363212 | 2/6 |
| prompt | 8/18 | 0.444444 | 0.444444 | 2/6 |
| ce5 | 7/18 | 0.388889 | 0.400000 | 0/6 |
| es5 | 9/18 | 0.500000 | 0.511111 | 2/6 |
| ce50 | 12/18 | 0.666667 | 0.555556 | 0/6 |
| es5prompt | 8/18 | 0.444444 | 0.444444 | 2/6 |
