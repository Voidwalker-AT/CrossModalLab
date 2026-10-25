# D. Prefix-parser correctness and exact-label compliance are separate endpoints.

| condition | semantic_invalid | semantic_correct | exact_label | malformed | source | status |
|---|---|---|---|---|---|---|
| frozen | 0/18 | 7/18 | 18/18 | 0/18 | audit/raw_predictions.json#frozen | VERIFIED |
| prompt | 0/18 | 8/18 | 18/18 | 0/18 | audit/raw_predictions.json#prompt | VERIFIED |
| ce5 | 6/18 | 7/18 | 0/18 | 18/18 | audit/raw_predictions.json#ce5 | VERIFIED |
| es5 | 0/18 | 9/18 | 18/18 | 0/18 | audit/raw_predictions.json#es5 | VERIFIED |
| ce50 | 0/18 | 12/18 | 0/18 | 18/18 | audit/raw_predictions.json#ce50 | VERIFIED |
| es5prompt | 0/18 | 8/18 | 18/18 | 0/18 | audit/raw_predictions.json#es5prompt | VERIFIED |
