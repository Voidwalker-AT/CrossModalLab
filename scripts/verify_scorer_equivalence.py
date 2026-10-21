"""CPU fixture comparing full-forward vs hybrid candidate scorers.

Uses a tiny deterministic model (no VLM weights). Verifies numerical agreement
and records runtime. Does not run GPU interventions.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn

from src.causal.scoring import score_candidates_hybrid, score_complete_candidates
from src.evaluation.candidate_scorer import score_candidate_answers


class TinyTok:
    """Byte-level fake tokenizer: id = ord(c) for printable chars, offset +1."""

    def encode(self, text: str, add_special_tokens: bool = False):
        return [ord(c) + 1 for c in text]

    def decode(self, ids, skip_special_tokens: bool = False):
        return "".join(chr(i - 1) for i in ids)


class TinyProc:
    def __init__(self):
        self.tokenizer = TinyTok()


class TinyModel(nn.Module):
    """Deterministic logits: depends only on last token id and position."""

    def __init__(self, vocab=300, hidden=16):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.proj = nn.Linear(hidden, vocab, bias=False)
        # fix weights deterministically
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.5)
        nn.init.eye_(self.proj.weight[:, :hidden])

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        h = self.embed(input_ids)
        # causal-ish: mean pool then last position residual-like
        pooled = h.mean(dim=1, keepdim=True)
        last = h[:, -1:, :] + 0.1 * pooled
        logits = self.proj(last)
        # expand to full sequence length for scorer API
        full = logits.expand(input_ids.shape[0], input_ids.shape[1], -1).contiguous()
        return type("O", (), {"logits": full})()


def main() -> int:
    torch.manual_seed(0)
    model = TinyModel()
    model.eval()
    proc = TinyProc()
    # prompt ids
    prompt = torch.tensor([[10, 11, 12, 13]], dtype=torch.long)
    base_inputs = {"input_ids": prompt}
    cands = [" A", " B", " CD"]

    t0 = time.perf_counter()
    ref = score_candidate_answers(model, proc, base_inputs, cands, device="cpu")
    t_ref = time.perf_counter() - t0

    t1 = time.perf_counter()
    full = score_complete_candidates(model, proc, base_inputs, cands, device="cpu")
    t_full = time.perf_counter() - t1

    t2 = time.perf_counter()
    hyb = score_candidates_hybrid(model, proc, base_inputs, cands, device="cpu")
    t_hyb = time.perf_counter() - t2

    ref_lp = {k: v.total_log_likelihood for k, v in ref.candidates.items()}
    max_diff_full = max(abs(ref_lp[k] - full["logprobs"][k]) for k in cands)
    max_diff_hyb = max(abs(ref_lp[k] - hyb["logprobs"][k]) for k in cands)

    print("SCORER_EQUIVALENCE")
    print(f"  max|ref-full|={max_diff_full:.3e}")
    print(f"  max|ref-hybrid|={max_diff_hyb:.3e}")
    print(f"  runtime_ref_s={t_ref:.4f} full_s={t_full:.4f} hybrid_s={t_hyb:.4f}")
    ok = max_diff_full <= 1e-9 and max_diff_hyb <= 1e-9
    print("  RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
