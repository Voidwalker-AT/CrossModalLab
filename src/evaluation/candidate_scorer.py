"""CrossModalLab — Full Candidate-Sequence Scorer.

Computes exact conditional sequence log-likelihoods:
    log P(Y | X) = sum_{t=1}^T log P(y_t | X, y_{<t})
and length-normalized average log-likelihoods:
    log P_norm(Y | X) = (1 / T) sum_{t=1}^T log P(y_t | X, y_{<t})
for arbitrary multi-token candidate answers under vision-language Transformer models.

Scientific Invariants:
- Preserves token alignment: strictly isolates candidate tokens from prompt context.
- Handles multi-token candidates (' Undetermined') and punctuation variants (' Yes.', ' No.').
- Computes both raw sequence log-likelihood and length-normalized scores.
- Computes candidate-normalized probabilities via softmax over the candidate set.
- Epistemic clarity: candidate-normalized scores are NOT treated as calibrated probabilities of correctness.
- Memory safe: cleans intermediate GPU tensors between evaluations.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import math
import torch
import torch.nn.functional as F

@dataclass
class TokenLogProb:
    token_id: int
    token_str: str
    log_prob: float
    prob: float

@dataclass
class CandidateScore:
    candidate_text: str
    token_ids: List[int]
    token_strings: List[str]
    token_log_probs: List[float]
    token_probs: List[float]
    total_log_likelihood: float
    length_normalized_log_likelihood: float
    sequence_length: int

@dataclass
class CandidateScorerResult:
    prompt_token_length: int
    candidates: Dict[str, CandidateScore]
    raw_ranking: List[str]
    length_normalized_ranking: List[str]
    candidate_probabilities_raw: Dict[str, float]
    candidate_probabilities_normalized: Dict[str, float]
    length_rank_flip: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_token_length": self.prompt_token_length,
            "candidates": {k: asdict(v) for k, v in self.candidates.items()},
            "raw_ranking": self.raw_ranking,
            "length_normalized_ranking": self.length_normalized_ranking,
            "candidate_probabilities_raw": self.candidate_probabilities_raw,
            "candidate_probabilities_normalized": self.candidate_probabilities_normalized,
            "length_rank_flip": self.length_rank_flip
        }

def compute_sequence_log_likelihood_from_logits(
    logits: torch.Tensor,
    candidate_token_ids: List[int],
    prompt_len: int
) -> Tuple[List[float], List[float], float, float]:
    """
    Computes exact conditional log probabilities for candidate tokens from logits.
    
    Args:
        logits: Tensor of shape [1, total_seq_len, vocab_size]
        candidate_token_ids: List of target token IDs for candidate answer Y (length T)
        prompt_len: Length of prompt X (L).
        
    Positions:
        The token y_1 is predicted from position prompt_len - 1.
        The token y_k is predicted from position prompt_len + k - 2.
        In general, for candidate token index k in 0..T-1:
            prediction_position = prompt_len - 1 + k
            target_token = candidate_token_ids[k]
    
    Returns:
        token_log_probs: List of float log-probs per candidate token
        token_probs: List of float probs per candidate token
        total_log_likelihood: sum of log-probs
        length_normalized_log_likelihood: total_log_likelihood / T
    """
    T = len(candidate_token_ids)
    if T == 0:
        raise ValueError("Candidate token sequence must be non-empty.")
    
    token_log_probs = []
    token_probs = []
    
    for k in range(T):
        pred_pos = prompt_len - 1 + k
        token_id = candidate_token_ids[k]
        pos_logits = logits[0, pred_pos, :].to(torch.float32)
        pos_log_probs = F.log_softmax(pos_logits, dim=-1)
        
        lp = pos_log_probs[token_id].item()
        p = math.exp(lp)
        token_log_probs.append(round(lp, 6))
        token_probs.append(round(p, 6))
        
    total_ll = sum(token_log_probs)
    norm_ll = total_ll / T
    
    return token_log_probs, token_probs, round(total_ll, 6), round(norm_ll, 6)

def score_candidate_answers(
    model: Any,
    processor: Any,
    base_inputs: Dict[str, Any],
    candidate_strings: List[str],
    device: str = "cuda:0"
) -> CandidateScorerResult:
    """
    Evaluates conditional sequence log-likelihood for multiple candidate answer strings
    under the exact same visual and prompt prefix.
    
    Args:
        model: Vision-language model
        processor: Processor / Tokenizer
        base_inputs: Processed prompt inputs (input_ids, pixel_values, attention_mask, etc.)
        candidate_strings: List of candidate answers, e.g. [" Yes", " No", " Undetermined"]
        device: Torch device
    """
    tokenizer = processor.tokenizer
    prompt_ids = base_inputs["input_ids"]
    prompt_len = prompt_ids.shape[1]
    
    candidate_scores: Dict[str, CandidateScore] = {}
    
    for cand_str in candidate_strings:
        cand_ids = tokenizer.encode(cand_str, add_special_tokens=False)
        cand_pieces = [tokenizer.decode([i]) for i in cand_ids]
        cand_tensor = torch.tensor([cand_ids], dtype=prompt_ids.dtype, device=prompt_ids.device)
        
        # Concatenate prompt + candidate tokens
        full_input_ids = torch.cat([prompt_ids, cand_tensor], dim=1)
        full_attention_mask = torch.ones_like(full_input_ids)
        
        # Prepare inputs dictionary
        cand_inputs = {
            "input_ids": full_input_ids,
            "attention_mask": full_attention_mask,
        }
        if "pixel_values" in base_inputs:
            cand_inputs["pixel_values"] = base_inputs["pixel_values"]
        if "pixel_attention_mask" in base_inputs:
            cand_inputs["pixel_attention_mask"] = base_inputs["pixel_attention_mask"]
            
        with torch.inference_mode():
            outputs = model(**cand_inputs)
            logits = outputs.logits
            
            token_lps, token_ps, total_ll, norm_ll = compute_sequence_log_likelihood_from_logits(
                logits, cand_ids, prompt_len
            )
            
            # Immediate GPU memory release
            del outputs, logits
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
                
        candidate_scores[cand_str] = CandidateScore(
            candidate_text=cand_str,
            token_ids=cand_ids,
            token_strings=cand_pieces,
            token_log_probs=token_lps,
            token_probs=token_ps,
            total_log_likelihood=total_ll,
            length_normalized_log_likelihood=norm_ll,
            sequence_length=len(cand_ids)
        )
        
    # Rankings
    raw_ranking = sorted(candidate_scores.keys(), key=lambda c: candidate_scores[c].total_log_likelihood, reverse=True)
    norm_ranking = sorted(candidate_scores.keys(), key=lambda c: candidate_scores[c].length_normalized_log_likelihood, reverse=True)
    length_flip = (raw_ranking != norm_ranking)
    
    # Softmax candidate-normalized probabilities (raw)
    raw_lls = [candidate_scores[c].total_log_likelihood for c in candidate_strings]
    max_raw_ll = max(raw_lls)
    raw_exps = [math.exp(ll - max_raw_ll) for ll in raw_lls]
    sum_raw_exps = sum(raw_exps)
    cand_probs_raw = {c: round(raw_exps[i] / sum_raw_exps, 6) for i, c in enumerate(candidate_strings)}
    
    # Softmax candidate-normalized probabilities (length-normalized)
    norm_lls = [candidate_scores[c].length_normalized_log_likelihood for c in candidate_strings]
    max_norm_ll = max(norm_lls)
    norm_exps = [math.exp(nll - max_norm_ll) for nll in norm_lls]
    sum_norm_exps = sum(norm_exps)
    cand_probs_norm = {c: round(norm_exps[i] / sum_norm_exps, 6) for i, c in enumerate(candidate_strings)}
    
    return CandidateScorerResult(
        prompt_token_length=prompt_len,
        candidates=candidate_scores,
        raw_ranking=raw_ranking,
        length_normalized_ranking=norm_ranking,
        candidate_probabilities_raw=cand_probs_raw,
        candidate_probabilities_normalized=cand_probs_norm,
        length_rank_flip=length_flip
    )
