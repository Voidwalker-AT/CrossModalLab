"""Phase 2I-B CPU tests: differentiable candidate scoring + ES loss."""

from __future__ import annotations

import json
import unittest

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

from src.config.paths import REPO_ROOT

if torch is not None:
    from src.training.differentiable_scoring import (
        candidate_class_loglikelihoods,
        evidence_sufficiency_from_class_scores,
        total_loss,
    )


class TinyCausal(nn.Module if nn is not None else object):
    """SYNTHETIC DEMO: causal fixture; never used for empirical metrics."""

    is_synthetic_demo = True

    def __init__(self, vocab=64, hidden=16):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.scale = nn.Parameter(torch.ones(hidden))
        self.head = nn.Linear(hidden, vocab, bias=False)
        nn.init.normal_(self.emb.weight, std=0.5)
        nn.init.eye_(self.head.weight[:, :hidden])

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        h = self.emb(input_ids) * self.scale
        logits = self.head(h.cumsum(dim=1))
        return type("O", (), {"logits": logits})()


@unittest.skipIf(torch is None, "optional torch unavailable")
class TestCandidateScoring(unittest.TestCase):
    def test_single_and_multi_token(self):
        torch.manual_seed(0)
        model = TinyCausal().train()
        tok_ids = [[10, 11, 12]]  # 3-token prompt
        ids = [[5], [5, 6]]  # one-token and two-token candidates (fake ids)
        base = {"input_ids": torch.tensor(tok_ids)}
        ll = candidate_class_loglikelihoods(
            model, base, ids, candidates=(" A", " B")
        )
        self.assertEqual(tuple(ll.shape), (1, 2))
        self.assertTrue(torch.isfinite(ll).all())
        self.assertTrue(ll.requires_grad)

    def test_gradients_flow_to_parameters(self):
        torch.manual_seed(0)
        model = TinyCausal().train()
        base = {"input_ids": torch.tensor([[10, 11, 12]])}
        ll = candidate_class_loglikelihoods(model, base, [[5], [6, 7]], candidates=(" A", " B"))
        loss = ll.sum()
        loss.backward()
        self.assertIsNotNone(model.scale.grad)
        self.assertGreater(float(model.scale.grad.abs().sum()), 0.0)

    def test_frozen_weights_unchanged_after_backward(self):
        model = TinyCausal()
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name == "scale")
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        ll = candidate_class_loglikelihoods(model, {"input_ids": torch.tensor([[1, 2, 3]])},
                                           [[5], [6], [7, 8]])
        es = evidence_sufficiency_from_class_scores(ll, torch.tensor([2]),
                                                   sufficiency=torch.tensor([0]), margin=100.0)
        es.backward()
        self.assertGreater(model.scale.grad.abs().sum().item(), 0)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
            if name != "scale":
                self.assertIsNone(parameter.grad)

    def test_causal_shift_multi_token_stable(self):
        torch.manual_seed(1)
        model = TinyCausal().eval()
        base = {"input_ids": torch.tensor([[1, 2, 3, 4]])}
        with torch.no_grad():
            ll = candidate_class_loglikelihoods(model, base, [[20, 21], [20]], candidates=(" X", " Y"))
        self.assertTrue(torch.isfinite(ll).all())

    def test_complete_scores_match_autoregressive_oracle(self):
        torch.manual_seed(8)
        model = TinyCausal().eval()
        prompt = torch.tensor([[1, 2, 3]])
        candidates = [[20], [20, 21], [20, 21, 22]]
        ll = candidate_class_loglikelihoods(model, {"input_ids": prompt}, candidates, candidates=("a", "b", "c"))
        expected = []
        # Independent token-at-a-time chain rule, with no future candidate tokens.
        for candidate in candidates:
            prefix = prompt
            score = torch.zeros(())
            for token in candidate:
                logits = model(input_ids=prefix).logits[0, -1].float()
                score = score + logits[token] - torch.logsumexp(logits, dim=0)
                prefix = torch.cat([prefix, torch.tensor([[token]])], dim=1)
            expected.append(score)
        torch.testing.assert_close(ll[0], torch.stack(expected))
        expected_grad = torch.autograd.grad(torch.stack(expected).sum(), model.scale, retain_graph=True)[0]
        actual_grad = torch.autograd.grad(ll.sum(), model.scale)[0]
        torch.testing.assert_close(actual_grad, expected_grad)

    def test_metadata_extension_and_no_duplication(self):
        calls = []
        model = TinyCausal()
        model.register_forward_pre_hook(lambda m, a, kw: calls.append(kw), with_kwargs=True)
        base = {"input_ids": torch.tensor([[0, 1, 2]]),
                "attention_mask": torch.tensor([[0, 1, 1]]),
                "mm_token_type_ids": torch.tensor([[0, 1, 0]]),
                "token_type_ids": torch.tensor([[0, 1, 0]]),
                "pixel_values": torch.randn(4, 8), "image_grid_thw": torch.tensor([[1, 2, 2]]),
                "position_ids": torch.arange(3).view(1, 1, 3).expand(3, 1, 3)}
        before = {k: v.clone() for k, v in base.items()}
        candidate_class_loglikelihoods(model, base, [[5], [6, 7]], candidates=("a", "b"))
        for call, ids in zip(calls, [[5], [6, 7]]):
            self.assertEqual(call["input_ids"].tolist(), [[0, 1, 2] + ids])
            self.assertEqual(call["attention_mask"].tolist(), [[0, 1, 1] + [1] * len(ids)])
            for key in ("token_type_ids", "mm_token_type_ids"):
                self.assertEqual(call[key].tolist(), [[0, 1, 0] + [0] * len(ids)])
            self.assertNotIn("position_ids", call)
            self.assertFalse(call["use_cache"])
            self.assertIs(call["pixel_values"], base["pixel_values"])
            self.assertIs(call["image_grid_thw"], base["image_grid_thw"])
        for key in before:
            torch.testing.assert_close(base[key], before[key])

    def test_explicit_text_positions_extend(self):
        for positions in (torch.tensor([[3, 4, 5]]), torch.tensor([[[3, 4, 5]]] * 3)):
            calls = []
            model = TinyCausal()
            model.register_forward_pre_hook(lambda m, a, kw: calls.append(kw), with_kwargs=True)
            candidate_class_loglikelihoods(model, {"input_ids": torch.tensor([[1, 2, 3]]),
                                                  "position_ids": positions}, [[5, 6]], candidates=("a",))
            expected = torch.cat([positions, positions[..., -1:] + torch.tensor([1, 2])], -1)
            torch.testing.assert_close(calls[0]["position_ids"], expected)

    def test_original_qwen_mask_mismatch_regression(self):
        try:
            from transformers import Qwen3VLConfig
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
        except ImportError:
            self.skipTest("optional Qwen implementation unavailable")
        from types import SimpleNamespace, MethodType
        config = Qwen3VLConfig()
        rope = SimpleNamespace(config=config)
        rope.get_vision_position_ids = MethodType(Qwen3VLModel.get_vision_position_ids, rope)
        class QwenPositionFixture(TinyCausal):
            def forward(self, input_ids=None, attention_mask=None, mm_token_type_ids=None, image_grid_thw=None, **kw):
                pos, _ = Qwen3VLModel.get_rope_index(rope, input_ids, mm_token_type_ids,
                                                   image_grid_thw=image_grid_thw, attention_mask=attention_mask)
                assert pos.shape == (3, 1, input_ids.shape[1])
                return super().forward(input_ids=input_ids)
        base = {"input_ids": torch.ones((1, 228), dtype=torch.long),
                "attention_mask": torch.ones((1, 228), dtype=torch.long),
                "mm_token_type_ids": torch.zeros((1, 228), dtype=torch.long),
                "image_grid_thw": torch.tensor([[1, 2, 2]])}
        base["mm_token_type_ids"][0, 2] = 1
        scores = candidate_class_loglikelihoods(QwenPositionFixture(), base, [[5], [6, 7]], candidates=("a", "b"))
        self.assertTrue(torch.isfinite(scores).all())

    def test_invalid_inputs_rejected_before_forward(self):
        valid = {"input_ids": torch.tensor([[1, 2, 3]])}
        invalid = [({"input_ids": torch.empty((1, 0), dtype=torch.long)}, [[5]]),
                   ({"input_ids": torch.ones((2, 3), dtype=torch.long)}, [[5]]),
                   ({"input_ids": torch.ones((1, 3))}, [[5]]),
                   (valid, [[]]), (valid, [[-1]]), (valid, [[1.5]]), (valid, [[True]]),
                   (valid, [])]
        for key in ("attention_mask", "token_type_ids", "mm_token_type_ids", "position_ids"):
            invalid.append(({**valid, key: torch.ones((1, 2), dtype=torch.long)}, [[5]]))
        for key in ("past_key_values", "inputs_embeds", "labels", "cache_position"):
            invalid.append(({**valid, key: torch.ones(1)}, [[5]]))
        invalid += [({**valid, "attention_mask": torch.tensor([[1, 1, 0]])}, [[5]]),
                    ({**valid, "attention_mask": torch.tensor([[1, 2, 1]])}, [[5]]),
                    ({**valid, "logits_to_keep": 1}, [[5]])]
        for base, ids in invalid:
            with self.subTest(keys=list(base), ids=ids):
                with self.assertRaises(ValueError):
                    candidate_class_loglikelihoods(TinyCausal(), base, ids, candidates=("a",))


@unittest.skipIf(torch is None, "optional torch unavailable")
class TestESScore(unittest.TestCase):
    def test_prefers_abstain_when_insufficient(self):
        # abstain score high → low hinge when sufficiency=0
        class_ll = torch.tensor([[-2.0, -2.0, 0.0]], dtype=torch.float32, requires_grad=True)
        gold = torch.tensor([0])
        suff = torch.tensor([0])
        loss = evidence_sufficiency_from_class_scores(class_ll, gold, sufficiency=suff, margin=1.0)
        self.assertAlmostEqual(float(loss), 0.0, places=5)
        loss.backward()
        self.assertIsNotNone(class_ll.grad)

    def test_penalizes_definite_when_insufficient(self):
        class_ll = torch.tensor([[0.0, 0.0, -5.0]], dtype=torch.float32, requires_grad=True)
        gold = torch.tensor([0])
        suff = torch.tensor([0])
        loss = evidence_sufficiency_from_class_scores(class_ll, gold, sufficiency=suff, margin=1.0)
        self.assertGreater(float(loss), 0.0)

    def test_anti_over_abstention_when_sufficient(self):
        class_ll = torch.tensor([[-4.0, -4.0, 4.0]], dtype=torch.float32, requires_grad=True)
        gold = torch.tensor([0])
        suff = torch.tensor([1])
        loss = evidence_sufficiency_from_class_scores(class_ll, gold, sufficiency=suff, margin=1.0)
        self.assertGreater(float(loss), 0.0)

    def test_sufficient_and_correct_low_hinge(self):
        class_ll = torch.tensor([[4.0, -2.0, -2.0]], dtype=torch.float32)
        gold = torch.tensor([0])
        suff = torch.tensor([1])
        loss = evidence_sufficiency_from_class_scores(class_ll, gold, sufficiency=suff, margin=1.0)
        self.assertAlmostEqual(float(loss), 0.0, places=5)

    def test_extreme_stable(self):
        class_ll = torch.tensor([[100.0, -100.0, -100.0]], dtype=torch.float32)
        loss = evidence_sufficiency_from_class_scores(
            class_ll, torch.tensor([0]), sufficiency=torch.tensor([1])
        )
        self.assertTrue(torch.isfinite(loss))

    def test_zero_mu_equals_ce(self):
        ce = torch.tensor(1.5, requires_grad=True)
        es = torch.tensor(9.9, requires_grad=True)
        total = total_loss(ce, es, mu_es=0.0)
        self.assertAlmostEqual(float(total), 1.5)

    def test_negative_mu_rejected(self):
        ce = torch.tensor(1.0)
        es = torch.tensor(1.0)
        with self.assertRaises(ValueError):
            total_loss(ce, es, mu_es=-0.1)


class TestConfigFrozen(unittest.TestCase):
    def test_ablation_config(self):
        cfg = json.loads(
            (REPO_ROOT / "configs/phase_2i_b/es_ablation_config_v1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(cfg["es_objective"]["mu_es"], 0.25)
        self.assertEqual(cfg["es_objective"]["margin"], 1.0)
        self.assertFalse(cfg["matched_to_2i_a"]["holdout_allowed"])
        self.assertTrue(cfg["no_tuning_on_val"])


if __name__ == "__main__":
    unittest.main()
