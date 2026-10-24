"""Analytic CPU gradient checks; no model, optimizer or training data."""
import unittest
try:
    import torch
    from src.training.differentiable_scoring import evidence_sufficiency_from_class_scores
    from src.training.evidence_losses import counterfactual_consistency
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'optional Torch unavailable')
class PublicationObjectiveTests(unittest.TestCase):
    def test_active_hinge_gradient_directions_for_both_evidence_states(self):
        for scores,gold,sufficiency,expected in [([2.,0.,1.],2,0,[1.,0.,-1.]),([0.,2.,3.],1,1,[0.,-1.,1.])]:
            values=torch.tensor([scores],requires_grad=True)
            loss=evidence_sufficiency_from_class_scores(values,torch.tensor([gold]),sufficiency=torch.tensor([sufficiency]),margin=1.)
            loss.backward()
            torch.testing.assert_close(values.grad,torch.tensor([expected]))

    def test_counterfactual_objective_is_paired_ce_not_consistency_constraint(self):
        left=torch.tensor([[1.,0.,0.]],requires_grad=True)
        right=torch.tensor([[0.,1.,0.]],requires_grad=True)
        a,b=torch.tensor([0]),torch.tensor([1])
        actual=counterfactual_consistency(left,right,a,b)
        expected=.5*(torch.nn.functional.cross_entropy(left,a)+torch.nn.functional.cross_entropy(right,b))
        torch.testing.assert_close(actual,expected)


if __name__=='__main__':unittest.main()
