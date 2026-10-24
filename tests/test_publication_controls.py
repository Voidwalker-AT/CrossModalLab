import unittest

try:
    import numpy as np
    from src.causal.interventions import InterventionSpec
    from src.causal.publication_controls import aligned_span,controlled_replacement,eligible_pair
except ImportError:
    np=None


@unittest.skipIf(np is None,'optional NumPy unavailable')
class PublicationControlTests(unittest.TestCase):
    def test_unique_alignment_handles_shift_and_rejects_ambiguity(self):
        self.assertEqual(aligned_span([0,3,4],[9,8,3,4],[3,4]),((1,3),(2,4)))
        for anchor in ([],[6],[3]):
            with self.assertRaises(ValueError):aligned_span([3,3,4],[3,4],anchor)

    def test_no_op_and_single_aligned_token_replacement(self):
        source=np.arange(6,dtype=np.float32).reshape(3,2)
        target=np.ones((4,2),dtype=np.float32)
        original=target.copy()
        options=dict(source_ids=[0,3,4],target_ids=[9,8,3,4],anchor=[3,4])
        no_op=controlled_replacement(target,source,spec=InterventionSpec(1,3,'no_op','none'),**options)
        np.testing.assert_array_equal(no_op,target)
        patched=controlled_replacement(target,source,spec=InterventionSpec(1,3,'clean_to_cf','counterfactual'),**options)
        np.testing.assert_array_equal(patched[:3],target[:3])
        np.testing.assert_array_equal(patched[3],source[2])
        np.testing.assert_array_equal(target,original)
        with self.assertRaisesRegex(ValueError,'position'):controlled_replacement(target,source,spec=InterventionSpec(1,2,'clean_to_cf','counterfactual'),**options)

    def test_opposed_margin_screen_never_uses_unstable_normalization(self):
        self.assertTrue(eligible_pair([0,2,1],[0,1,2])['eligible'])
        self.assertFalse(eligible_pair([0,2,1],[0,2,1])['eligible'])
        self.assertFalse(eligible_pair([0,1,1],[0,1,1])['eligible'])
        with self.assertRaises(ValueError):eligible_pair([0,float('nan'),1],[0,1,2])


if __name__=='__main__':unittest.main()
