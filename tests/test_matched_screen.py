import copy
import unittest

from src.evaluation.matched_screen import complete_matched_summary


class MatchedScreenTests(unittest.TestCase):
    def setUp(self):
        # Deliberately artificial unit-test fixtures, never empirical inputs.
        self.cases = [dict(sample_id=str(i), family_id='fixture', ground_truth_label=label,
                           image_sha256='fixture-image', prompt_sha256='fixture-prompt')
                      for i, label in enumerate(('Yes', 'No', 'Undetermined'))]
        self.rows = []
        for model in ('a', 'b'):
            for i, case in enumerate(self.cases):
                scores = [-5.0] * 3
                scores[i] = -1.0
                self.rows.append(dict(case, model=model, candidate_strings=[' Yes', ' No', ' Undetermined'],
                                      is_generated_completion=False, summed_loglikelihoods=scores,
                                      argmax_answer=case['ground_truth_label']))

    def evaluate(self, rows=None, status='PASSED'):
        return complete_matched_summary(self.cases, self.rows if rows is None else rows,
                                        ['a', 'b'], execution_status=status)

    def test_complete_matched_metrics(self):
        self.assertEqual(self.evaluate()['models']['a']['balanced_accuracy'], 1.0)
        self.assertEqual(self.evaluate()['families'], 1)

    def test_failed_execution_refused_even_with_rows(self):
        with self.assertRaisesRegex(ValueError, 'incomplete'): self.evaluate(status='FAILED')

    def test_missing_and_duplicate_scores_refused(self):
        for rows in (self.rows[:-1], self.rows + self.rows[:1]):
            with self.assertRaises(ValueError): self.evaluate(rows)

    def test_input_mismatch_refused(self):
        rows = copy.deepcopy(self.rows); rows[0]['image_sha256'] = 'other'
        with self.assertRaisesRegex(ValueError, 'identity mismatch'): self.evaluate(rows)

    def test_nonfinite_and_wrong_argmax_refused(self):
        for scores in ([float('nan'), -2, -3], [-5, -1, -3]):
            rows = copy.deepcopy(self.rows); rows[0]['summed_loglikelihoods'] = scores
            with self.assertRaises(ValueError): self.evaluate(rows)

    def test_two_label_pilot_cannot_become_three_class_result(self):
        with self.assertRaisesRegex(ValueError, 'three ground-truth'):
            complete_matched_summary(self.cases[:2], [r for r in self.rows if r['sample_id'] != '2'],
                                    ['a', 'b'], execution_status='PASSED')
