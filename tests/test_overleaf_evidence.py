"""Metric-audit regressions. All records here are synthetic test fixtures only."""
import unittest
from scripts.build_overleaf_evidence import recompute,same_metrics

class OverleafEvidenceTests(unittest.TestCase):
    def rows(self):
        return [dict(sample_id=str(i),family_id='fixture',ground_truth_label=g,parsed_answer=p,raw_completion=r)
                for i,(g,p,r)in enumerate([('Yes','Yes','Yes'),('No','No','No'+'\ufffd'*15),('Undetermined','Invalid','nonsense')])]

    def test_malformed_correct_prefix_is_not_format_compliance(self):
        rows=self.rows();original=rows[1]['raw_completion'];m=recompute(rows)
        self.assertEqual(m['correct'],2);self.assertEqual(m['semantic_invalid'],1)
        self.assertEqual(m['exact_format'],1);self.assertEqual(rows[1]['raw_completion'],original)

    def test_invalid_predictions_remain_in_recall_and_f1_denominators(self):
        m=recompute(self.rows());self.assertEqual(m['balanced_accuracy'],2/3)
        self.assertEqual(m['macro_f1'],2/3);self.assertEqual(m['support']['Undetermined'],1)
        self.assertEqual(m['confusion_matrix']['Undetermined']['Invalid'],1)

    def test_duplicate_missing_class_and_demo_records_rejected(self):
        rows=self.rows()
        for bad in (rows+[rows[0]],rows[:2],[{**r,'is_synthetic_demo':True}for r in rows]):
            with self.assertRaises(ValueError):recompute(bad)

    def test_cross_python_reduction_roundoff_does_not_hide_count_changes(self):
        self.assertTrue(same_metrics({'BA':.5,'correct':9},{'BA':.49999999999999994,'correct':9}))
        self.assertFalse(same_metrics({'BA':.5,'correct':8},{'BA':.5,'correct':9}))
        self.assertFalse(same_metrics({'BA':.51},{'BA':.5}))

if __name__=='__main__':unittest.main()
