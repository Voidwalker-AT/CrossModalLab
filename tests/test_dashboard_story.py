"""Scientific/export regressions: preserve evidence while simplifying presentation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
from dashboard.backend.artifacts import ArtifactStore
from dashboard.backend.exports import comparison_svg, logical_svg
from dashboard.backend.explanations import comparison_takeaway
from dashboard.backend.gallery import render_gallery, describe_visual
from scripts.export_research_evidence import finalize


class DashboardStoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = ArtifactStore()
        cls.data = cls.store.workspace()

    def test_all_logic_exports_preserve_premises_and_countermodels(self):
        ns = {'s': 'http://www.w3.org/2000/svg'}
        for case in self.data['cases']:
            svg = ET.fromstring(logical_svg(self.store, case['sample_id']))
            metadata = json.loads(svg.find('s:metadata', ns).text)
            self.assertEqual(metadata['case'], case)
            self.assertFalse(metadata['neural_trace'])
            text = ' '.join(svg.itertext())
            self.assertIn('not a measured neural reasoning trace', text)
            self.assertIn(case['ground_truth_label'], text)
            if case['symbolic_status'] == 'underdetermined':
                self.assertEqual({o['query_holds'] for o in metadata['case']['reasoning']['compatible_orderings']}, {True, False})

    def test_featured_wrong_answers_cannot_be_presented_as_improvement(self):
        case = next(c for c in self.data['cases'] if c['sample_id'] == 'xc_ho_0011::underdetermined')
        self.assertEqual(comparison_takeaway(case), 'All four checkpoints answered this case incorrectly.')
        svg = ET.fromstring(comparison_svg(self.store, case['sample_id']))
        metadata = json.loads(svg.find('{http://www.w3.org/2000/svg}metadata').text)
        self.assertEqual(metadata['predictions']['ce50']['raw_completion'], ' No' + '\ufffd' * 15)
        self.assertIn('Internal neural measurements have not yet been recorded', ' '.join(svg.itertext()))

    def test_gallery_is_escaped_offline_and_explained(self):
        paths=['02_Visual_Inputs/xc_ho_0011__underdetermined.png',
               '04_Logical_Evidence_Graphs/xc_ho_0011__underdetermined_Symbolic_Graph.png',
               '03_Model_Output_Comparisons/xc_ho_0011__underdetermined_Input_Output.png',
               '05_Neural_Activation_Analysis/Probe_Trajectory_last.png',
               '06_Experiment_Analytics/Confusion_Matrix_es5.png']
        catalog={p:{'sample_id':'xc_ho_0011::underdetermined','condition_id':'es5','classification':'OBSERVED','source_reference':['<script>untrusted</script>']} for p in paths}
        original=copy.deepcopy(self.data)
        html=render_gallery(catalog,self.data)
        self.assertNotIn('<script>',html)
        self.assertNotIn('https://',html)
        self.assertIn('&lt;script&gt;untrusted&lt;/script&gt;',html)
        self.assertIn('All four standard-prompt checkpoints answered No',html)
        self.assertIn('different Transformer',html)
        self.assertEqual(self.data,original)
        for path,info in catalog.items():
            title,description,limits=describe_visual(path,info,self.data)
            self.assertNotIn('.png',title)
            self.assertGreater(len(description),80)
            self.assertGreater(len(limits),80)

    def test_finalized_release_rejected_before_any_write(self):
        # Use checkout-local temp storage; a collision must not mutate old evidence.
        cache=Path(__file__).resolve().parents[1]/'.cache'
        with tempfile.TemporaryDirectory(dir=cache) as directory:
            out=Path(directory)/'release';out.mkdir()
            manifest=out/'MANIFEST.json';manifest.write_bytes(b'old exact evidence')
            with self.assertRaisesRegex(ValueError,'preserved'):finalize(out)
            self.assertEqual(manifest.read_bytes(),b'old exact evidence')
            self.assertEqual(list(out.iterdir()),[manifest])


if __name__ == '__main__':
    unittest.main()

