import base64
import copy
import hashlib
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
from dashboard.backend.artifacts import ArtifactStore
from dashboard.backend.exports import comparison_svg
from dashboard.backend.neural import validate
from dashboard.register_wsctl import registration_bytes


class NeuralExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = ArtifactStore()
        cls.bundle = json.loads((cls.store.root / 'dashboard/assets/neural_evidence_v1.json').read_text())

    def test_validated_measurements_are_separate_from_qwen(self):
        bundle = validate(self.bundle)
        self.assertTrue(bundle['available'])
        self.assertEqual(bundle['validation']['activation_rows'], 200)
        self.assertNotEqual(bundle['model']['id'], self.store.workspace()['model']['id'])
        self.assertFalse(bundle['causal_audit']['causal_mechanism_established'])

    def test_misattribution_and_synthetic_rejected(self):
        for key,value in [('is_synthetic_demo',True),('schema','untrusted')]:
            bundle=copy.deepcopy(self.bundle);bundle[key]=value
            with self.assertRaises(ValueError):validate(bundle)
        bundle=copy.deepcopy(self.bundle);bundle['model']['id']='Qwen/Qwen3-VL-2B-Instruct'
        with self.assertRaisesRegex(ValueError,'identity'):validate(bundle)

    def test_nonfinite_and_causal_upgrade_rejected(self):
        bundle=copy.deepcopy(self.bundle);bundle['captures'][0]['points'][0]['l2_norm']=float('nan')
        with self.assertRaisesRegex(ValueError,'norm'):validate(bundle)
        bundle=copy.deepcopy(self.bundle);bundle['causal_audit']['causal_mechanism_established']=True
        with self.assertRaisesRegex(ValueError,'causal'):validate(bundle)

    def test_all_svg_images_and_raw_predictions_are_exact(self):
        ns={'s':'http://www.w3.org/2000/svg'}
        for case in self.store.workspace()['cases']:
            svg=ET.fromstring(comparison_svg(self.store,case['sample_id']))
            image=svg.find('s:image',ns)
            raw=base64.b64decode(image.attrib['href'].split(',',1)[1])
            self.assertEqual(hashlib.sha256(raw).hexdigest(),case['image_sha256'])
            metadata=json.loads(svg.find('s:metadata',ns).text)
            self.assertEqual(metadata['prompt'],case['prompt'])
            self.assertEqual(set(metadata['predictions']),{'frozen','ce5','ce50','es5'})
            for key,prediction in metadata['predictions'].items():
                self.assertEqual(prediction['raw_completion'],case['predictions'][key]['raw_completion'])
            self.assertEqual(svg.findall('.//s:script',ns),[])

    def test_comparison_api_is_allowlisted_and_read_only(self):
        import threading
        from urllib.error import HTTPError
        from urllib.request import Request, urlopen
        from dashboard.backend.server import make_server
        server=make_server(0,store=self.store)
        worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        base=f'http://127.0.0.1:{server.server_port}'
        try:
            with urlopen(base+'/api/comparison/xc_ho_0011%3A%3Aunderdetermined.svg') as response:
                self.assertIn('image/svg+xml',response.headers['Content-Type'])
                self.assertIn(b'Query only:',response.read())
            for path in ['/api/comparison/not-allowlisted.svg','/api/comparison/..%2fREADME.md.svg']:
                with self.assertRaises(HTTPError) as error:urlopen(base+path)
                self.assertEqual(error.exception.code,404)
            with self.assertRaises(HTTPError) as error:urlopen(Request(base+'/api/comparison/x.svg',data=b'{}',method='POST'))
            self.assertEqual(error.exception.code,405)
        finally:
            server.shutdown();server.server_close();worker.join(timeout=5)

    def test_registration_preserves_existing_bytes_and_objects(self):
        raw=b'{\r\n  "schemaVersion": 2, "other": "unchanged",\r\n  "projects": [{"id":"existing","path":"Academic/example"}]\r\n}\r\n'
        entry={'id':'crossmodallab','name':'CrossModalLab'}
        changed,before=registration_bytes(raw,entry)
        self.assertTrue(changed.startswith(raw.split(b']')[0]))
        self.assertEqual(json.loads(changed)['projects'][:-1],before['projects'])
        self.assertEqual(registration_bytes(changed,entry)[0],changed)
        with self.assertRaisesRegex(ValueError,'differs'):
            registration_bytes(changed,{'id':'crossmodallab','name':'unexpected replacement'})


if __name__ == '__main__':unittest.main()
