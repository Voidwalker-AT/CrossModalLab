import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.launch_dashboard import build_fingerprint, ensure_frontend
from src.config.paths import get_temp_dir


class DashboardLauncherTests(unittest.TestCase):
    def setUp(self):
        folder=get_temp_dir();folder.mkdir(parents=True,exist_ok=True)
        self.scratch=tempfile.TemporaryDirectory(dir=folder);self.addCleanup(self.scratch.cleanup)
        self.root=Path(self.scratch.name);self.front=self.root/'dashboard/frontend'
        (self.front/'src').mkdir(parents=True)
        for name in ('package.json','package-lock.json','tsconfig.json','vite.config.ts','index.html'):
            (self.front/name).write_text('{}')
        (self.front/'src/main.tsx').write_text('export const version=1;')

    def test_source_change_invalidates_fingerprint(self):
        old=build_fingerprint(self.root)
        (self.front/'src/main.tsx').write_text('export const version=2;')
        self.assertNotEqual(old,build_fingerprint(self.root))

    def test_current_production_bundle_needs_no_node(self):
        (self.front/'dist/assets').mkdir(parents=True)
        (self.front/'dist/index.html').write_text('<html></html>')
        (self.front/'dist/assets/app.js').write_text('console.log(1)')
        (self.root/'.cache').mkdir()
        (self.root/'.cache/dashboard-build.json').write_text(json.dumps({'input_sha256':build_fingerprint(self.root)}))
        with patch('scripts.launch_dashboard.shutil.which',side_effect=AssertionError('must not invoke a tool')):
            ensure_frontend(self.root)

    def test_missing_build_cannot_install_implicitly(self):
        with patch('scripts.launch_dashboard.shutil.which',return_value=None),patch('scripts.launch_dashboard.subprocess.run') as runner:
            with self.assertRaisesRegex(RuntimeError,'nothing was installed'):
                ensure_frontend(self.root)
            runner.assert_not_called()

    def test_missing_referenced_css_requires_rebuild(self):
        (self.front/'dist/assets').mkdir(parents=True)
        (self.front/'dist/index.html').write_text('<script src="/assets/app.js"></script><link href="/assets/app.css" rel="stylesheet">')
        (self.front/'dist/assets/app.js').write_text('console.log(1)')
        (self.root/'.cache').mkdir()
        (self.root/'.cache/dashboard-build.json').write_text(json.dumps({'input_sha256':build_fingerprint(self.root)}))
        with patch('scripts.launch_dashboard.shutil.which',return_value=None):
            with self.assertRaisesRegex(RuntimeError,'nothing was installed'):
                ensure_frontend(self.root)


if __name__=='__main__':unittest.main()
