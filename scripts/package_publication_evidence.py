"""Build a small owned research-review bundle, never a model/holdout archive."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {'.md', '.html', '.py', '.json', '.jsonl', '.csv', '.png', '.svg', '.txt'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve(); archive = output.with_suffix('.zip')
    if not output.is_relative_to(ROOT / 'submission') or output.exists() or archive.exists():
        raise ValueError('fresh submission-local folder and ZIP required')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    selected = set()
    for directory in ('paper', 'artifacts/publication/20261025'):
        for path in (ROOT / directory).rglob('*'):
            if path.is_file() and path.suffix in ALLOWED and '__pycache__' not in path.parts:
                selected.add(path.relative_to(ROOT))
    for path in (ROOT / 'docs').glob('publication*.md'): selected.add(path.relative_to(ROOT))
    selected.add(Path('docs/international_research_standards.md'))
    for pattern in ('*publication*.py', 'test_matched_screen.py'):
        for directory in ('scripts', 'tests'):
            selected.update(p.relative_to(ROOT) for p in (ROOT / directory).glob(pattern))
    selected.update(map(Path, ('src/evaluation/matched_screen.py', 'src/evaluation/screen_budget.py',
                               'src/benchmark/publication_factorial.py', 'src/causal/publication_controls.py')))
    if not selected: raise ValueError('empty evidence selection')
    changed = set(subprocess.check_output(['git', 'diff', 'HEAD', '--name-only'], cwd=ROOT, text=True).splitlines())
    changed.update(subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'], cwd=ROOT, text=True).splitlines())
    dirty_selection = bool(changed & {p.as_posix() for p in selected})
    for relative in selected:
        path = ROOT / relative
        if not path.resolve().is_relative_to(ROOT) or any(p in {'.cache', 'checkpoints', 'node_modules'} for p in relative.parts):
            raise ValueError('unsafe selection')
        if path.suffix not in ALLOWED or path.stat().st_size > 20 * 1024**2:
            raise ValueError('unexpected binary or oversized file')
    output.mkdir(parents=True)
    repo = 'https://github.com/Voidwalker-AT/CrossModalLab/blob/' + revision + '/'
    def rewrite_target(target, relative):
        if '://' in target or target.startswith('#'): return target
        target_path = (ROOT / relative.parent / unquote(target.split('#')[0])).resolve()
        if not target_path.is_relative_to(ROOT): raise ValueError('link escapes workspace')
        dest = target_path.relative_to(ROOT)
        if dest in selected: return target
        return repo + dest.as_posix() + ('#' + target.split('#', 1)[1] if '#' in target else '')
    files = {}
    for relative in sorted(selected):
        source = ROOT / relative; destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        raw = source.read_bytes()
        if source.suffix == '.md':
            raw = re.sub(r'\]\(([^)]+)\)', lambda m: '](' + rewrite_target(m[1], relative) + ')', raw.decode('utf-8')).encode('utf-8')
        elif source.suffix == '.html':
            raw = re.sub(r'href="([^"]+)"', lambda m: 'href="' + rewrite_target(m[1], relative) + '"', raw.decode('utf-8')).encode('utf-8')
        destination.write_bytes(raw)
        files[relative.as_posix()] = {'sha256': hashlib.sha256(raw).hexdigest(),
                                     'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                                     'bytes': len(raw)}
    readme = ('# CrossModalLab publication evidence — October 25, 2026\n\n'
              'Start with paper/manuscript/draft.html and paper/prior_art_evidence_matrix.md. '
              'Own graphs are in paper/figures/publication_20261025/. '
              'Successful pilot: 12 forwards/two cases/one family. Separate failed screen: '
              'one attempted forward/zero scores. No causal or independent-generalization claim.\n\n'
              'This is a research review bundle, not a complete runnable environment. Source base revision: `'
              + revision + '`. Selected sources differ from that commit: ' + str(dirty_selection) + '. '
              'Exact source-file hashes are in the manifest. Use the repository and docs/publication_reproducibility_20261025.md '
              'for CPU reproduction. No GPU retry is authorized by this package. '
              'External links require network access; manuscript figures are local. '
              'The frozen October 28 package is unchanged.\n')
    (output / 'README.md').write_text(readme, encoding='utf-8')
    files['README.md'] = {'sha256': hashlib.sha256((output / 'README.md').read_bytes()).hexdigest()}
    manifest = {'schema': 'publication_review_bundle_v1', 'source_revision': revision, 'source_selection_dirty': dirty_selection, 'files': files,
                'excluded': ['weights', 'optimizer binaries', 'caches', 'external photos', 'source PDFs/full text', 'credentials', 'closed holdout'],
                'scope': 'Review bundle. Markdown external local links rewritten to declared source revision; own source hashes retained.'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(output.rglob('*')):
            if path.is_file(): zipped.write(path, path.relative_to(output).as_posix())
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None: raise ValueError('archive CRC failure')
        for name, row in files.items():
            if hashlib.sha256(zipped.read(name)).hexdigest() != row['sha256']:
                raise ValueError('archive content mismatch')
    report = {'status': 'PASS', 'source_revision': revision, 'source_selection_dirty': dirty_selection, 'files': len(files) + 1,
              'archive_bytes': archive.stat().st_size, 'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
              'archive': archive.relative_to(ROOT).as_posix(), 'member_hashes_verified': True,
              'manifest_sha256': hashlib.sha256((output / 'manifest.json').read_bytes()).hexdigest()}
    public = ROOT / 'artifacts/publication/20261025' / (output.name + '_verification.json')
    if public.exists(): raise ValueError('public package report already exists')
    public.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__': main()
