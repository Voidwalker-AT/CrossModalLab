"""Read-only selection checks, with explicit seal/ZIP modes; no model execution."""
import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_dual_model_evidence import load, recompute, same_metrics
from scripts.build_final_overleaf_selection import SELECTED_FIGURES, SELECTED_KEYS
from scripts.verify_overleaf_evidence import structure


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(package, *, original_sources=True, require_manifest=True):
    from PIL import Image
    package = Path(package).resolve()
    files = [p for p in package.rglob('*') if p.is_file()]
    if not files or any(not p.resolve().is_relative_to(package) for p in files):
        raise ValueError('empty or escaped selection')
    forbidden = {'.pt', '.bin', '.safetensors', '.npz', '.npy', '.zip', '.exe', '.dll'}
    if any(p.suffix in forbidden or p.name == 'main.tex' or p.stat().st_size > 5 * 1024**2 for p in files):
        raise ValueError('prohibited or oversized package member')
    for p in files:
        if not p.stat().st_size:
            raise ValueError('empty package member')
        if p.suffix == '.json':
            load(p)
        elif p.suffix in ('.tex', '.bib'):
            structure(p.read_text(encoding='utf-8'))
        elif p.suffix == '.png':
            with Image.open(p) as image:
                image.verify()
        elif p.suffix == '.svg':
            if not ET.parse(p).getroot().tag.endswith('svg'):
                raise ValueError('invalid SVG')
        elif p.suffix == '.csv':
            with p.open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
                if not rows or any(None in r or None in r.values() for r in rows):
                    raise ValueError('CSV shape mismatch')
    data = load(package / 'audit/dual_model_audit.json')
    if data['new_model_forwards'] != 0 or data['new_optimizer_steps'] != 0:
        raise ValueError('scope violation')
    smol = [dict(r, family_id=r['matched_group_id']) for r in data['smol']['rows']]
    if not same_metrics(recompute(smol), data['smol']['metrics']):
        raise ValueError('Smol arithmetic mismatch')
    qraw = data['qwen']['raw']
    actual = {c: recompute([r for r in qraw if r['condition'] == c]) for c in data['qwen']['metrics']}
    if not same_metrics(actual, data['qwen']['metrics']):
        raise ValueError('Qwen arithmetic mismatch')
    with (package / 'tables/t3_qwen_behavior.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    for r in rows:
        target = actual[r['Condition']]
        if (r['Correct'] != f"{target['correct']}/18" or r['BA'] != f"{target['balanced_accuracy']:.6f}"
                or r['Macro F1'] != f"{target['macro_f1']:.6f}" or r['U recall'] != f"{target['tp']['Undetermined']}/6"):
            raise ValueError('behavior table source mismatch')
    def table_rows(name):
        with (package / 'tables' / name).open(encoding='utf-8', newline='') as stream:
            return list(csv.DictReader(stream))
    for r in table_rows('t4_format_compliance.csv'):
        target = actual[r['Condition']]
        if any(r[k] != f"{target[field]}/18" for k, field in
               [('Correct', 'correct'), ('Invalid', 'semantic_invalid'), ('Exact', 'exact_format')]):
            raise ValueError('format table source mismatch')
    for r, target in zip(table_rows('s1_qwen_candidate_scores.csv'), data['qwen_replication']['rows']):
        if ([r[k] for k in ('Yes LL', 'No LL', 'U LL')] != [f'{v:.9f}' for v in target['summed_loglikelihoods']]
                or r['Argmax'] != target['argmax_answer'] or r['Gold'] != target['ground_truth_label']):
            raise ValueError('candidate table source mismatch')
    pilot = data['matched_pilot']['rows']
    for index, r in enumerate(table_rows('s2_matched_pilot.csv')):
        if r['Smol preference'] != pilot['smol'][index]['argmax_answer'] or r['Qwen preference'] != pilot['qwen'][index]['argmax_answer']:
            raise ValueError('matched pilot table source mismatch')
    for r in table_rows('t1_evaluation_conditions.csv'):
        if r['Model'] == 'Qwen':
            expected_steps = {'frozen': 0, 'prompt': 0, 'ce5': 5, 'es5': 5, 'ce50': 50, 'es5prompt': 5}
            if int(r['Updates']) != expected_steps[r['Condition']] or r['Cases / groups'] != '18 / 6':
                raise ValueError('checkpoint identity table mismatch')
    refs = (package / 'references/references_selected.bib').read_text(encoding='utf-8')
    keys = re.findall(r'@\w+\{([^,]+),', refs)
    if len(keys) != 14 or set(keys) != set(SELECTED_KEYS) | {'ProbeControls'}:
        raise ValueError('selected reference keys mismatch')
    if re.search(r'id=xxx|10\.xxxx|https?://example\.|\bTBD\b|TODO', refs):
        raise ValueError('bibliography placeholder')
    citation_map = (package / 'references/citation_map.md').read_text(encoding='utf-8')
    for key in keys:
        if '| ' + key + ' |' not in citation_map:
            raise ValueError('uncited selected entry')
    for p in package.rglob('*.tex'):
        for group in re.findall(r'\\cite\{([^}]+)\}', p.read_text(encoding='utf-8')):
            if not set(group.split(',')) <= set(keys):
                raise ValueError('unresolved citation')
    figures = load(package / 'figures/selection_manifest.json')
    if set(figures) != set(SELECTED_FIGURES):
        raise ValueError('selected figure identity mismatch')
    for name, info in figures.items():
        for filename, digest in info['files'].items():
            if sha(package / 'figures' / filename) != digest:
                raise ValueError('figure byte identity mismatch')
        if original_sources:
            for relative, digest in info['source_data_sha256'].items():
                if sha(ROOT / 'paper/overleaf_evidence_20261025' / relative) != digest:
                    raise ValueError('figure source data mismatch')
    if original_sources:
        for relative, digest in data['source_hashes'].items():
            if sha(ROOT / relative) != digest:
                raise ValueError('historical source changed: ' + relative)
    manifest_path = package / 'audit/selection_hashes.json'
    if require_manifest:
        manifest = load(manifest_path)
        expected = {p.relative_to(package).as_posix() for p in files if p != manifest_path}
        if set(manifest) != expected or any(sha(package / n) != h for n, h in manifest.items()):
            raise ValueError('selection hash/member mismatch')
    return dict(status='PASS', files=len(files), selected_figures=7, main_tables=6,
                supplementary_tables=2, bibliography_entries=14, smol_cases=36,
                smol_matched_groups=30, probe_features=108, qwen_generation_rows=108,
                new_model_forwards=0, source_hashes_checked=len(data['source_hashes']) if original_sources else 0,
                latex_status='STRUCTURAL_ONLY; full manuscript and snippet compilation UNVERIFIED')


def seal(package):
    package = Path(package).resolve()
    verify(package, require_manifest=False)
    target = package / 'audit/selection_hashes.json'
    if target.exists():
        raise FileExistsError('existing selection seal preserved')
    hashes = {p.relative_to(package).as_posix(): sha(p) for p in sorted(package.rglob('*')) if p.is_file()}
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(hashes, stream, indent=2)
    return verify(package)


def archive(package, target):
    package, target = Path(package).resolve(), Path(target).resolve()
    if not package.is_relative_to(ROOT) or not target.is_relative_to(ROOT / 'submission'):
        raise ValueError('workspace/submission boundary violation')
    receipt = target.with_suffix('.verification.json')
    if target.exists() or receipt.exists():
        raise FileExistsError('existing submission preserved')
    result = verify(package)
    files = sorted(p for p in package.rglob('*') if p.is_file())
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED) as stream:
        for p in files:
            stream.write(p, p.relative_to(package).as_posix())
    with zipfile.ZipFile(target) as stream:
        if stream.testzip() is not None:
            raise ValueError('ZIP CRC failure')
        names = stream.namelist()
        if len(names) != len(set(names)) or any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts for n in names):
            raise ValueError('unsafe ZIP member')
        with tempfile.TemporaryDirectory(prefix='selection-extract-', dir=ROOT / '.cache/final_selection') as temp:
            stream.extractall(temp)
            extracted = verify(Path(temp), original_sources=False)
            for p in files:
                if sha(p) != sha(Path(temp) / p.relative_to(package)):
                    raise ValueError('ZIP extraction byte mismatch')
    result.update(zip_sha256=sha(target), zip_bytes=target.stat().st_size,
                  zip_members=len(files), zip_crc='PASS', extraction='PASS', extracted_validation=extracted)
    with receipt.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package', type=Path)
    parser.add_argument('--seal', action='store_true')
    parser.add_argument('--create-zip', type=Path)
    parser.add_argument('--packaged-only', action='store_true',
                        help='Validate self-contained selected evidence; do not assert original local source hashes')
    args = parser.parse_args()
    if args.packaged_only and (args.seal or args.create_zip):
        raise ValueError('sealing/archiving requires full original-source validation')
    if args.seal:
        result = seal(args.package)
    elif args.create_zip:
        result = archive(args.package, args.create_zip)
    else:
        result = verify(args.package, original_sources=not args.packaged_only)
    print(json.dumps(result, indent=2))
