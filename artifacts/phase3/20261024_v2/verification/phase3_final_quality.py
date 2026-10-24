import ast
import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote
from PIL import Image, ImageChops

root=Path.cwd()
sys.path.insert(0,str(root))
from src.evaluation.research_audit import verify_partial_pilot

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
bundle=json.loads((root/'dashboard/assets/neural_evidence_v1.json').read_text())
for item in bundle['sources']:
    assert sha(root/item['path'])==item['sha256'],item['path']
plan=json.loads((root/'artifacts/phase3/20261024_v2/pilot/preregistered_plan.json').read_text())
for name,digest in plan['source_sha256'].items():assert sha(root/name)==digest,name
assert verify_partial_pilot(root/'artifacts/phase3/20261024_v2/pilot')['complete_cross_model'] is False
for name in ('analysis.json','historical_cross_model.csv'):
    assert (root/'paper/tables/phase3_20261024'/name).read_bytes()==(root/'.cache/phase3_tables_rebuild_final'/name).read_bytes(),name
for image in (root/'paper/figures/phase3_20261024').glob('*.png'):
    with Image.open(image) as original, Image.open(root/'.cache/phase3_figures_rebuild_final'/image.name) as rebuilt:
        assert original.size==rebuilt.size and ImageChops.difference(original.convert('RGB'),rebuilt.convert('RGB')).getbbox() is None,image
archive=root/'submission/CrossModalLab_Research_Evidence_Oct28_HumanCentered_20261024_r3.zip'
assert sha(archive)=='24cfd58d51640ed1baa6fb3696dbd0d6f22f4cec13e18c973b1a07f0f2ae1283'
for name in ('docs/research_progress_audit.md','docs/publication_gap_analysis.md','docs/phase_3_research_plan.md','docs/phase_3_dataset_audit.md','docs/phase_3_evaluation_protocol.md','docs/phase_3_intervention_protocol.md','docs/phase_3_training_feasibility.md','docs/phase_3_novelty_audit.md','README.md','paper/submission_plan.md'):
    path=root/name;text=path.read_text(encoding='utf-8')
    assert text and text.count('```')%2==0,name
    for target in re.findall(r'\]\(([^)]+)\)',text):
        if '://' not in target and not target.startswith('#'):
            assert (path.parent/unquote(target.split('#')[0])).exists(),(name,target)
for path in (root/'artifacts/phase3/20261024_v2').rglob('*.json'):
    json.loads(path.read_text(encoding='utf-8'))
print(json.dumps({'status':'PASS','neural_source_hashes_unchanged':len(bundle['sources']),'pilot_executed_source_hashes_unchanged':len(plan['source_sha256']),'tables_byte_identical':True,'figure_png_pixels_identical':True,'oct28_archive_unchanged':True,'new_document_links_valid':True,'pilot_status':'VERIFIED_PARTIAL_EVIDENCE; not complete cross-model'}))
