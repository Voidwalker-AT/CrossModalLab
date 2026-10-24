import ast, hashlib, json, re
from pathlib import Path
from urllib.parse import unquote
root=Path.cwd(); paper=root/'paper'
for p in paper.rglob('*'):
 if not p.is_file(): continue
 assert p.stat().st_size>0 and p.resolve().is_relative_to(root)
 if p.suffix=='.md':
  text=p.read_text(encoding='utf-8')
  assert text.count('```')%2==0, p
  for target in re.findall(r'\]\(([^)]+)\)',text):
   if '://' not in target and not target.startswith('#'):
    assert (p.parent/unquote(target.split('#')[0])).exists(), (p,target)
 if p.suffix=='.py': ast.parse(p.read_text(encoding='utf-8'))
 if p.suffix=='.json': json.loads(p.read_text())
for p in (root/'artifacts/training/phase_2i_b').rglob('*.json'): json.loads(p.read_text())
a=json.loads((root/'artifacts/training/phase_2i_b_runner_recovery/9833770dd2f04040a84b54a3133206d5/validated_development_plan.json').read_text())
for name, expected in a['historical_immutability_sha256'].items():
 p=root/Path(name); assert hashlib.sha256(p.read_bytes()).hexdigest()==expected, name
print('PAPER_FILES_AND_LOCAL_LINKS_PASS; PILOT_JSON_PASS; HISTORICAL_UNCHANGED',len(a['historical_immutability_sha256']))
# Numerical table generation is deterministic, with fixed seeds and source hashes.
import subprocess
files=[paper/'tables/development_summary.json',paper/'tables/family_predictions.csv']
before=[p.read_bytes() for p in files]
subprocess.run([__import__('sys').executable,str(paper/'tables/build_development_tables.py')],check=True,stdout=subprocess.DEVNULL)
assert before==[p.read_bytes() for p in files]
print('TABLE_REBUILD_BYTE_IDENTICAL_PASS')
