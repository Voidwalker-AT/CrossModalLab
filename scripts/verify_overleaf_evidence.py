"""Verify the new paper-support package and optionally create a fresh evidence ZIP."""
import argparse,csv,hashlib,json,math,re,subprocess,sys,zipfile
from pathlib import Path
from uuid import uuid4
import xml.etree.ElementTree as ET
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
try:
    from scripts.build_overleaf_evidence import recompute,same_metrics,json_write
except ModuleNotFoundError:
    from build_overleaf_evidence import recompute,same_metrics,json_write

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def structure(text):
    text=re.sub(r'(?m)(?<!\\)%.*$','',text)
    stack=[]
    for char in re.finditer(r'(?<!\\)[{}]',text):
        if char[0]=='{':stack.append(char.start())
        elif not stack:raise ValueError('unmatched closing brace')
        else:stack.pop()
    if stack:raise ValueError('unmatched opening brace')
    env=[]
    for kind,name in re.findall(r'\\(begin|end)\{([^}]+)\}',text):
        if kind=='begin':env.append(name)
        elif not env or env.pop()!=name:raise ValueError('LaTeX environment mismatch')
    if env:raise ValueError('unclosed LaTeX environment')

def verify(package,original_sources=True):
    d=json.loads((package/'audit/recomputed.json').read_text(encoding='utf-8'));raw=json.loads((package/'audit/raw_predictions.json').read_text(encoding='utf-8'))
    assert same_metrics({k:recompute([r for r in raw if r['condition']==k])for k in d['conditions']},d['conditions'])
    forbidden={'.pt','.bin','.safetensors','.npz','.npy','.zip','.exe','.dll'}
    files=[p for p in package.rglob('*')if p.is_file()and '__pycache__'not in p.parts]
    assert not any(p.suffix in forbidden for p in files),'binary/cache/weight prohibited'
    assert not any(p.stat().st_size>20*1024**2 for p in files),'unexpected oversized package member'
    for p in files:
        if p.suffix=='.json':json.loads(p.read_text(encoding='utf-8'))
        if p.suffix=='.csv':
            with p.open(encoding='utf-8',newline='')as f:
                reader=csv.DictReader(f);assert reader.fieldnames
                rows=list(reader);assert rows and all(None not in r and all(v is not None for v in r.values())for r in rows),p
        if p.suffix in ('.tex','.bib'):structure(p.read_text(encoding='utf-8'))
        if p.suffix=='.png':
            with Image.open(p)as im:im.verify()
        if p.suffix=='.svg':assert ET.parse(p).getroot().tag.endswith('svg')
    refs=(package/'references/references_verified.bib').read_text(encoding='utf-8');keys=re.findall(r'@\w+\{([^,]+),',refs)
    assert len(keys)==27 and len(set(keys))==27
    for p in package.rglob('*.tex'):
        for group in re.findall(r'\\cite\{([^}]+)\}',p.read_text(encoding='utf-8')):assert set(group.split(','))<=set(keys)
    cases=json.loads((package/'audit/development_cases.json').read_text())
    for c in cases:assert sha(package/'figures/source_images'/str(c['sample_id'].replace('::','__')+'.png'))==c['image_sha256']
    figures=json.loads((package/'figures/figure_manifest.json').read_text(encoding='utf-8'));assert len(figures)==12
    for name,f in figures.items():
        for file,h in f['files'].items():assert sha(package/'figures'/file)==h
        for file,h in f['source_data_sha256'].items():assert sha(package/file)==h
        with Image.open(package/'figures'/str(name+'.png'))as im:
            assert min(im.info['dpi'])>=300
            assert abs(im.width/320-f['width_inches'])<.01
    conditions=['frozen','prompt','ce5','es5','ce50'];m=d['conditions'];labs=['Yes','No','Undetermined']
    close=lambda a,b:math.isclose(a,b,rel_tol=0,abs_tol=1e-12)
    bars=figures['04_checkpoint_comparison']['plotted_artists'][0]['bars']
    assert all(close(p['height'],m[k]['balanced_accuracy'])for p,k in zip(bars,conditions))and len(bars)==5
    bars=figures['05_class_recall']['plotted_artists'][0]['bars'];expected=[m[k]['recall'][l]for l in labs for k in conditions]
    assert len(bars)==15 and all(close(p['height'],v)for p,v in zip(bars,expected))
    for ax,k in zip(figures['06_confusion_matrices']['plotted_artists'],conditions):assert ax['matrices'][0]==[[m[k]['confusion_matrix'][a][b]for b in [*labs,'Invalid']]for a in labs]
    bars=figures['07_format_vs_correctness']['plotted_artists'][0]['bars'];expected=[m[k]['correct']/18 for k in conditions]+[m[k]['exact_format']/18 for k in conditions]
    assert len(bars)==10 and all(close(p['height'],v)for p,v in zip(bars,expected))
    for i,key in enumerate(['ce_loss','es_loss','total_loss']):assert [p[1]for p in figures['08_five_step_losses']['plotted_artists'][0]['scatter'][i]]==[r[key]for r in d['training']]
    for i,key in enumerate(['torch_max_allocated_mib','torch_max_reserved_mib']):assert [p[1]for p in figures['09_gpu_memory']['plotted_artists'][0]['scatter'][i]]==[r['memory'][key]for r in d['training']]
    assert figures['10_historical_smol_probes']['plotted_artists'][0]['lines'][0]['y']==[p['balanced_accuracy']for p in d['probe']]
    neural=json.loads((package/'audit/neural_evidence_checked.json').read_text())
    assert [p[1]for p in figures['11_historical_activation_norms']['plotted_artists'][0]['scatter'][0]]==[p['l2_norm']for p in neural['captures'][0]['points']if p['selector']=='last']
    shortcut_keys=['majority','text_only_oracle_roles','image_only_oracle_roles','role_assignment','prompt_template','entity_identity','position_only','combined_nonsemantic']
    assert [p['width']for p in figures['12_shortcut_diagnostic']['plotted_artists'][0]['bars']]==[d['shortcut_audit']['baselines'][k]['in_sample_balanced_accuracy_ceiling']for k in shortcut_keys[::-1]]
    def csv_rows(name):
        with (package/'tables'/name).open(encoding='utf-8',newline='')as f:return list(csv.DictReader(f))
    for row in csv_rows('c_behavioral_results.csv'):
        s=m[row['condition']]
        assert row['correct']==f"{s['correct']}/18"and row['BA']==f"{s['balanced_accuracy']:.6f}"and row['macro_F1']==f"{s['macro_f1']:.6f}"
        assert row['recall_Y_N_U']=='/'.join(str(s['tp'][l])+'/6'for l in labs)
    for row in csv_rows('d_format_semantics.csv'):
        s=m[row['condition']]
        assert row['semantic_invalid']==f"{s['semantic_invalid']}/18"and row['exact_label']==f"{s['exact_format']}/18"
    for row,record in zip(csv_rows('f_five_step_telemetry.csv'),d['training']):
        assert float(row['CE'])==record['ce_loss']and float(row['ES'])==record['es_loss']and float(row['total'])==record['total_loss']
        assert float(row['peak_allocated_MiB'])==record['memory']['torch_max_allocated_mib']
    for row,record in zip(csv_rows('g_smol_probes.csv'),d['probe']):
        assert float(row['BA'])==record['balanced_accuracy']and float(row['CI_low'])==record['ci']['ci_low']and float(row['CI_high'])==record['ci']['ci_high']
    historical=json.loads((package/'manifest/protected_sources_before.json').read_text());intended_changes={'.gitignore','.gitattributes'}
    if original_sources:
        for manifest in ['manifest/source_hashes.json','manifest/supplementary_source_hashes.json']:
            for path,h in json.loads((package/manifest).read_text()).items():assert sha(ROOT/path)==h,path
        for path,h in historical.items():
            if path not in intended_changes:assert sha(ROOT/path)==h,'Protected historical file changed: '+path
    else:
        for path,h in json.loads((package/'manifest/package_manifest.json').read_text())['files'].items():assert sha(package/path)==h,path
    report={'status':'PASS','raw_prediction_records':len(raw),'conditions':6,'figures_png_svg':12,'original_images':18,'table_families_csv_md_tex':len(list((package/'tables').glob('*.csv'))),'bibliography_entries':27,'protected_source_hashes_checked':len(historical)-len(intended_changes)if original_sources else 0,'original_repository_sources':'CHECKED'if original_sources else 'NOT AVAILABLE / not revalidated in portable mode','no_new_model_forwards':True,'no_optimizer_steps':True,'latex_status':'BLOCKED compiler sandbox; structural checks only','replication_status':'BLOCKED first RAM admission; zero attempts','figure_to_raw_data':'Bar heights, confusion cells, loss/memory scatter and probe values checked against recomputed JSON','historical_probe_CI':'Stored provenance/bounds only; not newly resampled','preservation_exceptions':sorted(intended_changes),'archive_state':'Archive checks recorded separately after ZIP creation'}
    return report,files

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,default=ROOT/'paper/overleaf_evidence_20261025');p.add_argument('--create-zip',type=Path);p.add_argument('--portable',action='store_true',help='Validate extracted package content, without original repository sources');args=p.parse_args();package=args.package.resolve()
    if not package.is_relative_to(ROOT):raise ValueError('package outside workspace')
    result,files=verify(package,not args.portable)
    if not args.portable:json_write(package/'audit/verification.json',result)
    if args.portable and args.create_zip:raise ValueError('portable validation does not authorize creating a new distribution')
    if args.create_zip:
        archive=args.create_zip.resolve()
        if not archive.is_relative_to(ROOT)or archive.exists():raise ValueError('fresh workspace ZIP required; previous archive preserved')
        manifest={'schema':'overleaf_evidence_v1','source_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'files':{p.relative_to(package).as_posix():sha(p)for p in package.rglob('*')if p.is_file()and '__pycache__'not in p.parts and p.name!='package_manifest.json'},'limitations':['No new GPU replication; first RAM gate blocked.','LaTeX structural checks only; compiler sandbox blocked.','Exact Overleaf source unavailable; repository draft audited.']}
        json_write(package/'manifest/package_manifest.json',manifest)
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6)as z:
            for f in sorted(package.rglob('*')):
                if f.is_file()and '__pycache__'not in f.parts:z.write(f,f.relative_to(package).as_posix())
        extracted=ROOT/'.cache/overleaf_evidence'/('verified-extraction-'+uuid4().hex)
        with zipfile.ZipFile(archive)as z:
            assert z.testzip()is None
            assert len(z.namelist())==len(set(z.namelist()))
            assert all(not Path(n).is_absolute()and '..'not in Path(n).parts for n in z.namelist())
            z.extractall(extracted)
            for name,h in manifest['files'].items():assert sha(extracted/name)==h,name
            assert sha(extracted/'manifest/package_manifest.json')==sha(package/'manifest/package_manifest.json')
            inventory=z.namelist()
        # Parse/decode extracted assets, not just original members.
        for f in extracted.rglob('*.png'):
            with Image.open(f)as im:im.verify()
        for f in extracted.rglob('*.svg'):ET.parse(f)
        receipt={'status':'PASS','zip':archive.relative_to(ROOT).as_posix(),'sha256':sha(archive),'bytes':archive.stat().st_size,'members':len(inventory),'crc':'PASS','extraction_hashes':'PASS','extracted_png_svg_decode':'PASS','complete_original_manuscript_compiled':False,'new_model_forwards':0,'new_optimizer_steps':0}
        json_write(archive.with_suffix('.verification.json'),receipt);print(json.dumps(receipt))
    else:print(json.dumps(result))

if __name__=='__main__':main()
