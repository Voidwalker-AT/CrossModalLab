"""Allowlisted publication update from stored development artifacts; no ML imports."""
import base64
import hashlib
import json
import math


def load(root):
    base=root/'artifacts/publication/20261025'
    analysis=base/'analysis.json';plan=base/'pilot/plan.json'
    if not analysis.is_file() or not plan.is_file():return {'available':False,'status':'NOT PACKAGED'}
    data=json.loads(analysis.read_text(encoding='utf-8'))
    declaration=json.loads(plan.read_text(encoding='utf-8'))
    if data['new_model_forwards']!=12 or data['optimizer_steps'] or data['holdout_accessed'] or data['causal_interventions']:
        raise ValueError('publication artifact scope mismatch')
    cases=[]
    sources=[analysis,plan]
    for case in declaration['cases']:
        if case['split']!='development' or case['is_synthetic_demo'] is not False:raise ValueError('publication case is not development evidence')
        image=(root/case['image_file']).resolve()
        if not image.is_relative_to(root.resolve()):raise ValueError('publication image outside checkout')
        raw=image.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=case['image_sha256']:raise ValueError('publication image hash mismatch')
        for model in data['models'].values():
            row=next(r for r in model['cases'] if r['sample_id']==case['sample_id'])
            if any(row[k]!=case[k] for k in ('ground_truth_label','prompt_sha256','image_sha256')) or not all(math.isfinite(v) for v in row['summed_loglikelihoods']):
                raise ValueError('publication score identity mismatch')
        sources.append(image)
        cases.append({**case,'image_data_url':'data:image/png;base64,'+base64.b64encode(raw).decode()})
    screen_path=base/'screen108/execution.json'
    screen={}
    if screen_path.is_file():
        failed=json.loads(screen_path.read_text(encoding='utf-8'))
        sources.append(screen_path)
        screen={'status':failed['status'],'attempts':failed['additional_forward_attempts'],
                'completed_rows':sum(len(p.read_text().splitlines()) for p in (base/'screen108').glob('*/candidate_scores.jsonl')),
                'note':'Separate approved 108-forward screen stopped at a resource gate. No retry; do not pool it with the completed two-case pilot.'}
    return {'available':True,'status':'OBSERVED BEHAVIOR','date':'2026-10-25','cases':cases,'models':data['models'],'screen':screen,
            'scope':'Two fixed cases in one exposed development family. Complete-answer likelihood scoring, not generated answers, calibrated confidence or causal explanation. No three-class BA or independent generalization claim.',
            'new_model_forwards':12,'sources':[{'path':p.relative_to(root).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in dict.fromkeys(sources)]}
