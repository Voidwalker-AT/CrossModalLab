"""Full-factorial development controls; no final-holdout reads or model calls."""
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import random

from src.benchmark.factorial_development import ALIASES
from src.benchmark.contrast_v2 import render_canvas
from src.config.paths import REPO_ROOT
from src.evaluation.research_audit import verified_label, balanced_accuracy, role_key
from src.symbolic.relations import Premise, Query
from src.symbolic.solver import SymbolicSolver

LABELS=('Yes','No','Undetermined')
SEED=2026102502


def build(out, *, families=6, seed=SEED):
    out=Path(out).resolve()
    if not out.is_relative_to(REPO_ROOT) or out.exists():raise ValueError('fresh checkout-local output required')
    if families<2 or families>12 or families%2:raise ValueError('even family count between 2 and 12 required')
    rng=random.Random(seed)
    triples=list(itertools.combinations(ALIASES,3));rng.shuffle(triples)
    out.mkdir(parents=True)
    rows=[];images={}
    for family_index,triple in enumerate(triples[:families]):
        family=f'pubdev_{seed}_{family_index:03d}'
        # Distinct triples stay grouped. Both partitions are exposed DEVELOPMENT.
        partition='baseline_fit' if family_index<families//2 else 'baseline_evaluation'
        for role_index,(a0,b0,c0) in enumerate(itertools.permutations(triple)):
            a,b,c=[ALIASES[x] for x in (a0,b0,c0)]
            for ab_forward,bc_forward,q_forward,visual_edge in itertools.product((0,1),repeat=4):
                ab=(a0,b0) if ab_forward else (b0,a0)
                bc=(b0,c0) if bc_forward else (c0,b0)
                visual,text=(ab,bc) if visual_edge==0 else (bc,ab)
                qs,qo=(a,c) if q_forward else (c,a)
                vp=tuple(ALIASES[x] for x in visual);tp=tuple(ALIASES[x] for x in text)
                premises=[{'subject':vp[0],'object':vp[1],'modality':'visual'},
                          {'subject':tp[0],'object':tp[1],'modality':'textual'}]
                query={'subject':qs,'object':qo,'relation':'left_of'}
                for layout,query_first,choices_reversed,content_reversed in itertools.product((0,1),repeat=4):
                    key=(family,visual,layout)
                    if key not in images:
                        image=out/'images'/f'{family_index:03d}_{list(triple).index(visual[0])}{list(triple).index(visual[1])}_l{layout}.png'
                        meta=render_canvas(*visual,image,random.Random(seed+family_index*100+layout),gap_scale=(.8 if layout==0 else 1.2))
                        images[key]={'image_file':image.relative_to(REPO_ROOT).as_posix(),'image_sha256':meta['image_sha256'],
                                     'position_features':tuple(v for box in sorted(meta['bboxes'].values()) for v in box)}
                    statement=f'The {tp[0].replace("_"," ")} is to the left of the {tp[1].replace("_"," ")}.'
                    question=f'Question: Is the {qs.replace("_"," ")} to the left of the {qo.replace("_"," ")}?'
                    choices='Undetermined, No, or Yes' if choices_reversed else 'Yes, No, or Undetermined'
                    prompt='\n\n'.join([question,statement] if query_first else [statement,question])+f'\nAnswer directly with: {choices}.'
                    status=SymbolicSolver.solve([Premise(**p) for p in premises],Query(**query),verify_with_oracle=True)
                    if not status.algorithm_agreement:raise ValueError('oracle disagreement')
                    label={'entailed':'Yes','contradicted':'No','underdetermined':'Undetermined'}[status.epistemic_status.value]
                    identity=f'r{role_index}a{ab_forward}b{bc_forward}q{q_forward}m{visual_edge}l{layout}p{query_first}c{choices_reversed}o{content_reversed}'
                    rows.append({'schema':'publication_factorial_v1','sample_id':f'{family}::{identity}','family_id':family,
                                 'split':'development','partition':partition,'is_synthetic_demo':False,
                                 'entities':{'A':a,'B':b,'C':c},'premises':premises,'query':query,
                                 'ground_truth_label':label,'oracle_agreement':True,'prompt':prompt,
                                 'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),**images[key],
                                 'factors':{'roles':role_index,'ab_forward':ab_forward,'bc_forward':bc_forward,'query_forward':q_forward,
                                            'visual_edge':visual_edge,'layout':layout,'query_first':query_first,
                                            'choices_reversed':choices_reversed,'content_reversed':content_reversed},
                                 'content_order':['text','image'] if content_reversed else ['image','text']})
    record_path=out/'records.jsonl'
    with record_path.open('x',encoding='utf-8',newline='\n') as stream:
        for row in rows:stream.write(json.dumps(row,separators=(',',':'),sort_keys=True)+'\n')
    manifest={'schema':'publication_factorial_v1','seed':seed,'families':families,'records':len(rows),'images':len(images),
              'status':'EXPOSED_DEVELOPMENT_NOT_CONFIRMATORY','records_sha256':hashlib.sha256(record_path.read_bytes()).hexdigest(),
              'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'family_partitions':{r['family_id']:r['partition'] for r in rows},
              'labels':dict(Counter(r['ground_truth_label'] for r in rows)),
              'frozen_holdout_accessed':False,'model_forwards':0,
              'independence_limit':'All variants within a triple are dependent. Shared entities/renderer/grammar remain; baseline evaluation is exposed development, never confirmatory.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
    return rows,manifest


def feature(row, name):
    f=row['factors']
    if name in ('text_only_oracle_roles','image_only_oracle_roles'):
        return role_key(row,'textual' if name.startswith('text') else 'visual')
    if name=='entity_identity':return tuple(sorted(row['entities'].values()))
    if name=='prompt_template':return (f['query_first'],f['choices_reversed'],f['content_reversed'])
    if name=='position_only':return tuple(row['position_features'])
    if name=='role_assignment':return (f['roles'],f['visual_edge'])
    if name=='combined_nonsemantic':return (feature(row,'entity_identity'),feature(row,'prompt_template'),feature(row,'position_only'),feature(row,'role_assignment'))
    if name=='symbolic_oracle':return verified_label(row)
    if name=='majority':return ()
    raise ValueError(name)


def shortcut_audit(rows):
    train=[r for r in rows if r['partition']=='baseline_fit']
    test=[r for r in rows if r['partition']=='baseline_evaluation']
    if not train or not test or {r['family_id'] for r in train}&{r['family_id'] for r in test}:raise ValueError('invalid family split')
    for row in rows:verified_label(row)
    names=('majority','text_only_oracle_roles','image_only_oracle_roles','role_assignment','prompt_template','entity_identity','position_only','combined_nonsemantic')
    results={}
    def majority(counts):return max(LABELS,key=lambda x:(counts[x],-LABELS.index(x)))
    fallback=majority(Counter(r['ground_truth_label'] for r in train))
    for name in names:
        counts=defaultdict(Counter)
        for r in train:counts[feature(r,name)][r['ground_truth_label']]+=1
        predictions=[majority(counts[feature(r,name)]) if feature(r,name) in counts else fallback for r in test]
        all_groups=defaultdict(Counter)
        for r in rows:all_groups[feature(r,name)][r['ground_truth_label']]+=1
        # Weighted class recall Bayes ceiling on inspected feature strata, not a fitted model result.
        support=Counter(r['ground_truth_label'] for r in rows)
        ceiling=sum(max(c[label]/support[label] for label in LABELS) for c in all_groups.values())/3
        results[name]={'balanced_accuracy':balanced_accuracy(test,predictions),'correct':sum(r['ground_truth_label']==p for r,p in zip(test,predictions)),
                       'n':len(test),'unseen_feature_rows':sum(feature(r,name) not in counts for r in test),
                       'in_sample_balanced_accuracy_ceiling':ceiling,'feature_groups':len(all_groups)}
    train_images={r['image_sha256'] for r in train};test_images={r['image_sha256'] for r in test}
    train_prompts={r['prompt_sha256'] for r in train};test_prompts={r['prompt_sha256'] for r in test}
    return {'baselines':results,'symbolic_oracle_balanced_accuracy':1.,
            'oracle_minus_best_single_modality_ba':1-max(results[n]['balanced_accuracy'] for n in ('text_only_oracle_roles','image_only_oracle_roles')),
            'fit_families':len({r['family_id'] for r in train}),'evaluation_families':len({r['family_id'] for r in test}),
            'cross_partition_image_hash_overlap':len(train_images&test_images),'cross_partition_prompt_hash_overlap':len(train_prompts&test_prompts),
            'interpretation':'Oracle-parsed role features are dataset diagnostics, not image/text VLM inference. Oracle gain establishes available information, not model composition. Nonsemantic ceilings diagnose predictability even when held-family lookups fall back.'}
