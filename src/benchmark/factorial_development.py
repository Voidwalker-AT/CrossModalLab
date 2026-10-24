"""Counterbalanced development task: both premise directions and query direction.

This creates NEW DEVELOPMENT evidence, never a final or confirmatory test set.
Mixed-pool entity triples cannot coincide with within-pool contrast-v2 families.
"""
import hashlib
import itertools
import json
import random
from pathlib import Path
from src.benchmark.contrast_v2 import ID_ENTITIES, HELD_OUT_ENTITIES, render_canvas
from src.config.paths import REPO_ROOT
from src.symbolic.relations import Premise, Query
from src.symbolic.solver import SymbolicSolver

SEED = 2026102403
ALIASES = dict(zip(ID_ENTITIES+HELD_OUT_ENTITIES,(
    'blue_rectangle','red_ellipse','green_rounded_rectangle','yellow_triangle',
    'purple_diamond','orange_ring','turquoise_rectangle','pink_hexagon',
    'gray_hexagon','amber_ring','teal_triangle','violet_diamond',
    'crimson_rectangle','green_ellipse','golden_ellipse','indigo_rounded_rectangle')))


def build_factorial(out, *, families=12, seed=SEED):
    out=Path(out).resolve()
    if not out.is_relative_to(REPO_ROOT) or out.exists():
        raise ValueError("fresh checkout-local output required; existing evidence preserved")
    if not 2 <= families <= 24:raise ValueError("bounded development family count required")
    rng=random.Random(seed);pool=ID_ENTITIES+HELD_OUT_ENTITIES
    triples=[t for t in itertools.combinations(pool,3) if 0 < len(set(t)&set(ID_ENTITIES)) < 3]
    rng.shuffle(triples);out.mkdir(parents=True)
    rows=[]
    for index,triple in enumerate(triples[:families]):
        source_a,source_b,source_c=rng.sample(triple,3)
        a,b,c=[ALIASES[e]for e in (source_a,source_b,source_c)]
        family=f"p3dev2_{seed}_{index:03d}";layout_seed=rng.randrange(10**9)
        for visual_forward in (True,False):
            left,right=(a,b) if visual_forward else (b,a)
            image=out/'images'/f'{index:03d}_v{int(visual_forward)}.png'
            render_left,render_right=(source_a,source_b) if visual_forward else (source_b,source_a)
            meta=render_canvas(render_left,render_right,image,random.Random(layout_seed),bg_jitter=0)
            meta['entities_shown']={ALIASES[k]:v for k,v in meta['entities_shown'].items()}
            meta['bboxes']={ALIASES[k]:v for k,v in meta['bboxes'].items()}
            for text_forward,query_forward in itertools.product((True,False),repeat=2):
                tl,tr=(b,c) if text_forward else (c,b);qs,qo=(a,c) if query_forward else (c,a)
                premises=[Premise(left,right,modality='visual'),Premise(tl,tr,modality='textual')];query=Query(qs,qo)
                result=SymbolicSolver.solve(premises,query,verify_with_oracle=True)
                label={'entailed':'Yes','contradicted':'No','underdetermined':'Undetermined'}[result.epistemic_status.value]
                prompt=f"The {tl.replace('_',' ')} is to the left of the {tr.replace('_',' ')}.\n\nQuestion: Is the {qs.replace('_',' ')} to the left of the {qo.replace('_',' ')}?\nAnswer directly with: Yes, No, or Undetermined."
                rows.append({'schema_version':'phase3_factorial_development_v2','sample_id':f'{family}::v{int(visual_forward)}t{int(text_forward)}q{int(query_forward)}',
                             'family_id':family,'split':'development','is_synthetic_demo':False,'entities':{'A':a,'B':b,'C':c},
                             'premises':[{'subject':p.subject,'object':p.object,'modality':p.modality.value} for p in premises],
                             'query':{'subject':qs,'object':qo,'relation':'left_of'},'ground_truth_label':label,'symbolic_status':result.epistemic_status.value,
                             'prompt':prompt,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                             'image_file':image.relative_to(REPO_ROOT).as_posix(),'image_sha256':meta['image_sha256'],
                             'layout':meta,'oracle_agreement':result.algorithm_agreement,'seed':seed,
                             'factorial':{'visual_forward':visual_forward,'text_forward':text_forward,'query_forward':query_forward}})
    (out/'records.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows),encoding='utf-8')
    manifest={'schema':'phase3_factorial_development_v2','seed':seed,'families':families,'cases':len(rows),
              'status':'NEW_DEVELOPMENT_NOT_CONFIRMATORY','labels_per_family':{'Yes':2,'No':2,'Undetermined':4},
              'factors':['visual direction','text direction','query direction'],'source_generator':'src/benchmark/factorial_development.py',
              'records_sha256':hashlib.sha256((out/'records.jsonl').read_bytes()).hexdigest(),
              'frozen_holdout_accessed':False,'model_forwards':0,
              'renderer_name_mapping':ALIASES,
              'independence_limit':'New seed and mixed-pool rendered entity sets disjoint from within-pool v2 families; renderer and prompt grammar shared. Shape names corrected to literal 2D geometry. Development inspected now, never relabel as independent confirmation.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return rows,manifest
