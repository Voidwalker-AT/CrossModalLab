"""Freeze a proposed 108-forward DEVELOPMENT screen; this executes no inference."""
import json
from pathlib import Path
import random
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.training.run_state import file_hash,write_once
from src.benchmark.factorial_development import ALIASES


def main():
    base=ROOT/'artifacts/publication/20261025'
    rows=[json.loads(line) for line in (base/'benchmark/records.jsonl').read_text().splitlines()]
    prior=[json.loads(line) for path in ('artifacts/phase3/20261024_v2/selected_development_records.jsonl','artifacts/phase3/20261024_v2/factorial_development/records.jsonl') for line in (ROOT/path).read_text().splitlines()]
    triple=lambda r:tuple(sorted(ALIASES.get(entity,entity) for entity in r['entities'].values()))
    leakage={'scope':'New development benchmark versus only the 68 selected legacy and 96 earlier development records; no mixed/final file opened.',
             'unordered_triple_overlap':len({triple(r) for r in rows}&{triple(r) for r in prior}),
             'exact_prompt_hash_overlap':len({r['prompt_sha256'] for r in rows}&{r['prompt_sha256'] for r in prior}),
             'exact_image_hash_overlap':len({r['image_sha256'] for r in rows}&{r['image_sha256'] for r in prior}),
             'full_training_or_holdout_independence':'UNVERIFIED','entity_names_renderer_grammar_shared':True}
    write_once(base/'selected_source_leakage.json',leakage)
    rng=random.Random(2026102503);selected=[]
    for family in sorted({r['family_id'] for r in rows}):
        for label in ('Yes','No','Undetermined'):
            selected.append(rng.choice([r for r in rows if r['family_id']==family and r['ground_truth_label']==label]))
    plan={'schema':'publication_followup_108_v1','status':'PROPOSED_REQUIRES_EXPLICIT_APPROVAL','authorized':False,
          'seed':2026102503,'dataset_records_sha256':file_hash(base/'benchmark/records.jsonl'),'cases':selected,
          'models':{'smol':'482adb537c021c86670beed01cd58990d01e72e4','qwen':'89644892e4d85e24eaac8bacfd4f463576704203'},
          'conditions':['frozen standard multimodal full-candidate summed LL'],'additional_forward_budget':108,
          'accounting':'2 models x 18 cases x 3 candidates; failures count; no generation, capture, patch, backward or optimizer',
          'decision_rule':'argmax complete-answer summed LL; LL/token diagnostic reported separately',
          'primary_endpoint':'unsupported definite answers among six U cases/model; report sufficient-case incorrect abstention jointly',
          'secondary_endpoints':['three-class BA','macro-F1','per-class recall','paired family consistency','all raw counts'],
          'statistics':'Six family clusters; paired descriptive bootstrap B=2000; no confirmatory significance or generalization claim.',
          'exclusions':'No removal for incorrect/invalid answer; preserve partial runs; halt on first safety/integrity failure without retry.',
          'runtime_estimate_sec':{'smol_scoring':267,'qwen_scoring':30,'load_and_audit_projection':220,'central_total':517,'hard_worker_cap':600,'hard_total':1230},
          'resources':'Available physical RAM >=4GiB throughout and >=6GiB recommended at start; free dedicated VRAM >=3500MiB before each process; stop above3900MiB physical sampled use. Observed Smol physical peak3876MiB leaves little headroom. Separate processes, baseline return verified.',
          'storage':'4GiB disk reserve; private parameter snapshots removed only by their owning audit; persistent results/telemetry expected <5MiB, no weights/checkpoints',
          'failure_and_recovery':'No model updates; retain partial outputs/ledger and stop. Terminate only owned workers on timeout; verify process exit and physical memory return. No automatic reset/retry.',
          'scope_limits':'Exposed development only. New model outcomes absent. No trained adapters, unimodal VLM conditions, full nuisance battery, independent confirmation or causal claim. These require separately costed protocols.'}
    write_once(base/'followup_108_plan.json',plan)
    print(json.dumps({'status':'FROZEN_PROPOSAL_NOT_EXECUTED','cases':len(selected),'families':6,'additional_forwards_requested':108,'selected_source_leakage':leakage}))


if __name__=='__main__':main()
