"""Run full discovery behind a read guard for closed mixed/final record files.

This is a per-process verification guard, not a machine-wide access control.
Three legacy full-mixed-dataset tests are explicitly skipped via the documented
environment flag; any unexpected restricted open fails instead of being skipped.
"""
import json
import os
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    os.environ['CROSSMODAL_EXCLUDE_MIXED_DATASETS']='1'
    blocked={ROOT/p for p in ('configs/phase_1c/balanced_final_holdout_v1.jsonl',
                              'configs/phase_1c/final_holdout_family_ids.txt',
                              'datasets/cross_modal_contrast_v2/cross_modal_contrast_v2.jsonl',
                              'datasets/cross_modal_dependency_v1/cross_modal_dependency_v1.jsonl')}
    final_images=ROOT/'datasets/cross_modal_dependency_v1/images'
    denied=[]
    def guard(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0])).resolve()
        if path in blocked or path.is_relative_to(final_images):
            denied.append(path.relative_to(ROOT).as_posix())
            raise PermissionError('Phase 3 closed-data access refused before file open')
    sys.addaudithook(guard)
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'status':'PASS' if result.wasSuccessful() and not denied else 'FAIL',
            'tests_run':result.testsRun,'skipped':len(result.skipped),'failures':len(result.failures),'errors':len(result.errors),
            'restricted_open_attempts':denied,'scope':'per-process Python open audit; not OS-wide tracing or a custodian holdout audit'}
    print(json.dumps(report))
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
