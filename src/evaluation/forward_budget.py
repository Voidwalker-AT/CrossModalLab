"""Fail-closed, persisted accounting of top-level model forward attempts."""
from pathlib import Path
from src.training.run_state import write_once


class ForwardBudget:
    def __init__(self,path,limit=12):
        self.path=Path(path)
        if not 1<=limit<=12 or self.path.exists():
            raise ValueError('fresh bounded ledger required; no silent retry or budget reset')
        self.limit=limit;self.used=0;self.path.mkdir(parents=True)
        write_once(self.path/'budget.json',{'limit':limit,'unit':'top-level model forward attempts, including failed forwards'})

    def consume(self,**identity):
        if self.used>=self.limit:raise RuntimeError('forward budget exhausted before model execution')
        step=self.used+1
        write_once(self.path/f'forward_{step:03d}.json',{'attempt':step,**identity})
        self.used=step
