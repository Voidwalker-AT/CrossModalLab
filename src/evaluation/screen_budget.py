"""One approved screen worker's persisted quota; separate from the 12-forward pilot."""
from pathlib import Path
from src.training.run_state import write_once


class ScreenBudget:
    def __init__(self,path,limit=54):
        self.path=Path(path)
        if limit!=54 or self.path.exists():raise ValueError('fresh 54-forward worker ledger required; no reset')
        self.limit=limit;self.used=0;self.path.mkdir(parents=True)
        write_once(self.path/'budget.json',{'limit':54,'mission_total_limit':108,'unit':'forward attempts, including failures','authorization':'2026-10-25 frozen followup_108_plan.json'})

    def consume(self,**identity):
        if self.used>=54:raise RuntimeError('approved worker forward budget exhausted')
        attempt=self.used+1
        write_once(self.path/f'forward_{attempt:03d}.json',{'attempt':attempt,**identity})
        self.used=attempt
