"""Generate and audit a fresh full-factorial DEVELOPMENT benchmark on CPU."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.benchmark.publication_factorial import build,shortcut_audit

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--families',type=int,default=6)
    args=parser.parse_args()
    rows,manifest=build(args.output,families=args.families)
    report=shortcut_audit(rows)
    (args.output/'shortcut_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'manifest':manifest,'shortcut_audit':report},indent=2))

if __name__=='__main__':main()
