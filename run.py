"""Command-line workflow. See START_HERE.html or README.md."""
import argparse,json
from pathlib import Path
import joblib,pandas as pd
from core import simulate,train,score,MODEL_NAMES
ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description='Procurement Pulse V2 — synthetic ML workbench')
    parser.add_argument('command',choices=['all','simulate','train','predict'])
    parser.add_argument('--input',type=Path,default=ROOT/'data'/'synthetic_procurement.csv')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs')
    parser.add_argument('--bundle',type=Path,default=ROOT/'outputs'/'model_bundle.joblib')
    parser.add_argument('--model',choices=list(MODEL_NAMES),default='isolation_forest')
    parser.add_argument('--percentile',type=float,default=None)
    parser.add_argument('--train-end',default='2026-05')
    parser.add_argument('--calibration-end',default='2026-06')
    a=parser.parse_args()
    try:
        if a.command in ['all','simulate']:
            a.input.parent.mkdir(parents=True,exist_ok=True);simulate().to_csv(a.input,index=False)
            print(f'Created 1,200 SYNTHETIC cases: {a.input}')
        if a.command in ['all','train']:
            _,_,report=train(pd.read_csv(a.input),a.output,a.percentile if a.percentile is not None else 90,a.train_end,a.calibration_end)
            print(json.dumps(report,indent=2));print(f'Outputs: {a.output}')
        if a.command=='predict':
            # Only load your own trusted bundle; joblib deserialization can execute code.
            b=joblib.load(a.bundle);s=score(pd.read_csv(a.input),b,a.model,a.percentile)
            a.output.mkdir(parents=True,exist_ok=True);s.to_csv(a.output/'new_case_scores.csv',index=False)
            print(f'Scored {len(s)} cases, {int(s.flagged.sum())} alerts: {a.output / "new_case_scores.csv"}')
    except (ValueError,FileNotFoundError,KeyError,TypeError) as exc:parser.exit(2,f'Input error: {exc}\n')
if __name__=='__main__':main()
