"""Exactly one SyntheticControl seed0 fit per READY method, then no-fit resume."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
BASE=Path(__file__).resolve().parents[1]
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMBA_NUM_THREADS"):
    os.environ[key]="1"
os.environ["PYTHONHASHSEED"]="0";os.environ["PYTHONDONTWRITEBYTECODE"]="1"
os.environ["PYTHONPATH"]=str(BASE/"src")
sys.path.insert(0,str(BASE/"src"))
from baselines2026.registry import READY_IDS,REPOSITORY
from baselines2026.util import external_output,read_json,write_json_new,file_hash

def command(argv,log,env):
    with Path(log).open("x",encoding="utf-8") as stream:
        p=subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT,env=env)
    text=Path(log).read_text(encoding="utf-8")
    print(text[-2500:],flush=True)
    if p.returncode:raise RuntimeError("Smoke command failed: "+str(log))

def main():
    p=argparse.ArgumentParser()
    for name in ("output-root","source-root","data-root","gate"):p.add_argument("--"+name,required=True)
    a=p.parse_args()
    out=external_output(a.output_root,REPOSITORY);out.mkdir(parents=True,exist_ok=False)
    lock=BASE/"configs/requirements-wave1.lock.txt"
    results={}
    for method in READY_IDS:
        env=dict(os.environ);env["NUMBA_CACHE_DIR"]=str(out/"numba-cache"/method)
        print("One real fit: SyntheticControl seed0",method,flush=True)
        args=[sys.executable,"-m","baselines2026","run","--method",method,
              "--data-root",a.data_root,"--source-root",a.source_root,"--output-root",str(out/"runs"),
              "--gate",a.gate,"--lock",str(lock)]
        command(args,out/(method+"-fit.log"),env)
        folders=list((out/"runs"/method).iterdir())
        if len(folders)!=1:raise RuntimeError("Expected one native real fit artifact")
        run=folders[0]
        command([sys.executable,"-m","baselines2026","evaluate","--predictions",str(run/"predictions.json"),
                 "--data-root",a.data_root,"--output",str(run/"evaluation.json")],
                out/(method+"-evaluation.log"),env)
        # Reinvoke the runner with identical identity: MUST reuse, never fit.
        before=file_hash(run/"predictions.json")
        command(args,out/(method+"-resume.log"),env)
        if "REUSED WITHOUT FIT" not in (out/(method+"-resume.log")).read_text(encoding="utf-8"):
            raise RuntimeError("Resume did not reuse immutable result")
        if file_hash(run/"predictions.json")!=before:raise RuntimeError("Resume overwrote predictions")
        results[method]={"folder":str(run),"manifest":read_json(run/"manifest.json"),
                         "evaluation":read_json(run/"evaluation.json"),
                         "resume_without_fit":True}
    write_json_new(out/"summary.json",{"schema":"baselines2026.wave1-smoke.v1",
                   "dataset":"SyntheticControl","seed":0,"real_fits":3,"results":results,
                   "CORE44_authorized":False})
    print("THREE INTEGRATION SMOKES COMPLETE; CORE-44 NOT AUTHORIZED",flush=True)

if __name__=="__main__":main()
