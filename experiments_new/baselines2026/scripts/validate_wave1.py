"""Gated unit/source/import/synthetic validation; no real UCR data reads."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

BASE=Path(__file__).resolve().parents[1]
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMBA_NUM_THREADS"):
    os.environ[key]="1"
os.environ["PYTHONHASHSEED"]="0"
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
os.environ["PYTHONPATH"]=str(BASE/"src")
sys.path.insert(0,str(BASE/"src"))
from baselines2026.provenance import adapter_source,environment
from baselines2026.registry import READY_IDS,REPOSITORY
from baselines2026.util import external_output,read_json,write_json_new

def command(argv,log):
    p=subprocess.run(argv,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    with Path(log).open("x",encoding="utf-8") as f:f.write(p.stdout)
    print(p.stdout[-4000:],flush=True)
    if p.returncode:raise RuntimeError("Validation command failed; retained log: "+str(log))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output-root",required=True);p.add_argument("--source-root",required=True)
    a=p.parse_args()
    out=external_output(a.output_root,REPOSITORY);out.mkdir(parents=True,exist_ok=False)
    os.environ["NUMBA_CACHE_DIR"]=str(out/"numba-cache")
    xml=out/"unit-tests.xml"
    print("Running unit/native-parity tests, no real data",flush=True)
    command([sys.executable,"-m","pytest",str(BASE/"tests"),"-q","-p","no:cacheprovider",
             "--basetemp",str(out/"pytest-temp"),"--junitxml",str(xml)],out/"unit-tests.log")
    suites=list(ET.parse(xml).getroot().iter("testsuite"))
    totals={key:sum(int(s.get(key,0)) for s in suites) for key in ("tests","failures","errors","skipped")}
    if totals["failures"] or totals["errors"] or totals["skipped"]:raise RuntimeError("Tests incomplete")
    lock=BASE/"configs/requirements-wave1.lock.txt"
    for method in READY_IDS:
        print("Running synthetic seed/reload validation:",method,flush=True)
        command([sys.executable,"-m","baselines2026.synthetic","--method",method,
                 "--source-root",a.source_root,"--output-root",str(out/"synthetic"/method),
                 "--lock",str(lock)],out/("synthetic-"+method+".log"))
    adapter=adapter_source();env,h=environment(lock)
    result={"schema":"baselines2026.validation.v1","passed":True,
            "adapter_tree_sha256":adapter["adapter_tree_sha256"],"environment_sha256":h,
            "unit_tests":{"passed":totals["tests"],"failed":totals["failures"],
                          "errors":totals["errors"],"skipped":totals["skipped"]},
            "synthetic_passed":list(READY_IDS),
            "synthetic":{m:read_json(out/"synthetic"/m/"result.json") for m in READY_IDS}}
    write_json_new(out/"gate.json",result)
    print("VALIDATION GATE PASSED",out/"gate.json",flush=True)

if __name__=="__main__":main()
