"""New/native/common tests, unchanged Wave1 regressions, synthetic gate; no real fits."""
import argparse,os,subprocess,sys
from pathlib import Path
import xml.etree.ElementTree as ET
BASE=Path(__file__).resolve().parents[1]
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMBA_NUM_THREADS"):os.environ[k]="1"
os.environ["PYTHONDONTWRITEBYTECODE"]="1";os.environ["PYTHONHASHSEED"]="0";os.environ["PYTHONPATH"]=str(BASE/"src")
sys.dont_write_bytecode=True;sys.path.insert(0,str(BASE/"src"))
from baselines2026.provenance import adapter_source,environment
from baselines2026.registry import REPOSITORY
from baselines2026.util import read_json,write_json_new,external_output
from baselines2026.wave1b.sources import IDS
def command(args,log,env):
    with Path(log).open("x",encoding="utf-8") as s:p=subprocess.run(args,stdout=s,stderr=subprocess.STDOUT,env=env)
    print(Path(log).read_text(encoding="utf-8")[-5500:],flush=True)
    if p.returncode:raise RuntimeError("Validation failed; retained log "+str(log))
def counts(path):
    suites=list(ET.parse(path).getroot().iter("testsuite"))
    return {k:sum(int(s.get(k,0)) for s in suites) for k in ("tests","failures","errors","skipped")}
def main():
    p=argparse.ArgumentParser()
    for k in ("output-root","source-root","wave1-python"):p.add_argument("--"+k,required=True)
    a=p.parse_args();out=external_output(a.output_root,REPOSITORY);out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ);env["NUMBA_CACHE_DIR"]=str(out/"numba-cache")
    xml=out/"wave1b-tests.xml";print("Wave1B/native/common tests; no real data",flush=True)
    command([sys.executable,"-m","pytest",str(BASE/"tests_wave1b"),str(BASE/"tests/test_contracts.py"),"-q","-p","no:cacheprovider","--basetemp",str(out/"test-temp"),"--junitxml",str(xml)],out/"wave1b-tests.log",env)
    tests=counts(xml);regxml=out/"wave1-regression.xml";regenv=dict(env);regenv["NUMBA_CACHE_DIR"]=str(out/"wave1-regression-numba")
    print("All unchanged Wave1 regressions in original locked environment",flush=True)
    command([a.wave1_python,"-m","pytest",str(BASE/"tests"),"-q","-p","no:cacheprovider","--basetemp",str(out/"wave1-test-temp"),"--junitxml",str(regxml)],out/"wave1-regression.log",regenv)
    old=counts(regxml)
    if any(v[k] for v in (tests,old) for k in ("failures","errors","skipped")):raise RuntimeError("Validation incomplete")
    for method in IDS:
        print("Standalone synthetic",method,flush=True)
        command([sys.executable,"-m","baselines2026.wave1b","synthetic","--method",method,"--source-root",a.source_root,"--derived-root",str(out/"native-derived"),"--output-root",str(out/"synthetic"/method)],out/("synthetic-"+method+".log"),env)
    _,h=environment(BASE/"configs/requirements-wave1b.lock.txt");adapter=adapter_source()
    write_json_new(out/"gate.json",{"schema":"baselines2026.wave1b-validation.v1","passed":True,"tests":tests,"wave1_regression":old,
        "synthetic_passed":list(IDS),"synthetic":{m:read_json(out/"synthetic"/m/"result.json") for m in IDS},
        "adapter_tree_sha256":adapter["adapter_tree_sha256"],"environment_sha256":h})
    print("WAVE1B VALIDATION GATE PASSED",out/"gate.json",flush=True)
if __name__=="__main__":main()
