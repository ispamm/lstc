import argparse,os,sys
from pathlib import Path
from ..registry import BASELINE_ROOT
from ..util import write_json_new,digest
from ..provenance import adapter_source,environment
from .runner import run,replay_artifact,lock

def main():
    p=argparse.ArgumentParser();p.add_argument("action",choices=("validate","run","replay","evaluate"));p.add_argument("--method",choices=("ts2vec_kmeans","fcacc"),required=True);p.add_argument("--source-root",required=True);p.add_argument("--data-root");p.add_argument("--output",required=True);p.add_argument("--gate");p.add_argument("--artifact");a=p.parse_args()
    if a.action=="validate":
        import pytest,xml.etree.ElementTree as ET
        out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
        os.environ["BASELINE_SOURCE_ROOT"]=a.source_root;os.environ["WAVE2_METHOD"]=a.method;os.environ["WAVE2_OUTPUT"]=str(out)
        code=pytest.main(["-q",str(BASELINE_ROOT/"tests_wave2a"),str(BASELINE_ROOT/"tests/test_contracts.py"),"--basetemp",str(out/"pytest-tmp"),"-o","cache_dir="+str(out/"pytest-cache"),"--junitxml="+str(out/"tests.xml")])
        suites=list(ET.parse(str(out/"tests.xml")).getroot().iter("testsuite"));stats={key:sum(int(s.get(key,0)) for s in suites) for key in ("tests","failures","errors","skipped")}
        _,eh=environment(lock(a.method));gate={"schema":"baselines2026.wave2a-validation.v1","method":a.method,"passed":code==0 and not any(stats[k] for k in ("failures","errors","skipped")),"tests":stats,"adapter_tree_sha256":adapter_source()["adapter_tree_sha256"],"environment_sha256":eh}
        write_json_new(out/"gate.json",gate);raise SystemExit(code)
    if a.action=="run":run(a.method,a.source_root,a.data_root,a.output,a.gate)
    if a.action=="replay":replay_artifact(a.artifact,a.source_root,a.data_root)
    if a.action=="evaluate":
        from ..evaluator import evaluate_file
        print(evaluate_file(Path(a.artifact)/"predictions.json",a.data_root,a.output))
