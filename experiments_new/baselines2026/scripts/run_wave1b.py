"""One ECG200 seed0 fit per method; no scientific rescue, preserve every failure."""
import argparse,os,subprocess,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMBA_NUM_THREADS"):os.environ[k]="1"
os.environ["PYTHONDONTWRITEBYTECODE"]="1";os.environ["PYTHONHASHSEED"]="0";os.environ["PYTHONPATH"]=str(BASE/"src")
sys.dont_write_bytecode=True;sys.path.insert(0,str(BASE/"src"))
from baselines2026.registry import REPOSITORY
from baselines2026.util import read_json,write_json_new,external_output,file_hash
from baselines2026.wave1b.sources import IDS
def command(args,log,env):
    with Path(log).open("x",encoding="utf-8") as s:p=subprocess.run(args,stdout=s,stderr=subprocess.STDOUT,env=env)
    print(Path(log).read_text(encoding="utf-8")[-3000:],flush=True)
    if p.returncode:raise RuntimeError("Command failed; retained "+str(log))
def main():
    p=argparse.ArgumentParser()
    for k in ("output-root","source-root","data-root","gate"):p.add_argument("--"+k,required=True)
    a=p.parse_args();out=external_output(a.output_root,REPOSITORY);out.mkdir(parents=True,exist_ok=False);results={}
    for method in IDS:
        env=dict(os.environ);env["NUMBA_CACHE_DIR"]=str(out/"numba-cache"/method);derived=out/"native-derived"/method
        args=[sys.executable,"-m","baselines2026.wave1b","run","--method",method,"--source-root",a.source_root,
              "--data-root",a.data_root,"--output-root",str(out/"runs"),"--derived-root",str(derived),"--gate",a.gate]
        try:
            print("ONE ECG200 seed0 native fit",method,flush=True);command(args,out/(method+"-fit.log"),env)
            folders=list((out/"runs"/method).iterdir())
            if len(folders)!=1:raise RuntimeError("Expected one result")
            run=folders[0]
            command([sys.executable,"-m","baselines2026","evaluate","--predictions",str(run/"predictions.json"),
                     "--data-root",a.data_root,"--output",str(run/"evaluation.json")],out/(method+"-evaluation.log"),env)
            command([sys.executable,"-m","baselines2026.wave1b","replay","--folder",str(run),
                     "--source-root",a.source_root,"--data-root",a.data_root,"--derived-root",str(derived)],out/(method+"-replay.log"),env)
            before=file_hash(run/"predictions.json");command(args,out/(method+"-resume.log"),env)
            if "REUSED WITHOUT FIT" not in (out/(method+"-resume.log")).read_text() or before!=file_hash(run/"predictions.json"):raise RuntimeError("Resume failed")
            results[method]={"status":"SUCCESS","folder":str(run),"manifest":read_json(run/"manifest.json"),
                "evaluation":read_json(run/"evaluation.json"),"fresh_process_replay_exact":True,"resume_without_fit":True}
        except Exception as error:
            failures=list((out/"runs"/method).glob("*/failure.json"))
            results[method]={"status":"FAILED","error":repr(error),"failure":read_json(failures[0]) if failures else {"classification":"UNKNOWN"},
                "action":"stopped method; no retry/scientific change"}
            print("STOPPED METHOD",method,results[method]["failure"]["classification"],flush=True)
    write_json_new(out/"summary.json",{"schema":"baselines2026.wave1b-smoke.v1","dataset":"ECG200","seed":0,
        "results":results,"successful_real_fits":sum(v["status"]=="SUCCESS" for v in results.values()),"CORE44_authorized":False})
    print("WAVE1B SMOKE COMPLETE; CORE44 AUTHORIZATION NO",flush=True)
    if any(v["status"]!="SUCCESS" for v in results.values()):sys.exit(1)
if __name__=="__main__":main()
