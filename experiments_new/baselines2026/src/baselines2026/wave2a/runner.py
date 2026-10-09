"""Authorized one-fit scope, immutable identity/export and fresh-process replay."""
import os,traceback,warnings
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,torch
from ..datasets import load_training_input,normalize_rows
from ..predictions import submission,save_submission,load_submission,utilization,integer_clusters
from ..provenance import adapter_source,environment,snapshot_adapter
from ..registry import BASELINE_ROOT,REPOSITORY
from ..resources import ResourceMonitor,hardware
from ..timing import PipelineTimer
from ..util import ContractError,read_json,write_json_new,digest,file_hash,array_hash,external_output
from .sources import verify
from .runtime import cuda_probe,GPUResource
from .checkpoints import save,replay

def lock(method):return BASELINE_ROOT/("configs/requirements-wave2a-"+("ts2vec" if method=="ts2vec_kmeans" else "fcacc")+".lock.txt")
def config(method):
    if method not in ("ts2vec_kmeans","fcacc"):raise ContractError("Wave2A admits exactly two methods")
    return read_json(BASELINE_ROOT/"configs/wave2a.json")[method]
def run(method,root,data_root,output,gate_path):
    cfg=config(method);adapter=adapter_source();env,eh=environment(lock(method));gate=read_json(gate_path)
    if not gate.get("passed") or gate.get("method")!=method or gate.get("adapter_tree_sha256")!=adapter["adapter_tree_sha256"] or gate.get("environment_sha256")!=eh:raise ContractError("Validation gate stale or failed")
    _,source=verify(method,root);data=load_training_input(data_root,cfg["dataset"],cfg["expected"]);identity=data.identity()
    key={"method":method,"dataset":cfg["dataset"],"seed":0,"input_hashes":data.input_hashes,"config_sha256":digest(cfg),"source_commit":source["source_commit"],"environment_sha256":eh,"adapter_tree_sha256":adapter["adapter_tree_sha256"]}
    run_id=digest(key);folder=external_output(output,REPOSITORY)/method/run_id
    if folder.exists():
        m=read_json(folder/"manifest.json");load_submission(folder/"predictions.json",identity)
        if any(m.get(k)!=v for k,v in key.items()):raise ContractError("Resume identity mismatch")
        for rel,h in m["artifact_sha256"].items():
            if file_hash(folder/rel)!=h:raise ContractError("Artifact changed")
        for rel,h in m["adapter"]["files"].items():
            if file_hash(folder/"adapter-source"/rel)!=h:raise ContractError("Archived adapter changed")
        print("REUSED WITHOUT FIT",folder,flush=True);return folder
    device=cuda_probe()
    ledger=Path(root).parent/"wave2a/authorized-fits"/(method+".json")
    if ledger.exists():raise ContractError("The single authorized real fit has already been claimed; no refit")
    write_json_new(ledger,{**key,"run_id":run_id,"artifact_directory":str(folder),"authorization":"one full native fit only"})
    folder.mkdir(parents=True);snapshot_adapter(folder,adapter);timer=PipelineTimer(torch.cuda.synchronize)
    previous=os.getcwd();os.chdir(str(folder))
    gpu=None;cpu=None
    try:
        with ResourceMonitor() as cpu,GPUResource() as gpu,warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with timer.pipeline():
                with timer.stage("common_preprocessing"):
                    normalized=normalize_rows(data.raw);x=normalized.astype(np.float32)[:,:,None]
                if method=="ts2vec_kmeans":
                    from .ts2vec_adapter import fit
                    model,head,reps,labels,detail=fit(x,data.k,0,cfg["dataset"],root,timer)
                    if detail["completed_steps"]!=detail["native_budget"]:raise ContractError("Native budget incomplete")
                else:
                    from .fcacc_adapter import fit
                    model,head,reps,labels,detail=fit(x,data.k,0,root,folder,timer)
                    if detail["completed_pretraining_epochs"]!=100 or detail["completed_joint_epochs"]!=30:raise ContractError("Native budget incomplete")
                with timer.stage("canonical_prediction_validation"):
                    labels=integer_clusters(labels,data.N)
                    payload=submission(identity,labels,{**key,"run_id":run_id,"normalized_array_sha256":array_hash(normalized),"native_input_sha256":array_hash(x)})
        save(folder,method,model,head,detail,x,0,data.k);np.save(str(folder/"representations.npy"),reps)
        restored,restored_reps=replay(folder,method,x,root)
        if not np.array_equal(restored,labels) or not np.array_equal(restored_reps,reps):raise ContractError("Active checkpoint round trip differs")
        save_submission(folder/"predictions.json",payload,identity)
        artifacts=["checkpoint.pt","representations.npy","predictions.json"]+(["head.pkl"] if head is not None else [])
        if method=="fcacc":artifacts +=["native_Pretraining_phase","native_Finetuning_phase","native_Centers","pretraining.csv","finetuning.csv"]
        m={**key,"run_id":run_id,"execution_status":"SUCCESS","timestamp_utc":datetime.now(timezone.utc).isoformat(),"dataset_identity":identity,"normalized_array_sha256":array_hash(normalized),"native_input_sha256":array_hash(x),"source":source,"adapter":adapter,"environment":env,"recipe":cfg,"device":device,"host":hardware(),"pipeline_wall_seconds":timer.total,"stage_seconds":timer.stages,"GPU_resources":gpu.result,"CPU_resources":cpu.result(),"native_details":detail,"warnings":[{"category":type(w.message).__name__,"message":str(w.message)} for w in caught],"utilization":utilization(labels,data.k),"checkpoint_predictions_exact":True,"checkpoint_representations_exact":True,"timing_scope":"normalized input conversion through initialization, all native training/diagnostic execution, extraction/head and ordered prediction validation; excludes raw decode, checkpoint/export/reload/evaluation","RNG":{"root_seed":0,"stream_seeds":detail.get("native_RNG_seeds",{"python":0,"numpy":0,"torch_cpu":0,"torch_cuda":0}),"cudnn_deterministic":True,"cudnn_benchmark":False,"CPU_threads":1,"FCACC_center_seed":0 if method=="fcacc" else None,"head_seed":detail.get("head_seed")},"artifact_sha256":{n:file_hash(folder/n) for n in artifacts}}
        write_json_new(folder/"manifest.json",m);print("SUCCESS",folder,"pipeline_seconds",timer.total,flush=True);return folder
    except Exception as e:
        text=str(e).lower();category="RESOURCE FAILURE" if "out of memory" in text or isinstance(e,MemoryError) else "NUMERICAL FAILURE" if isinstance(e,FloatingPointError) else "ADAPTER BUG" if isinstance(e,ContractError) else "DEPENDENCY ISSUE" if isinstance(e,ImportError) else "UNKNOWN"
        write_json_new(folder/"failure.json",{**key,"run_id":run_id,"classification":category,"error":repr(e),"traceback":traceback.format_exc(),"configuration":cfg,"source":source,"environment":env,"device":device,"pipeline_seconds":timer.total,"stages":timer.stages,"GPU_resources":getattr(gpu,"result",None),"CPU_resources":cpu.result() if cpu else None})
        raise
    finally:os.chdir(previous)

def replay_artifact(folder,root,data_root):
    folder=Path(folder);m=read_json(folder/"manifest.json");method=m["method"]
    _,eh=environment(lock(method));_,source=verify(method,root)
    if eh!=m["environment_sha256"] or source["source_commit"]!=m["source_commit"]:raise ContractError("Replay identity differs")
    for rel,h in m["artifact_sha256"].items():
        if file_hash(folder/rel)!=h:raise ContractError("Artifact changed")
    data=load_training_input(data_root,m["dataset"],config(method)["expected"]);p=load_submission(folder/"predictions.json",data.identity());x=normalize_rows(data.raw).astype(np.float32)[:,:,None]
    labels,reps=replay(folder,method,x,root)
    if not np.array_equal(labels,p["predicted_clusters"]) or not np.array_equal(reps,np.load(str(folder/"representations.npy"))):raise ContractError("Fresh process replay differs")
    print("FRESH-PROCESS CHECKPOINT/REPRESENTATION REPLAY EXACT; NO FIT",flush=True)
