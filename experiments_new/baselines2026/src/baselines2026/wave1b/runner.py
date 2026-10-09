"""One ECG200 seed0 fit per method; immutable export and no-fit replay."""
import pickle,traceback,warnings
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ..datasets import load_training_input,normalize_rows
from ..predictions import submission,save_submission,load_submission,utilization,integer_clusters
from ..provenance import adapter_source,environment,rng_settings,snapshot_adapter
from ..registry import BASELINE_ROOT,REPOSITORY
from ..resources import ResourceMonitor,hardware
from ..timing import PipelineTimer
from ..util import ContractError,read_json,write_json_new,digest,file_hash,array_hash,external_output
from .sources import IDS,verify,load_r
from .r_clustering import fit_pipeline,NativeDomainError
from .fasa import fit as fit_fasa
def classify(e):
    if isinstance(e,NativeDomainError):return e.classification
    if isinstance(e,MemoryError):return "RESOURCE FAILURE"
    if isinstance(e,FloatingPointError):return "NUMERICAL FAILURE"
    if isinstance(e,ImportError):return "DEPENDENCY ISSUE"
    if isinstance(e,ContractError):return "ADAPTER BUG"
    return "UNKNOWN"
def replay_model(method,model,x,root,derived):
    if method=="fasa":return model.replay_membership()
    native,_=load_r(root,derived)
    with threadpool_limits(limits=1):return integer_clusters(model.predict(x,native),len(x))
def run_once(method,root,data_root,output_root,derived,gate_path):
    if method not in IDS:raise ContractError("Only two authorized Wave1B methods")
    cfg=read_json(BASELINE_ROOT/"configs/wave1b.json")
    if cfg["dataset"]!="ECG200" or cfg["seed"]!=0 or tuple(cfg["methods"])!=IDS:raise ContractError("Only ECG200 seed0 authorized")
    adapter=adapter_source();env,env_hash=environment(BASELINE_ROOT/"configs/requirements-wave1b.lock.txt")
    _,source=verify(method,root);gate=read_json(gate_path)
    if (gate.get("schema")!="baselines2026.wave1b-validation.v1" or not gate.get("passed")
        or gate["adapter_tree_sha256"]!=adapter["adapter_tree_sha256"] or gate["environment_sha256"]!=env_hash
        or set(gate["synthetic_passed"])!=set(IDS) or any(gate["tests"][k] for k in ("failures","errors","skipped"))):
        raise ContractError("Validation gate missing/stale/incomplete")
    data=load_training_input(data_root,"ECG200",cfg["expected"]);identity=data.identity()
    key={"method":method,"dataset":"ECG200","seed":0,"input_hashes":data.input_hashes,
        "config_sha256":digest({"recipe":cfg["recipes"][method],"dataset":"ECG200","seed":0,"k":data.k,"preprocessing":cfg["preprocessing"],"threads":1}),
        "source_commit":source["source_commit"],"environment_sha256":env_hash,"adapter_tree_sha256":adapter["adapter_tree_sha256"]}
    run_id=digest(key);folder=external_output(output_root,REPOSITORY)/method/run_id
    derived=external_output(derived,REPOSITORY)
    if folder.exists():
        p=load_submission(folder/"predictions.json",identity)
        if any(p.get(k)!=v for k,v in {**key,"run_id":run_id}.items()):raise ContractError("Run identity mismatch")
        m=read_json(folder/"manifest.json")
        for n,h in m["artifact_sha256"].items():
            if file_hash(folder/n)!=h:raise ContractError("Frozen artifact changed")
        for n,h in m["adapter"]["files"].items():
            if file_hash(folder/"adapter-source"/n)!=h:raise ContractError("Source bundle changed")
        print("REUSED WITHOUT FIT",method,run_id,flush=True);return folder
    folder.mkdir(parents=True);snapshot_adapter(folder,adapter);timer=PipelineTimer()
    try:
        with ResourceMonitor() as resource,warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with timer.pipeline():
                with timer.stage("common_normalization"):x=normalize_rows(data.raw)
                if method=="r_clustering":
                    model,detail=fit_pipeline(x,data.k,0,root,derived,timer)
                    with timer.stage("full_pool_stored_transform_and_inference"):inference=replay_model(method,model,x,root,derived)
                    inference_seconds=timer.stages["full_pool_stored_transform_and_inference"]
                else:
                    model,detail=fit_fasa(x,data.k,0,root,timer)
                    inference=model.replay_membership();inference_seconds=None
                with timer.stage("prediction_validation"):
                    clusters=integer_clusters(model.labels_,data.N)
                    if not np.array_equal(inference,clusters):raise ContractError("Fitted/native replay assignments differ")
                    payload=submission(identity,clusters,{**key,"run_id":run_id,"normalized_array_sha256":array_hash(x),"preprocessing":cfg["preprocessing"]})
        warning_records=[{"category":type(w.message).__name__,"message":str(w.message)} for w in caught]
        with (folder/"model.pkl").open("xb") as s:pickle.dump(model,s,protocol=pickle.HIGHEST_PROTOCOL)
        with (folder/"model.pkl").open("rb") as s:restored=pickle.load(s)
        if not np.array_equal(restored.labels_,clusters) or not np.array_equal(replay_model(method,restored,x,root,derived),clusters):raise ContractError("Checkpoint replay differs")
        save_submission(folder/"predictions.json",payload,identity)
        m={**key,"run_id":run_id,"execution_status":"SUCCESS","timestamp_utc":datetime.now(timezone.utc).isoformat(),
            "dataset_identity":identity,"normalized_array_sha256":array_hash(x),"source":source,"adapter":adapter,
            "environment":env,"hardware":hardware(),"RNG":{**rng_settings(0),"native_random_state":None,"numba_seed":0 if method=="r_clustering" else None},
            "recipe":cfg["recipes"][method],"native_details":detail,"preprocessing":cfg["preprocessing"],
            "pipeline_wall_seconds":timer.total,"stage_seconds":timer.stages,"inference_seconds":inference_seconds,
            "resources":resource.result(),"warnings":warning_records,"utilization":utilization(clusters,data.k),
            "checkpoint_partition_exact":True,"checkpoint_extraction_exact":True,
            "timing_quality":"integration only; not publication efficiency",
            "timing_scope":"raw decoded array before normalization through native/JIT/all scientific stages/ordered extraction; excludes decoding/checkpoint/export/reload/evaluation; source preparation included",
            "artifact_sha256":{n:file_hash(folder/n) for n in ("model.pkl","predictions.json")}}
        write_json_new(folder/"manifest.json",m)
        print("SUCCESS",method,"N",data.N,"pipeline_seconds",timer.total,"run",run_id,flush=True);return folder
    except Exception as error:
        write_json_new(folder/"failure.json",{**key,"run_id":run_id,"classification":classify(error),"error":repr(error),
            "pipeline_seconds":timer.total,"stages":timer.stages,"traceback":traceback.format_exc()})
        raise
def replay_artifact(folder,root,data_root,derived):
    folder=Path(folder);m=read_json(folder/"manifest.json")
    for n,h in m["artifact_sha256"].items():
        if file_hash(folder/n)!=h:raise ContractError("Frozen artifact changed")
    _,h=environment(BASELINE_ROOT/"configs/requirements-wave1b.lock.txt")
    if h!=m["environment_sha256"]:raise ContractError("Replay environment changed")
    _,source=verify(m["method"],root)
    if source["source_commit"]!=m["source_commit"]:raise ContractError("Replay source changed")
    data=load_training_input(data_root,"ECG200",read_json(BASELINE_ROOT/"configs/wave1b.json")["expected"])
    p=load_submission(folder/"predictions.json",data.identity())
    with (folder/"model.pkl").open("rb") as s:model=pickle.load(s)
    if not np.array_equal(replay_model(m["method"],model,normalize_rows(data.raw),root,derived),p["predicted_clusters"]):raise ContractError("Fresh-process replay differs")
    print("ARTIFACT REPLAY EXACT; NO FIT",m["method"],flush=True)
