"""Three native tiny fits at seeds0,0,1; scientific recipe unchanged."""
import pickle,warnings
import numpy as np
from ..datasets import TrainingInput,normalize_rows
from ..evaluator import metrics
from ..predictions import submission,save_submission,load_submission,utilization
from ..provenance import adapter_source,environment
from ..registry import BASELINE_ROOT,REPOSITORY
from ..util import digest,write_json_new,external_output
from .sources import verify
from .r_clustering import fit_pipeline
from .fasa import fit as fit_fasa
from .runner import replay_model
def tiny():
    rng=np.random.RandomState(19);t=np.linspace(0,2*np.pi,64)
    return np.vstack([np.sin(t)+rng.normal(0,.15,64) for _ in range(24)]+[np.sin(3*t)+rng.normal(0,.15,64) for _ in range(24)])
def validate(method,root,derived,output):
    out=external_output(output,REPOSITORY);out.mkdir(parents=True,exist_ok=False)
    raw=tiny();x=normalize_rows(raw);y=np.repeat([0,1],24)
    data=TrainingInput("SyntheticTinyWave1B",raw,tuple("SyntheticTinyWave1B/TRAIN/%d"%i for i in range(48)),2,{"generator":digest({"seed":19,"N":48,"L":64})},{})
    _,source=verify(method,root);adapter=adapter_source();_,env=environment(BASELINE_ROOT/"configs/requirements-wave1b.lock.txt")
    outputs=[];states=[];centers=[];warn=[]
    for repeat,seed in enumerate((0,0,1)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model,detail=fit_pipeline(x,2,seed,root,derived) if method=="r_clustering" else fit_fasa(x,2,seed,root)
            states.append(detail["biases_sha256"] if method=="r_clustering" else detail["initial_partition_sha256"])
            centers.append(model.kmeans.cluster_centers_.copy() if method=="r_clustering" else np.asarray([v[0] for v in model.clusters]))
            with (out/("model%d.pkl"%repeat)).open("xb") as s:pickle.dump(model,s,protocol=pickle.HIGHEST_PROTOCOL)
            with (out/("model%d.pkl"%repeat)).open("rb") as s:restored=pickle.load(s)
            assert np.array_equal(restored.labels_,model.labels_)
            assert np.array_equal(replay_model(method,restored,x,root,derived),model.labels_)
        warn.extend({"category":type(w.message).__name__,"message":str(w.message)} for w in caught)
        payload=submission(data.identity(),model.labels_,{"method":method,"seed":seed,"run_id":"tiny-%s-%d"%(method,repeat),
            "source_commit":source["source_commit"],"config_sha256":digest({"method":method,"seed":seed,"recipe":"full native"}),
            "environment_sha256":env,"adapter_tree_sha256":adapter["adapter_tree_sha256"]})
        path=out/("predictions%d.json"%repeat);save_submission(path,payload,data.identity());load_submission(path,data.identity())
        outputs.append(model.labels_.copy());write_json_new(out/("details%d.json"%repeat),detail)
    assert np.array_equal(outputs[0],outputs[1]) and np.array_equal(centers[0],centers[1]) and states[0]==states[1]
    assert states[0]!=states[2]
    result={"method":method,"passed":True,"fits":3,"seeds":[0,0,1],"N":48,"L":64,"k":2,
        "same_seed_labels_centers_and_state_exact":True,"changed_seed_stochastic_state_changed":True,
        "checkpoint_exact":True,"diagnostic_state_hashes":states,"warnings":warn,"source":source,
        "metrics_integration_only":metrics(y,outputs[0]),"utilization":utilization(outputs[0],2),
        "adapter_tree_sha256":adapter["adapter_tree_sha256"],"environment_sha256":env}
    write_json_new(out/"result.json",result);print("SYNTHETIC PASS",method,"fixed/changed seed and checkpoint",flush=True)
