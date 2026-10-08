"""Synthetic-only full-default/seed/reload/contract validation, no real datasets."""
import argparse
import pickle
import warnings
from pathlib import Path
import numpy as np
from .adapters import fit_predict, infer
from .datasets import TrainingInput, normalize_rows
from .evaluator import metrics
from .predictions import submission, save_submission, load_submission, utilization
from .provenance import adapter_source, environment, verify_native
from .registry import REPOSITORY, default_config
from .util import digest, external_output, write_json_new

def validate(method, source_root, output_root, lock):
    folder=external_output(output_root,REPOSITORY)
    folder.mkdir(parents=True)
    t=np.linspace(0,2*np.pi,16)
    rng=np.random.RandomState(19)
    raw=np.vstack([np.sin(t)+rng.normal(0,.04,16) for _ in range(8)] +
                  [np.sign(np.sin(2*t))+rng.normal(0,.04,16) for _ in range(8)])
    labels=np.repeat([0,1],8)  # independent test evaluator only
    x=normalize_rows(raw)
    data=TrainingInput("SyntheticTiny",raw,tuple("SyntheticTiny/TRAIN/%d"%i for i in range(16)),
                       2,{"synthetic_generator":digest({"seed":19,"N":16,"L":16})},{})
    native=verify_native(method,source_root);env,env_hash=environment(lock)
    adapter=adapter_source();outputs=[];centers=[];warning_records=[]
    for repeat in range(2):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitted=fit_predict(method,x,2,0,source_root)
            inferred=infer(method,fitted.model,x,17)
            restored=pickle.loads(pickle.dumps(fitted.model,protocol=pickle.HIGHEST_PROTOCOL))
            assert np.array_equal(restored.labels_,fitted.clusters)
            assert np.array_equal(infer(method,restored,x,17),inferred)
        warning_records.extend(str(w.message) for w in caught)
        p=submission(data.identity(),fitted.clusters,{"method":method,"seed":0,
              "run_id":"synthetic-%s-repeat%d"%(method,repeat),
              "source_commit":native["source_commit"],
              "config_sha256":digest(default_config(method)),
              "environment_sha256":env_hash,"adapter_tree_sha256":adapter["adapter_tree_sha256"]})
        path=folder/("predictions-repeat%d.json"%repeat)
        save_submission(path,p,data.identity());load_submission(path,data.identity())
        outputs.append(fitted.clusters.copy())
        centers.append(getattr(fitted.model,"centroids_",getattr(fitted.model,"cluster_centers_",None)).copy())
    if not np.array_equal(outputs[0],outputs[1]) or not np.array_equal(centers[0],centers[1]):
        raise AssertionError("Same-seed native fits are not repeatable")
    scores=metrics(labels,outputs[0])  # assignments already immutable; no selection.
    result={"method":method,"passed":True,"fits":2,"dataset":"SyntheticTiny","N":16,"L":16,"k":2,
            "seed":0,"same_seed_partition_exact":True,"same_seed_centers_exact":True,
            "checkpoint_partition_exact":True,"checkpoint_inference_exact":True,
            "source":native,"environment_sha256":env_hash,"adapter_tree_sha256":adapter["adapter_tree_sha256"],
            "metrics_diagnostic_only":scores,"utilization":utilization(outputs[0],2),
            "warnings":warning_records}
    write_json_new(folder/"result.json",result)
    print("SYNTHETIC PASS",method,"two full-default same-seed fits",flush=True)
    return result

def main():
    p=argparse.ArgumentParser()
    for name in ("method","source-root","output-root","lock"):p.add_argument("--"+name,required=True)
    a=p.parse_args();validate(a.method,a.source_root,a.output_root,a.lock)

if __name__=="__main__":
    main()
