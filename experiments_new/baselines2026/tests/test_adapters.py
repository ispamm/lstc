import inspect
import random
import numpy as np
import pytest
from threadpoolctl import threadpool_limits
from baselines2026.adapters import fit_predict, infer, native_model, native_shape
from baselines2026.datasets import normalize_rows
from baselines2026.provenance import seeded, verify_native, phase_seed
from baselines2026.registry import READY_IDS, default_config, method_record
from baselines2026.util import ContractError

def tiny():
    rng=np.random.RandomState(19)
    t=np.linspace(0,2*np.pi,16)
    x=np.vstack([np.sin(t)+rng.normal(0,.04,16) for _ in range(8)]+
                [np.sign(np.sin(2*t)) + rng.normal(0,.04,16) for _ in range(8)])
    return normalize_rows(x)

@pytest.mark.parametrize("method",READY_IDS)
def test_native_source_and_adapter_partition_parity(method,source_root):
    verify_native(method,source_root)
    x=tiny()
    with seeded(0),threadpool_limits(limits=1):
        native=native_model(method,2,0,source_root)
        native.fit(native_shape(method,x))
    adapted=fit_predict(method,x,2,0,source_root)
    assert np.array_equal(adapted.clusters,native.labels_)
    centers=lambda m:getattr(m,"centroids_",getattr(m,"cluster_centers_",None))
    assert np.array_equal(centers(adapted.model),centers(native))
    assert np.array_equal(infer(method,adapted.model,x,7),native.predict(native_shape(method,x)))

def test_seed_isolation():
    random.seed(91);np.random.seed(91)
    py=random.getstate();n=np.random.get_state()
    with seeded(7):
        assert np.array_equal(np.random.random(3),np.random.RandomState(7).random_sample(3))
    assert random.getstate()==py
    restored=np.random.get_state()
    assert restored[0]==n[0] and np.array_equal(restored[1],n[1]) and restored[2:]==n[2:]

def test_seed_phase_is_stable_and_independent():
    assert phase_seed("a","Toy",0,"head")==phase_seed("a","Toy",0,"head")
    assert phase_seed("a","Toy",0,"head")!=phase_seed("a","Toy",0,"encoder")

@pytest.mark.parametrize("method",["ckm","dtcc_2023","cdcc","tfmcc","fasa","r_clustering"])
def test_no_conditional_promotion(method):
    with pytest.raises(ContractError):method_record(method)

def test_training_signature_has_no_target_membership():
    assert not {"y","labels","ground_truth"}&set(inspect.signature(fit_predict).parameters)

def test_native_settings_are_preserved():
    assert default_config("euclidean_kmeans")["parameters"]["n_init"]==10
    assert default_config("kshape")["parameters"]=={"centroid_init":"zero","max_iter":100,"n_jobs":1}
    p=default_config("kasba")["parameters"]
    assert p["max_iter"]==300 and p["distance"]=="msm" and p["distance_params"]=={"c":1.0}
