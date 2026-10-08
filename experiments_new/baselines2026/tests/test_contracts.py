import ast
import copy
import itertools
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from sklearn.metrics import rand_score, normalized_mutual_info_score
from baselines2026.datasets import load_training_input, load_evaluation_input, normalize_rows
from baselines2026.evaluator import metrics
from baselines2026.predictions import validate_submission, save_submission, load_submission, utilization
from baselines2026.registry import REPOSITORY
from baselines2026.util import ContractError, file_hash, write_json_new

@pytest.mark.parametrize("labels,predicted", [
    ([0,0,1,1],[0,0,1,1]),
    ([0,0,1,1],[7,7,-4,-4]),
    ([0,0,0,0],[1,1,1,1]),
    ([7],[-99]),
])
def test_perfect_and_degenerate(labels,predicted):
    assert metrics(labels,predicted) == {"ARI":1.0,"NMI_arithmetic":1.0,"RI":1.0,"ACC":1.0}

def test_single_cluster():
    result=metrics([0,0,1,1],[0,0,0,0])
    assert result["ARI"] == result["NMI_arithmetic"] == 0
    assert result["RI"] == pytest.approx(1/3)
    assert result["ACC"] == .5

def test_rectangular_hungarian():
    assert metrics([0,0,1,1],[0,0,1,2])["ACC"] == .75

def test_explicit_arithmetic_nmi():
    y=[0,0,0,1,1,1]; p=[0,0,0,0,1,2]
    r=metrics(y,p)["NMI_arithmetic"]
    assert r==normalized_mutual_info_score(y,p,average_method="arithmetic")
    assert r!=pytest.approx(normalized_mutual_info_score(y,p,average_method="geometric"))

@pytest.mark.parametrize("pred",[[0,0,1,1],[0,1,0,1],[0,0,0,0],[0,0,0,1],[0,1,2,3]])
def test_independent_pair_count_and_frozen_utility(pred):
    y=[0,0,1,1]
    agree=sum((y[a]==y[b])==(pred[a]==pred[b]) for a,b in itertools.combinations(range(4),2))/6
    assert metrics(y,pred)["RI"]==agree==rand_score(y,pred)
    path=REPOSITORY/"experiments_new/loster2026/src/loster2026/metrics.py"
    spec=importlib.util.spec_from_file_location("frozen_public_metrics",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert metrics(y,pred)==module.evaluate(y,pred)

@pytest.mark.parametrize("change",["reverse_ids","duplicate_ids","count","dataset","hashes","raw_hash",
                                  "nan","inf","fraction","boolean","string","two_dimensional",
                                  "bad_source","bad_env","bad_config","bool_seed","bad_status"])
def test_reject_malformed(change,toy,payload):
    p=copy.deepcopy(payload)
    if change=="reverse_ids":p["sample_ids"].reverse()
    elif change=="duplicate_ids":p["sample_ids"][1]=p["sample_ids"][0]
    elif change=="count":p["predicted_clusters"].pop()
    elif change=="dataset":p["dataset"]="Other"
    elif change=="hashes":p["input_hashes"]["TRAIN"]="9"*64
    elif change=="raw_hash":p["raw_array_sha256"]="9"*64
    elif change=="nan":p["predicted_clusters"][0]=float("nan")
    elif change=="inf":p["predicted_clusters"][0]=float("inf")
    elif change=="fraction":p["predicted_clusters"][0]=.5
    elif change=="boolean":p["predicted_clusters"]=[False]*4
    elif change=="string":p["predicted_clusters"]=["0"]*4
    elif change=="two_dimensional":p["predicted_clusters"]=[[0]]*4
    elif change=="bad_source":p["source_commit"]="not-a-commit"
    elif change=="bad_env":p["environment_sha256"]="bad"
    elif change=="bad_config":p["config_sha256"]="bad"
    elif change=="bool_seed":p["seed"]=True
    elif change=="bad_status":p["execution_status"]="FAILED"
    with pytest.raises(ContractError):validate_submission(p,toy.identity())

def test_immutable_persistence_and_collapse(tmp_path,toy,payload):
    path=tmp_path/"predictions.json";save_submission(path,payload,toy.identity())
    assert load_submission(path,toy.identity())==payload
    before=file_hash(path)
    with pytest.raises(FileExistsError):save_submission(path,payload,toy.identity())
    assert file_hash(path)==before
    collapsed=copy.deepcopy(payload);collapsed["predicted_clusters"]=[7]*4
    validate_submission(collapsed,toy.identity())
    assert utilization([7]*4,2)["collapsed"]

def _fixtures(tmp_path):
    p=tmp_path/"Toy";p.mkdir()
    (p/"Toy_TRAIN.tsv").write_text("1\t1\t2\t3\n2\t3\t2\t1\n",encoding="utf-8")
    (p/"Toy_TEST.tsv").write_text("2\t2\t0\t1\n1\t0\t2\t1\n",encoding="utf-8")

def test_headerless_pool_ids_and_label_firewall(tmp_path):
    _fixtures(tmp_path)
    t=load_training_input(tmp_path,"Toy",{"N":4,"L":3,"k":2,"TRAIN":2,"TEST":2})
    e=load_evaluation_input(tmp_path,"Toy")
    assert t.sample_ids==("Toy/TRAIN/0","Toy/TRAIN/1","Toy/TEST/0","Toy/TEST/1")
    assert not hasattr(t,"labels") and not hasattr(t,"class_counts")
    assert np.array_equal(e.labels,[0,1,1,0])
    assert t.raw[0].tolist()==[1,2,3]
    assert not t.raw.flags.writeable
    assert t.identity()==e.training_identity
    assert len(t.input_hashes["TRAIN"])==64

def test_wrong_expected_hash(tmp_path):
    _fixtures(tmp_path)
    with pytest.raises(ContractError):
        load_training_input(tmp_path,"Toy",{"input_hashes":{"TRAIN":"0"*64,"TEST":"0"*64}})

@pytest.mark.parametrize("bad",["ragged","nonfinite","header"])
def test_reject_invalid_raw(tmp_path,bad):
    _fixtures(tmp_path)
    text={"ragged":"1\t1\t2\n2\t1\t2\t3\n","nonfinite":"1\t1\tNaN\t3\n",
          "header":"label\tx1\tx2\tx3\n1\t1\t2\t3\n"}[bad]
    (tmp_path/"Toy/Toy_TRAIN.tsv").write_text(text,encoding="utf-8")
    with pytest.raises(ContractError):load_training_input(tmp_path,"Toy")

@pytest.mark.parametrize("bad",[np.ones((3,4)),np.array([[0,np.nan,2]]),np.ones((3,1))])
def test_reject_undefined_normalization(bad):
    with pytest.raises(ContractError):normalize_rows(bad)

def test_population_normalization_matches_frozen_public_function():
    x=np.array([[0,1,5,2],[3,-1,2,9]],dtype=float)
    path=REPOSITORY/"experiments_new/loster2026/src/loster2026/data.py"
    node=next(n for n in ast.parse(path.read_text(encoding="utf-8-sig")).body
              if isinstance(n,ast.FunctionDef) and n.name=="normalize_rows")
    namespace={"np":np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),"exec"),namespace)
    assert np.array_equal(normalize_rows(x),namespace["normalize_rows"](x))


def test_int64_cluster_boundaries():
    from baselines2026.predictions import integer_clusters
    limits = np.array([-(2**63), 2**63-1], dtype=np.int64)
    np.testing.assert_array_equal(integer_clusters(limits, 2), limits)
    with pytest.raises(ContractError):
        integer_clusters(np.array([2**63], dtype=np.uint64), 1)
    with pytest.raises(ContractError):
        integer_clusters(np.array([float(2**63)]), 1)
