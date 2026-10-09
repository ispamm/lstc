import ast,warnings
import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from baselines2026.datasets import normalize_rows
from baselines2026.provenance import seeded,environment
from baselines2026.registry import BASELINE_ROOT,method_record
from baselines2026.util import ContractError,array_hash
from baselines2026.wave1b.sources import verify,r_cells,load_r,load_fasa,pipeline_statements,STEPS
from baselines2026.wave1b.r_clustering import seed_numba,compiled_draws,pca_dimensions,fit_pipeline,NativeDomainError
from baselines2026.wave1b.fasa import fit,membership_labels
from baselines2026.wave1b.synthetic import tiny

@pytest.mark.parametrize("method",["r_clustering","fasa"])
def test_frozen_sources_and_historical_admission(method,source_root):
    _,info=verify(method,source_root);assert info["verified_files"]
    with pytest.raises(ContractError):method_record(method)
@pytest.mark.parametrize("method",["ckm","cdcc","dtcc_2023","ts2vec","kasba"])
def test_no_other_method_admitted(method,source_root):
    with pytest.raises(ContractError):verify(method,source_root)
def test_only_exact_scientific_notebook_definitions_and_AST(source_root,tmp_path):
    cell,_,_=r_cells(source_root);native,derived=load_r(source_root,tmp_path)
    assert open(derived["path"],encoding="utf-8").read()=="import numpy as np\n"+cell
    stmts=pipeline_statements(source_root)
    assert tuple(s.targets[0].id for s in stmts)==STEPS
    assert not {"Y","y","metrics","to_excel","load_classification_with_fixes"} & {n.id for s in stmts for n in ast.walk(s) if isinstance(n,ast.Name)}
    assert native._fit_biases.targetoptions["fastmath"] and native.transform.targetoptions["parallel"]
def test_compiled_rng_independent_and_seeded():
    seed_numba(17);a=compiled_draws(12);np.random.seed(17);b=compiled_draws(12)
    assert not np.array_equal(a,b)
    seed_numba(17);assert np.array_equal(a,compiled_draws(12))
    seed_numba(18);assert not np.array_equal(a,compiled_draws(12))
def test_compiled_seed_controls_actual_biases_and420(source_root,tmp_path):
    native,_=load_r(source_root,tmp_path);x=normalize_rows(tiny()).astype(np.float32)
    def run(seed):
        with seeded(0):seed_numba(seed);return native.fit(x,num_features=500)
    a=run(3);b=run(3);c=run(4)
    assert np.array_equal(a[2],b[2]) and not np.array_equal(a[2],c[2])
    assert len(a[2])==420 and 84*sum(a[1])==420
def test_full_native_notebook_pipeline_state_equivalence(source_root,tmp_path):
    x=normalize_rows(tiny());native,_=load_r(source_root,tmp_path);captured=[]
    def capture(*a,**k):
        m=KMeans(*a,**k);captured.append(m);return m
    ns={"np":np,"X":x.astype(np.float32),"fit":native.fit,"transform":native.transform,"num_features":500,"num_clusters":2,"StandardScaler":StandardScaler,"PCA":PCA,"KMeans":capture}
    with seeded(0),threadpool_limits(limits=1):
        seed_numba(0);exec(compile(ast.Module(body=pipeline_statements(source_root),type_ignores=[]),"<original-cell12>","exec"),ns)
    m,d=fit_pipeline(x,2,0,source_root,tmp_path)
    assert np.array_equal(m.labels_,ns["labels_pred"]) and np.array_equal(m.parameters[2],ns["parameters"][2])
    assert m.features_sha256==array_hash(ns["transformed_data"])
    assert np.array_equal(m.pca.components_,ns["pca_optimal"].components_)
    assert np.array_equal(m.kmeans.cluster_centers_,captured[0].cluster_centers_)
    assert d["effective_features"]==420 and d["kmeans_parameters"]["n_init"]==10
    assert d["kmeans_parameters"]["random_state"] is None
    assert np.array_equal(m.predict(x,native),m.labels_)
@pytest.mark.parametrize("ratios,n",[([.6,.25,.1,.04,.009,.001],4),([.5,.3,.2],0),([.005,.004],0),([np.nan,np.nan],0),([.01,.009],1)])
def test_exact_PCA_rule_no_clamp(ratios,n,source_root):assert pca_dimensions(ratios,source_root)==n
def test_native_zero_PC_input_stops_without_repair(source_root,tmp_path):
    x=normalize_rows(np.tile(np.sin(np.linspace(0,2*np.pi,64)),(8,1)))
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        with pytest.raises(NativeDomainError,match="zero components"):fit_pipeline(x,2,0,source_root,tmp_path)
    assert any("invalid" in str(w.message) for w in seen)
def test_exact_new_environment_lock(tmp_path):
    info,h=environment(BASELINE_ROOT/"configs/requirements-wave1b.lock.txt")
    assert len(info["packages"])==20 and len(h)==64
    p=tmp_path/"bad.txt";p.write_text("numpy==0\n")
    with pytest.raises(ContractError):environment(p)
def test_FASA_native_centers_order_and_export(source_root):
    x=normalize_rows(tiny());native=load_fasa(source_root)
    with seeded(0):original=native.FASA_I_I(x[:,:,None],2,centroid_init="zero",max_iter=100)
    m,d=fit(x,2,0,source_root)
    assert np.array_equal(m.labels_,membership_labels(original,len(x)))
    assert np.array_equal(np.asarray([v[0] for v in m.clusters]),np.asarray([v[0] for v in original]))
    assert np.array_equal(m.replay_membership(),m.labels_) and not d["itr_is_seed"]
def test_member_scatter_preserves_record_order():assert membership_labels([(None,[3,0]),(None,[2,1])],4).tolist()==[0,1,1,0]
@pytest.mark.parametrize("members",[[[0,1],[1,2,3]],[[0,1],[2]],[[0,1],[2,4]],[[0,1],[2,-1]],[[0,1],[2,3.0]],[[0,1],[2,True]],[[0,1],[2,"3"]]])
def test_bad_member_coverage_or_indices(members):
    with pytest.raises(ContractError):membership_labels([(None,v) for v in members],4)
def test_native_zero_norm_distance(source_root):
    n=load_fasa(source_root);assert np.array_equal(n._ncc_c_3dim(np.zeros((16,1)),np.arange(16)[:,None]),np.zeros(31))
def test_native_cancelling_centroid_warning_preserved(source_root):
    n=load_fasa(source_root);v=np.array([1.,-1.,1.,-1.]);x=np.array([v,-v])[:,:,None]
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always");center=n._extract_shape(np.array([0,0]),x,0,np.zeros((4,1)))
    assert np.array_equal(center,np.zeros(4)) and any("invalid" in str(w.message) for w in seen)
def test_native_empty_cluster_recovery_seeded(source_root):
    n=load_fasa(source_root);x=normalize_rows(tiny())[:,:,None]
    with seeded(7):a=n._extract_shape(np.zeros(len(x),dtype=int),x,1,np.zeros((64,1)))
    with seeded(7):b=n._extract_shape(np.zeros(len(x),dtype=int),x,1,np.zeros((64,1)))
    assert np.array_equal(a,b) and any(np.array_equal(a,row[:,0]) for row in x)
def test_actual_native_initial_draw_replays_and_changes_without_itr(source_root):
    n=load_fasa(source_root);original=n.randint;draws=[]
    def spy(*a,**k):
        v=original(*a,**k);draws.append(v.copy());return v
    n.randint=spy
    try:
        for seed in (0,0,1):fit(normalize_rows(tiny()),2,seed,source_root)
    finally:n.randint=original
    assert np.array_equal(draws[0],draws[1]) and not np.array_equal(draws[0],draws[2])


def test_derived_path_short_and_full_content_hash_preserved(source_root,tmp_path):
    from pathlib import Path
    native,detail=load_r(source_root,tmp_path)
    assert len(Path(detail["path"]).parent.name)==16
    assert len(detail["sha256"])==64 and detail["cell10_exact"]
