import ast,copy,inspect,os
from pathlib import Path
import numpy as np,pytest,torch
from torch.utils.data import DataLoader,TensorDataset
from baselines2026.datasets import normalize_rows
from baselines2026.provenance import environment,phase_seed
from baselines2026.registry import BASELINE_ROOT
from baselines2026.util import ContractError,array_hash,write_json_new
from baselines2026.timing import PipelineTimer
from baselines2026.resources import ResourceMonitor
from baselines2026.wave2a.sources import verify,load
from baselines2026.wave2a.runtime import seeded,cuda_probe,GPUResource
from baselines2026.wave2a.runner import lock
from baselines2026.wave2a.checkpoints import save,replay,validate,state_hash
METHOD=os.environ["WAVE2_METHOD"];ROOT=os.environ["BASELINE_SOURCE_ROOT"];OUTPUT=Path(os.environ["WAVE2_OUTPUT"])
def tiny():
    rng=np.random.RandomState(29);t=np.linspace(0,2*np.pi,16)
    return normalize_rows(np.vstack([np.sin((1+i//8)*t)+rng.normal(0,.1,16) for i in range(16)])).astype(np.float32)[:,:,None]

def test_source_clean_and_exact():
    _,info=verify(METHOD,ROOT);assert len(info["source_commit"])==40 and info["verified_files"]
def test_environment_exact():
    info,h=environment(lock(METHOD));assert len(h)==64
    packages=dict(info["packages"]);assert packages["torch"]==("1.8.1+cu111" if METHOD=="ts2vec_kmeans" else "1.10.0+cu113")
def test_actual_cuda_matmul():
    result=cuda_probe();assert result["compute_capability"]==[6,1] and result["tensor_matmul_exact"]
    write_json_new(OUTPUT/"cuda.json",result)
def test_native_cpu_encoder():
    native,_=load(METHOD,ROOT)
    cls=native.TSEncoder if METHOD=="ts2vec_kmeans" else native.FCACCEncoder
    m=cls(input_dims=1,output_dims=320 if METHOD=="ts2vec_kmeans" else 64,hidden_dims=64,depth=10).eval()
    x=torch.from_numpy(tiny());y=m(x.clone());assert y.shape[:2]==x.shape[:2] and torch.isfinite(y).all()
    y.sum().backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
@pytest.fixture(scope="module")
def runs():
    x=tiny();all_runs=[];old=os.getcwd()
    try:
        for i,seed in enumerate((0,0,1)):
            folder=OUTPUT/("tiny-"+str(i));folder.mkdir();os.chdir(str(folder));timer=PipelineTimer(torch.cuda.synchronize)
            with ResourceMonitor() as cpu,GPUResource() as gpu:
                with timer.pipeline():
                    if METHOD=="ts2vec_kmeans":
                        from baselines2026.wave2a.ts2vec_adapter import fit
                        model,head,reps,pred,d=fit(x,2,seed,"SyntheticTinyWave2A",ROOT,timer,toy_iters=3)
                    else:
                        from baselines2026.wave2a.fcacc_adapter import fit
                        model,head,reps,pred,d=fit(x,2,seed,ROOT,folder,timer,toy=True)
            checkpoint=save(folder,METHOD,model,head,d,x,seed,2)
            restored,rr=replay(folder,METHOD,x,ROOT)
            assert np.array_equal(pred,restored) and np.array_equal(reps,rr)
            result={"seed":seed,"detail":d,"active_SWA_sha256":checkpoint["active_SWA_sha256"],"representation_sha256":array_hash(reps),"predictions":pred.tolist(),"time":timer.total,"stage_seconds":timer.stages,"CPU":cpu.result(),"GPU":gpu.result,"reload_exact":True}
            write_json_new(folder/"result.json",result);all_runs.append((model,checkpoint,reps,pred,d,result))
    finally:os.chdir(old)
    return all_runs

def test_native_training_shapes_and_complete_toy_budget(runs):
    for model,c,r,p,d,_ in runs:
        assert r.shape==(16,320 if METHOD=="ts2vec_kmeans" else 64) and len(p)==16 and np.isfinite(r).all()
        assert int(model.net.n_averaged)==(4 if METHOD=="ts2vec_kmeans" else 7)
        assert d.get("completed_steps",3)==3

def test_fixed_seed_replay_and_changed_stochastic_state(runs):
    a,b,c=runs
    assert a[1]["active_SWA_sha256"]==b[1]["active_SWA_sha256"] and np.array_equal(a[2],b[2]) and np.array_equal(a[3],b[3])
    assert a[1]["active_SWA_sha256"]!=c[1]["active_SWA_sha256"] and not np.array_equal(a[2],c[2])

def test_active_SWA_is_not_raw_encoder(runs):
    c=runs[0][1];assert c["active_SWA_sha256"]!=c["raw_encoder_sha256"]
    weights={k:v for k,v in c["active_SWA"].items() if k!="n_averaged"}
    assert any(not torch.equal(v,c["raw_encoder"][k[len("module."):]]) for k,v in weights.items())
@pytest.mark.parametrize("corruption",["missing","raw_substitution","wrong_weight","wrong_input"])
def test_bad_active_checkpoints_are_rejected(runs,corruption):
    c=copy.deepcopy(runs[0][1])
    if corruption=="missing":del c["active_SWA"]
    if corruption=="raw_substitution":c["active_SWA"]=copy.deepcopy(c["raw_encoder"])
    if corruption=="wrong_weight":
        key=next(k for k in c["active_SWA"] if k!="n_averaged");c["active_SWA"][key].add_(1)
    if corruption=="wrong_input":c["input_sha256"]="0"*64
    with pytest.raises(ContractError):validate(c,METHOD,tiny())

def test_full_pipeline_timer_and_resources(runs):
    for *_,r in runs:
        assert r["time"]>0 and sum(r["stage_seconds"].values())<=r["time"]+.001
        assert r["GPU"]["peak_allocated_bytes"]>0 and r["CPU"]["host_process_tree_peak_rss_sampled_bytes"]>0

def test_native_correspondence_and_order(runs):
    x=tiny();native,_=load(METHOD,ROOT)
    if METHOD=="ts2vec_kmeans":
        from baselines2026.wave2a.ts2vec_adapter import CONFIG
        with seeded(0):
            __import__("utils").init_dl_program("cuda:0",seed=0,deterministic=True)
            m=native.TS2Vec(**CONFIG);m.fit(x,n_iters=3);r=m.encode(x,encoding_window="full_series")
        assert state_hash(m.net.state_dict())==runs[0][1]["active_SWA_sha256"] and np.array_equal(r,runs[0][2])
        assert runs[0][4]["head_seed"]==phase_seed(METHOD,"SyntheticTinyWave2A",0,"kmeans")
        m.net.eval()
        loader=DataLoader(TensorDataset(torch.from_numpy(x).float()),batch_size=8)
        with seeded(0),torch.no_grad():manual=np.concatenate([torch.max(m.net(batch[0].cuda()),dim=1).values.cpu().numpy() for batch in loader])
        assert np.array_equal(r,manual)
    else:
        from baselines2026.wave2a.fcacc_adapter import ordered_inference,DiagnosticFilter
        m=runs[0][0]
        # Deliberately shuffled/noncontiguous batches; canonical scatter is exact.
        order=np.array([7,1,12,3,14,0,5,15,8,2,13,6,11,4,10,9]);t=torch.from_numpy(x[order]);idx=torch.tensor(order)
        loader=DataLoader(TensorDataset(t,torch.zeros(16,dtype=torch.long),idx),batch_size=8)
        pred,r,details=ordered_inference(m,loader)
        assert np.array_equal(pred,runs[0][3]) and details["unique_complete"]
        oracle=np.zeros_like(r);oracle_pred=np.zeros_like(pred)
        for xx,_,ii in loader:
            u=m.encode_with_pooling(xx.cuda());oracle[ii.numpy()]=u.cpu().numpy()
            pp=m.cmp(u.unsqueeze(0).repeat(m.num_cluster,1,1),m.u_mean)
            oracle_pred[ii.numpy()]=pp.argmax(dim=1).cpu().numpy()
        assert np.array_equal(r,oracle) and np.array_equal(pred,oracle_pred)
        bad=DataLoader(TensorDataset(t,torch.zeros(16,dtype=torch.long),torch.zeros(16,dtype=torch.long)),batch_size=8)
        with pytest.raises(ContractError):ordered_inference(m,bad)
        # Native diagnostic removal preserves loader, forward calls, KMeans and mode transitions.
        path,_=verify(METHOD,ROOT);tree=ast.parse((path/"fcacc.py").read_text(encoding="utf-8"))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
        for name in ("Kmeans_model_evaluation","model_evaluation"):
            f=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name);filtered=DiagnosticFilter().visit(copy.deepcopy(f))
            calls=lambda n:[ast.dump(c) for c in ast.walk(n) if isinstance(c,ast.Call) and (isinstance(c.func,ast.Attribute) and c.func.attr in ("encode_with_pooling","train","eval","fit") or isinstance(c.func,ast.Name) and c.func.id=="KMeans")]
            assert calls(f)==calls(filtered)

def test_ground_truth_isolation_api_and_native_global_constants():
    module=__import__("baselines2026.wave2a."+("ts2vec_adapter" if METHOD=="ts2vec_kmeans" else "fcacc_adapter"),fromlist=["fit"])
    assert not {"y","labels","targets","ground_truth"} & set(inspect.signature(module.fit).parameters)
    path,_=verify(METHOD,ROOT)
    if METHOD=="fcacc":
        source=(path/"fcacc.py").read_text(encoding="utf-8");assert "random_state=0" in source and "lr=0.0001" in source
        assert "loss = loss_r + self.w_c * loss_c" in source
    else:
        source=(path/"ts2vec.py").read_text(encoding="utf-8");assert "200 if train_data.size <= 100000 else 600" in source

def test_fcacc_diagnostic_adapter_execution_equivalence(runs):
    if METHOD!="fcacc":
        # TS2Vec's direct official-fit equivalence is checked above.
        assert runs[0][4]["completed_steps"]==3
        return
    from baselines2026.wave2a.fcacc_adapter import create,ordered_inference
    native,_=load(METHOD,ROOT);folder=OUTPUT/"native-diagnostic-oracle";folder.mkdir();(folder/"features").mkdir()
    previous=os.getcwd();old_acc,old_nmi=native.acc,native.nmi
    try:
        os.chdir(str(folder));native.acc=lambda *a:None;native.nmi=lambda *a:None
        with seeded(0):
            m=create(tiny(),2,ROOT,folder,toy=True)
            import types
            for name in ("Kmeans_model_evaluation","model_evaluation"):
                setattr(m,name,types.MethodType(getattr(native.FCACCModel,name),m))
            m.Pretraining();m.Finetuning()
            m.encoder=torch.load(str(folder/"native_Finetuning_phase"))
            pred,reps,_=ordered_inference(m)
        assert state_hash(m.net.state_dict())==runs[0][1]["active_SWA_sha256"]
        assert torch.equal(m.u_mean.cpu(),runs[0][1]["centers"].cpu())
        assert np.array_equal(pred,runs[0][3]) and np.array_equal(reps,runs[0][2])
        write_json_new(folder/"parity.json",{"native_diagnostic_execution_vs_filtered":"SWA,centers,ordered representations,predictions exact","labels":"constant API placeholders; metric stubs returnNone; no label selection"})
    finally:os.chdir(previous);native.acc,native.nmi=old_acc,old_nmi

def test_published_objective_gradient_correspondence():
    if METHOD=="ts2vec_kmeans":
        native,_=load(METHOD,ROOT)
        z1=torch.randn(8,8,320,device="cuda",requires_grad=True);z2=torch.randn_like(z1,requires_grad=True)
        loss=native.hierarchical_contrastive_loss(z1,z2);loss.backward()
        assert z1.grad is not None and z2.grad is not None and torch.isfinite(z1.grad).all()
        return
    from baselines2026.wave2a.fcacc_adapter import create
    with seeded(0):
        model=create(tiny(),2,ROOT,OUTPUT/"gradient-probe-unused",toy=True)
        x=torch.from_numpy(tiny()).cuda()
        u=model.encode_with_pooling(x)
        p=model.cmp(u.unsqueeze(0).repeat(model.num_cluster,1,1),model.u_mean).detach().T.pow(model.m)
        loss_c=0
        for i in range(model.num_cluster):
            uu=model.encode_with_pooling(x)
            loss_c=loss_c+torch.matmul(p[i].unsqueeze(0),((uu-model.u_mean[i].unsqueeze(0).repeat(len(x),1))**2).sum(dim=1))
        result={"paper_equations":[8,11],"pooled_representation_requires_grad":u.requires_grad,"native_cluster_term_requires_grad":loss_c.requires_grad,"raw_encoder_parameter_count":sum(v.numel() for v in model.encoder.parameters()),"native_graph_reaches_raw_encoder":False,"classification":"SOURCE/MANUSCRIPT CORRESPONDENCE BLOCKER; no scientific repair authorized","real_dataset_fit_permitted":False}
        write_json_new(OUTPUT/"published-loss-gradient-probe.json",result)
        assert loss_c.requires_grad, "Published joint clustering term has no encoder gradient in released source"
        gradients=torch.autograd.grad(loss_c,tuple(model.encoder.parameters()),allow_unused=True)
        assert any(g is not None and torch.any(g!=0) for g in gradients)
