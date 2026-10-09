"""Use native training; remove label diagnostics while preserving their execution effects."""
import ast,types,csv,time
from contextlib import nullcontext
from pathlib import Path
import numpy as np,torch
from ..util import ContractError
from .sources import load,verify
from .runtime import seeded
CONFIG=dict(batch_size=8,output_dims=64,hidden_dims=64,depth=10,pretraining_epoch=100,MaxIter=30,lr=.001,m=1.5,w_c=.2,hard_w=.2,device="cuda")
class DiagnosticFilter(ast.NodeTransformer):
    def visit_Assign(self,node):
        if any(isinstance(t,ast.Name) and t.id in ("ACC","NMI") for t in node.targets):return None
        return self.generic_visit(node)
    def visit_Expr(self,node):
        if isinstance(node.value,ast.Call):
            call=node.value
            if isinstance(call.func,ast.Name) and call.func.id=="print":return None
            if isinstance(call.func,ast.Attribute) and call.func.attr=="save" and isinstance(call.func.value,ast.Name) and call.func.value.id=="np":return None
        return self.generic_visit(node)
    def visit_If(self,node):
        node=self.generic_visit(node)
        if not node.body:node.body=[ast.Pass()]
        return node
    def visit_Return(self,node):return ast.Return(value=ast.Tuple(elts=[ast.Constant(None),ast.Constant(None)],ctx=ast.Load()))
def diagnostic_functions(native,root):
    path,_=verify("fcacc",root);tree=ast.parse((path/"fcacc.py").read_text(encoding="utf-8"))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="FCACCModel")
    functions={}
    for name in ("Kmeans_model_evaluation","model_evaluation"):
        f=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
        f=DiagnosticFilter().visit(f);module=ast.fix_missing_locations(ast.Module(body=[f],type_ignores=[]))
        ns=dict(native.__dict__);exec(compile(module,"<native-label-free-diagnostic>","exec"),ns);functions[name]=ns[name]
    return functions

def create(x,k,root,work,toy=False):
    native,_=load("fcacc",root)
    # Dummy targets satisfy the native tuple API; actual class memberships never enter it.
    loader=native.datautils.create_data_loader(x,np.zeros(len(x),dtype=np.int64),np.arange(len(x)),8)
    config=dict(CONFIG)
    if toy:config.update(pretraining_epoch=1,MaxIter=2)
    model=native.FCACCModel(loader,dataset_size=len(x),timesteps_len=x.shape[1],n_cluster=k,dataset_name=str(Path(work)/"native"),input_dims=1,log_dir=str(work),**config)
    for name,f in diagnostic_functions(native,root).items():setattr(model,name,types.MethodType(f,model))
    # Stop on nonfinite native values; no repair, clamping, skipped batches or reseeding.
    for name in ("contrastive_loss","cmp"):
        original=getattr(model,name)
        def checked(*a,_original=original,**kw):
            value=_original(*a,**kw)
            if not torch.isfinite(value).all():raise FloatingPointError("Nonfinite native FCACC objective/membership")
            return value
        setattr(model,name,checked)
    return model

def ordered_inference(model,loader=None):
    model.encoder.eval();N=model.dataset_size
    labels=np.full(N,-1,dtype=np.int64);reps=np.zeros((N,model.latent_size));seen=np.zeros(N,dtype=np.int64);order=[]
    # Preserve original allocations and extra raw-encoder forwards. No memory optimization.
    data=np.zeros((N,model.timesteps_len,model.input_dims));unused=np.zeros((N,model.timesteps_len,model.latent_size))
    for x,_,indices in (loader or model.data_loader):
        idx=indices.cpu().numpy()
        if idx.dtype.kind not in "iu" or np.any(idx<0) or np.any(idx>=N):raise ContractError("Invalid canonical index")
        x=x.to(model.device);u=model.encode_with_pooling(x)
        features=model.encoder(x)
        unused[idx]=features.detach().cpu().numpy();data[idx]=x.cpu().numpy()
        p=model.cmp(u.unsqueeze(0).repeat(model.num_cluster,1,1),model.u_mean)
        labels[idx]=p.argmax(dim=1).cpu().numpy();reps[idx]=u.detach().cpu().numpy();np.add.at(seen,idx,1);order.extend(idx.tolist())
    model.encoder.train()
    if not np.all(seen==1):raise ContractError("Inference must cover each canonical row exactly once")
    if not np.isfinite(reps).all() or not np.isfinite(unused).all():raise FloatingPointError("Nonfinite native inference")
    return labels,reps,{"visited_indices":order,"unique_complete":True,"unused_representation_allocation_bytes":unused.nbytes}

def fit(x,k,seed,root,work,timer=None,toy=False):
    stage=timer.stage if timer else lambda _:nullcontext()
    with seeded(seed):
        with stage("initialization"):model=create(x,k,root,work,toy)
        with stage("pretraining"):model.Pretraining()
        original_init=model.initialization;center_time=[]
        def measured_init():
            torch.cuda.synchronize();start=time.perf_counter();result=original_init();torch.cuda.synchronize();center_time.append(time.perf_counter()-start);return result
        model.initialization=measured_init
        with stage("joint_training_including_center_initialization"):model.Finetuning()
        stage_losses={}
        for name,epochs in (("pretraining",model.pretraining_epoch),("finetuning",model.MaxIter1)):
            with open(str(Path(work)/(name+".csv")),newline="") as stream:rows=list(csv.reader(stream))
            values=np.array([float(v) for v in rows[1]])
            if len(values)!=epochs or not np.isfinite(values).all():raise FloatingPointError("Incomplete/nonfinite native epoch losses")
            stage_losses[name]=values.tolist()
        # Native batch runner reloads raw final encoder; SWA net remains the active inference path.
        with stage("inference"):
            model.encoder=torch.load(str(Path(work)/"native_Finetuning_phase"))
            labels,reps,order=ordered_inference(model)
    return model,None,reps,labels,{"completed_pretraining_epochs":model.pretraining_epoch,"completed_joint_epochs":model.MaxIter1,"pretraining_steps":model.pretraining_epoch*len(model.data_loader),"joint_steps":model.MaxIter1*len(model.data_loader),"SWA_n_averaged":int(model.net.n_averaged),"center_seed":0,"center_kmeans_n_init":"native sklearn1.3.2 default warn ->10","diagnostics":"label metrics/files omitted; native forward/shuffle/KMeans/mode effects retained","prediction_order":order,"toy_override":toy,"center_initialization_seconds":center_time[0],"epoch_losses":stage_losses}
