"""Direct official TS2Vec fit/encode plus a predeclared Lloyd head."""
from contextlib import nullcontext
import importlib
import numpy as np,torch,sklearn
from sklearn.cluster import KMeans
from ..provenance import phase_seed
from ..util import ContractError,array_hash
from .sources import load
from .runtime import seeded
CONFIG=dict(input_dims=1,output_dims=320,hidden_dims=64,depth=10,device="cuda",lr=.001,batch_size=8,max_train_length=3000,temporal_unit=0)
def fit(x,k,seed,dataset,root,timer=None,toy_iters=None):
    native,_=load("ts2vec_kmeans",root);stage=timer.stage if timer else lambda _:nullcontext()
    losses=[]
    def callback(model,loss):
        if not np.isfinite(loss):raise FloatingPointError("Nonfinite native loss")
        losses.append(float(loss))
    with seeded(seed):
        importlib.import_module("utils").init_dl_program("cuda:0",seed=seed,deterministic=True)
        with stage("initialization"):model=native.TS2Vec(**CONFIG,after_iter_callback=callback)
        with stage("representation_training"):epoch_losses=model.fit(x,n_iters=toy_iters,verbose=True)
        with stage("full_series_extraction"):reps=model.encode(x,encoding_window="full_series")
        if reps.shape!=(len(x),320) or not np.isfinite(reps).all():raise ContractError("Invalid full-series representation")
        head_seed=phase_seed("ts2vec_kmeans",dataset,seed,"kmeans")
        # sklearn0.24 calls the identical Lloyd algorithm 'full'; 'lloyd' added later.
        algorithm="full" if sklearn.__version__.startswith("0.24.") else "lloyd"
        with stage("downstream_kmeans"):
            head=KMeans(n_clusters=k,init="k-means++",n_init=10,algorithm=algorithm,max_iter=300,tol=1e-4,random_state=head_seed).fit(reps)
    detail={"completed_epochs":model.n_epochs,"completed_steps":model.n_iters,"native_budget":200 if x.size<=100000 else 600,"toy_override":toy_iters,"root_seed":seed,"native_RNG_seeds":{"python":seed,"numpy":seed+1,"torch_cpu":seed+2,"torch_cuda":seed+3},"head_seed":head_seed,"head_parameters":head.get_params(),"head_algorithm_semantics":"Lloyd","losses":losses,"epoch_losses":epoch_losses,"representation_sha256":array_hash(reps),"SWA_n_averaged":int(model.net.n_averaged)}
    return model,head,reps,head.labels_.copy(),detail
