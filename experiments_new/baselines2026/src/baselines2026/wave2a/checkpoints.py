"""Persist and validate the deployed averaged network, raw encoder and cluster state."""
import pickle
from pathlib import Path
import numpy as np,torch
from ..util import ContractError,array_hash,digest
from .runtime import seeded
from .sources import load
from .ts2vec_adapter import CONFIG as TS_CONFIG
from .fcacc_adapter import create,ordered_inference

def state_hash(state):return digest({k:array_hash(v.detach().cpu().numpy()) for k,v in state.items()})
def save(folder,method,model,head,detail,x,seed,k):
    raw=model._net if method=="ts2vec_kmeans" else model.encoder
    c={"schema":"baselines2026.wave2a-checkpoint.v1","method":method,"root_seed":seed,"k":k,"input_sha256":array_hash(x),"raw_encoder":raw.state_dict(),"active_SWA":model.net.state_dict(),"raw_mode":raw.training,"SWA_mode":model.net.training,"detail":detail,"centers":head.cluster_centers_ if head is not None else model.u_mean.detach().cpu()}
    c["representation_configuration"]={"pooling":"official full_series max", "batch_size":model.batch_size, "output_dims":320 if head is not None else model.latent_size,"sliding_length":None,"mask":"native eval allTrue"}
    c["active_SWA_sha256"]=state_hash(c["active_SWA"]);c["raw_encoder_sha256"]=state_hash(c["raw_encoder"])
    torch.save(c,str(Path(folder)/"checkpoint.pt"))
    if head is not None:
        with (Path(folder)/"head.pkl").open("xb") as f:pickle.dump(head,f,protocol=4)
    return c

def validate(c,method,x):
    if c.get("schema")!="baselines2026.wave2a-checkpoint.v1" or c.get("method")!=method or c.get("input_sha256")!=array_hash(x):raise ContractError("Checkpoint identity mismatch")
    for state in ("active_SWA","raw_encoder"):
        if state not in c or state_hash(c[state])!=c.get(state+"_sha256"):raise ContractError("Missing/wrong active model state: "+state)
    if "n_averaged" not in c["active_SWA"] or int(c["active_SWA"]["n_averaged"])<1:raise ContractError("Missing SWA averaging state")

def replay(folder,method,x,root):
    folder=Path(folder);c=torch.load(str(folder/"checkpoint.pt"),map_location="cuda");validate(c,method,x)
    with seeded(c["root_seed"]):
        if method=="ts2vec_kmeans":
            native,_=load(method,root);model=native.TS2Vec(**TS_CONFIG)
            model._net.load_state_dict(c["raw_encoder"],strict=True);model.net.load_state_dict(c["active_SWA"],strict=True)
            model._net.train(c["raw_mode"]);model.net.train(c["SWA_mode"])
            reps=model.encode(x,encoding_window="full_series")
            with (folder/"head.pkl").open("rb") as f:head=pickle.load(f)
            if not np.array_equal(head.cluster_centers_,c["centers"]):raise ContractError("Wrong head centers")
            labels=head.predict(reps)
        else:
            model=create(x,c["k"],root,folder/"replay-unused")
            model.encoder.load_state_dict(c["raw_encoder"],strict=True);model.net.load_state_dict(c["active_SWA"],strict=True)
            model.encoder.train(c["raw_mode"]);model.net.train(c["SWA_mode"]);model.u_mean=c["centers"].to(model.device)
            labels,reps,_=ordered_inference(model)
    return labels,reps
