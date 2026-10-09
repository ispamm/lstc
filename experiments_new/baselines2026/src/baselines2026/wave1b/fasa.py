"""Native univariate FASA, no vendored code or alternative centroid policy."""
from dataclasses import dataclass
from contextlib import nullcontext
import numpy as np
from threadpoolctl import threadpool_limits
from ..provenance import seeded
from ..util import ContractError,array_hash
from .sources import load_fasa
def membership_labels(clusters,N):
    labels=np.full(N,-1,dtype=np.int64);seen=np.zeros(N,dtype=np.int64)
    for cluster,(_,members) in enumerate(clusters):
        for i in members:
            if isinstance(i,(bool,np.bool_)) or not isinstance(i,(int,np.integer)) or not 0<=i<N:raise ContractError("Invalid FASA member index")
            seen[i]+=1;labels[i]=cluster
    if not np.all(seen==1):raise ContractError("FASA must cover every canonical row exactly once")
    return labels
@dataclass
class FASAModel:
    clusters:object
    labels_:object
    initialization_sha256:str
    def replay_membership(self):return membership_labels(self.clusters,len(self.labels_))
def fit(x,k,seed,root,timer=None):
    stage=timer.stage if timer else lambda _:nullcontext()
    with seeded(seed),threadpool_limits(limits=1):
        with stage("native_FASA_import"):native=load_fasa(root)
        initial=np.random.RandomState(seed).randint(0,k,size=len(x)) # independent diagnostic, no RNG consumption
        with stage("native_FASA_all_iterations"):clusters=native.FASA_I_I(np.asarray(x)[:,:,None],k,centroid_init="zero",max_iter=100)
        with stage("canonical_member_extraction"):labels=membership_labels(clusters,len(x))
        centers=np.asarray([v[0] for v in clusters])
        if not np.isfinite(centers).all():raise FloatingPointError("Nonfinite native FASA centroid; no repair")
        model=FASAModel(clusters,labels,array_hash(initial))
        detail={"entry_point":"FASA_I_I.FASA_I_I","variant":"univariate FASA; not MUFASA","max_iter":100,
            "centroid_init":"zero","stopping":"native stable assignments/max100","numpy_seed":seed,"itr_is_seed":False,
            "initial_partition_sha256":array_hash(initial),"centroid_sha256":array_hash(centers),
            "zero_centroids":int(sum(np.linalg.norm(c)==0 for c in centers)),"members_unique_and_complete":True,
            "zero_norm_policy":"native den=inf/zscore/nan_to_num preserved; report warnings","effective_batch_size":None,
            "inference_policy":"native membership export; no out-of-sample predictor invented"}
    return model,detail
