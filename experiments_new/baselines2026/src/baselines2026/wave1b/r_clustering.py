"""Execute original notebook scientific AST; add only seed/state instrumentation."""
import ast
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
from numba import njit
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from ..provenance import seeded
from ..util import array_hash
from .sources import load_r,pipeline_statements,STEPS
class NativeDomainError(RuntimeError):
    classification="PROTOCOL INCOMPATIBILITY"
@njit(cache=True)
def seed_numba(seed):np.random.seed(seed)
@njit(cache=True)
def compiled_draws(n):return np.random.random(n)
def pca_dimensions(ratios,root):
    stmt=next(s for s in pipeline_statements(root) if s.targets[0].id=="optimal_dimensions")
    ns={"np":np,"pca":SimpleNamespace(explained_variance_ratio_=np.asarray(ratios))}
    exec(compile(ast.Module(body=[stmt],type_ignores=[]),"<native-cell12>","exec"),ns)
    return int(ns["optimal_dimensions"])
@dataclass
class RModel:
    parameters:object
    scaler:object
    selector:object
    pca:object
    kmeans:object
    labels_:object
    features_sha256:str
    def predict(self,x,native):
        features=native.transform(np.asarray(x,dtype=np.float32),self.parameters)
        return self.kmeans.predict(self.pca.transform(self.scaler.transform(features)))
def fit_pipeline(x,k,seed,root,derived,timer=None):
    stage=timer.stage if timer else lambda _:nullcontext()
    with seeded(seed),threadpool_limits(limits=1):
        with stage("native_definitions_and_JIT"):native,source=load_r(root,derived)
        with stage("compiled_rng_seed"):seed_numba(seed)
        captured=[]
        def capture(*args,**kwargs):
            model=KMeans(*args,**kwargs);captured.append(model);return model
        ns={"np":np,"X":np.asarray(x,dtype=np.float32),"num_features":500,"num_clusters":k,
            "fit":native.fit,"transform":native.transform,"StandardScaler":StandardScaler,"PCA":PCA,"KMeans":capture}
        names=("kernel_bias_fitting","feature_extraction","scaler_initialization","feature_standardization","PCA_selector_fit","native_PCA_selection","selected_PCA_initialization","selected_PCA_fit_transform","native_KMeans_all_restarts")
        for stmt,name in zip(pipeline_statements(root),names):
            target=stmt.targets[0].id
            with stage(name):
                exec(compile(ast.Module(body=[stmt],type_ignores=[]),"<native-cell12>","exec"),ns)
                if target=="optimal_dimensions" and int(ns[target])==0:raise NativeDomainError("Native PCA selector returned zero components; no clamp")
                if target=="transformed_data" and (ns[target].shape!=(len(x),420) or not np.isfinite(ns[target]).all()):raise NativeDomainError("Native feature count/finite representation failed")
        model=RModel(ns["parameters"],ns["sc"],ns["pca"],ns["pca_optimal"],captured[0],ns["labels_pred"].copy(),array_hash(ns["transformed_data"]))
        detail={"requested_features":500,"effective_features":420,"kernel_length":9,"max_dilations":32,
            "dilations":model.parameters[0].tolist(),"features_per_dilation":model.parameters[1].tolist(),
            "biases_sha256":array_hash(model.parameters[2]),"feature_sha256":model.features_sha256,
            "PCA_rule":"native argmax(explained_variance_ratio_ < .01)","selected_components":int(ns["optimal_dimensions"]),
            "PCA_parameters":{**model.pca.get_params(),"n_components":int(ns["optimal_dimensions"])},
            "PCA_actual_solver":model.pca._fit_svd_solver,"kmeans_parameters":model.kmeans.get_params(),
            "kmeans_iterations":int(model.kmeans.n_iter_),"numpy_seed":seed,"numba_seed":seed,
            "RNG_policy":"host and compiled root seed; native global PCA/KMeans stream unchanged",
            "effective_batch_size":None,"derived_source":source,"pipeline_AST_exact":True,
            "zero_PC_policy":"PROTOCOL INCOMPATIBILITY; no clamp","representation_dtype":"float32"}
    return model,detail
