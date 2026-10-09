"""Pinned external scientific code; no archive loops or vendored FASA source."""
import ast, hashlib, importlib.util, json, subprocess, sys
from pathlib import Path
from ..registry import AUDIT_REGISTRY
from ..util import ContractError, file_hash
IDS=("r_clustering","fasa")
PINS={"r_clustering":"3ed571eebe9eb8d399a0c995a6917f62d838f0fc","fasa":"9c05d415efee81fca1a87c1e91623db613ea2ecc"}
DIRS={"r_clustering":"jorgemarcoes__R-Clustering","fasa":"TheDatumOrg__MUFASA"}
NOTEBOOK="R_Clustering_on_UCR_Archive.ipynb"
FASA_FILE="Clustering/FASA_I_I/FASA_I_I.py"
STEPS=("parameters","transformed_data","sc","X_std","pca","optimal_dimensions","pca_optimal","transformed_data_pca","labels_pred")
def verify(method,source_root):
    if method not in IDS:raise ContractError("Only explicitly authorized Wave1B methods")
    record=next(r for r in json.loads(AUDIT_REGISTRY.read_text(encoding="utf-8-sig"))["baselines"] if r["method_id"]==method)
    if record["source_commit"]!=PINS[method] or record["readiness"]!="READY WITH CONDITIONS":raise ContractError("Historical admission changed")
    folder=Path(source_root)/DIRS[method]
    def git(*a):return subprocess.check_output(["git","-C",str(folder),*a])
    if git("rev-parse","HEAD").decode().strip()!=PINS[method] or git("status","--porcelain"):raise ContractError("Source not clean at frozen commit")
    if git("remote","get-url","origin").decode().strip().removesuffix(".git").lower()!=record["source_url"].lower():raise ContractError("Source remote mismatch")
    checked={}
    for rel in ([NOTEBOOK] if method=="r_clustering" else [FASA_FILE,"Clustering/Running_baseline_iter_univariate.py"]):
        blob=git("show",PINS[method]+":"+rel)
        if blob.decode().replace("\r\n","\n")!=(folder/rel).read_text(encoding="utf-8").replace("\r\n","\n"):raise ContractError("Scientific source differs from Git blob")
        checked[rel]={"git_blob_sha256":hashlib.sha256(blob).hexdigest(),"working_file_sha256":file_hash(folder/rel)}
    info={"method":method,"source_repository":record["source_url"],"source_commit":PINS[method],"license":record["license"],"verified_files":checked}
    if method=="r_clustering":
        nb=json.loads((folder/NOTEBOOK).read_text(encoding="utf-8"))
        info["cell_sha256"]={str(i):hashlib.sha256("".join(nb["cells"][i]["source"]).encode()).hexdigest() for i in (2,4,8,10,12)}
    return folder,info
def r_cells(root):
    folder,info=verify("r_clustering",root);nb=json.loads((folder/NOTEBOOK).read_text(encoding="utf-8"))
    return "".join(nb["cells"][10]["source"]),"".join(nb["cells"][12]["source"]),info
def pipeline_statements(root):
    _,cell,_=r_cells(root);loop=next(n for n in ast.parse(cell).body if isinstance(n,ast.For))
    selected=[n for n in loop.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in STEPS]
    if tuple(n.targets[0].id for n in selected)!=STEPS:raise ContractError("Native pipeline structure changed")
    if any(isinstance(n,ast.Name) and n.id in ("Y","y","metrics") for s in selected for n in ast.walk(s)):raise ContractError("Labels in scientific fit")
    return selected
def load_r(root,derived_root):
    cell,_,_=r_cells(root);text="import numpy as np\n"+cell;sha=hashlib.sha256(text.encode()).hexdigest()
    folder=Path(derived_root)/sha[:16];folder.mkdir(parents=True,exist_ok=True);path=folder/"r_native.py"
    if path.exists():
        if path.read_text(encoding="utf-8")!=text:raise ContractError("Derived scientific definitions changed")
    else:
        with path.open("x",encoding="utf-8",newline="\n") as s:s.write(text)
    name="_loster2026_r_native_"+sha[:16]
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name],{"path":str(path),"sha256":sha,"cell10_exact":True,"only_added_import":"import numpy as np"}
def load_fasa(root):
    folder,_=verify("fasa",root);path=folder/FASA_FILE;name="_loster2026_fasa_native"
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    if Path(sys.modules[name].__file__).resolve()!=path.resolve():raise ContractError("Wrong FASA source")
    return sys.modules[name]
