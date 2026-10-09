"""Verify every external Git blob; load one native method per worker."""
import hashlib,importlib,subprocess,sys
from pathlib import Path
from ..util import ContractError,file_hash
PINS={
 "ts2vec_kmeans":("zhihanyue__ts2vec","b0088e14a99706c05451316dc6db8d3da9351163","https://github.com/zhihanyue/ts2vec","MIT"),
 "fcacc":("Du-Team__FCACC","78b5e5a138ed8c83fea64e5b6668a5cded79d2dc","https://github.com/Du-Team/FCACC","NOT STATED")}
def verify(method,root):
    if method not in PINS:raise ContractError("Only two Wave2A methods admitted")
    name,sha,url,license=PINS[method];path=Path(root)/name
    def git(*args):return subprocess.check_output(["git","-C",str(path),*args],text=True).strip()
    if git("rev-parse","HEAD")!=sha or git("status","--porcelain"):raise ContractError("Source pin/cleanliness failed")
    if git("remote","get-url","origin").rstrip("/").replace(".git","")!=url:raise ContractError("Source origin mismatch")
    files={}
    for rel in git("ls-files").splitlines():
        if git("hash-object",rel)!=git("rev-parse",sha+":"+rel):raise ContractError("Source blob differs "+rel)
        files[rel]=file_hash(path/rel)
    return path,{"source_commit":sha,"source_repository":url,"license":license,"verified_files":files}
def load(method,root):
    path,info=verify(method,root)
    other="fcacc" if method=="ts2vec_kmeans" else "ts2vec"
    if other in sys.modules:raise ContractError("Native top-level imports require separate workers")
    sys.path.insert(0,str(path))
    name="ts2vec" if method=="ts2vec_kmeans" else "fcacc"
    native=importlib.import_module(name)
    if Path(native.__file__).resolve()!= (path/(name+".py")).resolve():raise ContractError("Wrong native import")
    return native,info
