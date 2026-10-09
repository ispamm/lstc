"""Device validation, process-scoped RNG/backend control and telemetry."""
import random,subprocess,threading,time
from contextlib import contextmanager
import numpy as np
import torch
from ..util import ContractError
@contextmanager
def seeded(seed):
    py=random.getstate();npstate=np.random.get_state();cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state_all()
    flags=(torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    try:yield
    finally:
        random.setstate(py);np.random.set_state(npstate);torch.set_rng_state(cpu);torch.cuda.set_rng_state_all(gpu)
        torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark=flags

def cuda_probe():
    if not torch.cuda.is_available():raise ContractError("CUDA unavailable; no silent CPU substitution")
    name=torch.cuda.get_device_name(0);cap=torch.cuda.get_device_capability(0)
    if cap!=(6,1) or "1080" not in name:raise ContractError("Expected GTX1080 sm61")
    a=torch.arange(16.,device="cuda").reshape(4,4);b=a@a.T;torch.cuda.synchronize()
    assert torch.equal(b.cpu(),a.cpu()@a.cpu().T)
    return {"torch":torch.__version__,"cuda_runtime":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),"name":name,"compute_capability":list(cap),"tensor_matmul_exact":True,"arch_list":torch.cuda.get_arch_list()}
class GPUResource:
    def __enter__(self):
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();self.samples=[];self.stop=threading.Event()
        def poll():
            while not self.stop.is_set():
                try:self.samples.append(subprocess.check_output(["nvidia-smi","--query-gpu=utilization.gpu,memory.used,memory.total,driver_version","--format=csv,noheader,nounits"],text=True).strip())
                except Exception as e:self.samples.append("unavailable:"+repr(e))
                self.stop.wait(1)
        self.thread=threading.Thread(target=poll,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        torch.cuda.synchronize();self.result={"peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved(),"device_total_bytes":torch.cuda.get_device_properties(0).total_memory}
        self.stop.set();self.thread.join(timeout=5);self.result["nvidia_smi_samples"]=self.samples
