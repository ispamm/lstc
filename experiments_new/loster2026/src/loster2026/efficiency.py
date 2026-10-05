"""Audit 4 instrumentation resolves legacy timing provenance F22.
Smoke measurements are not scientific benchmarks.
"""
import contextlib
import time
import torch

def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)

@contextlib.contextmanager
def timed(device, result, key):
    synchronize(device)
    start = time.perf_counter()
    try:
        yield
    finally:
        synchronize(device)
        result[key] = time.perf_counter() - start

def reset_gpu_peak(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

def memory_and_parameters(model, device):
    values = {"registered_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "objective_active_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "deployed_original_parameters": sum(p.numel() for p in model.original.parameters()) + model.centers_original.numel(),
              "peak_gpu_allocated_bytes": None, "peak_gpu_reserved_bytes": None}
    if device.type == "cuda":
        values["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        values["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    # Peak working set on Windows, peak RSS on POSIX; no psutil dependency.
    import os
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize",
                "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        counter = Counters()
        counter.cb = ctypes.sizeof(counter)
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        query = ctypes.windll.psapi.GetProcessMemoryInfo
        query.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
        query.restype = wintypes.BOOL
        ok = query(kernel.GetCurrentProcess(), ctypes.byref(counter), counter.cb)
        values["host_peak_bytes"] = int(counter.PeakWorkingSetSize) if ok else None
    else:
        import resource,sys
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        values["host_peak_bytes"] = int(value if sys.platform == "darwin" else value * 1024)
    return values
