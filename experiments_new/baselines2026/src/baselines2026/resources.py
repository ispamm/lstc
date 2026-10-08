"""CPU process-tree RSS including Windows k-Shape pool workers."""
import platform
import threading
import psutil

class ResourceMonitor:
    def __init__(self, interval=0.02):
        self.interval = interval
        self.peak = 0
        self.stop = threading.Event()
        self.thread = None

    def _sample(self):
        proc = psutil.Process()
        procs = [proc]
        try:
            procs += proc.children(recursive=True)
        except psutil.Error:
            pass
        current = 0
        for p in procs:
            try:
                current += p.memory_info().rss
            except psutil.Error:
                pass
        self.peak = max(self.peak, current)

    def _loop(self):
        while not self.stop.wait(self.interval):
            self._sample()

    def __enter__(self):
        self._sample()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self._sample()
        self.stop.set()
        self.thread.join()

    def result(self):
        return {"device": "CPU", "gpu": None, "peak_gpu_bytes": None,
                "host_process_tree_peak_rss_sampled_bytes": self.peak,
                "sampling_interval_seconds": self.interval,
                "limitation": "sampled RSS, includes children; brief peaks may be missed"}

def hardware():
    return {"device": "CPU", "processor": platform.processor(), "platform": platform.platform(),
            "logical_cpus": psutil.cpu_count(), "physical_cpus": psutil.cpu_count(logical=False),
            "host_total_ram_bytes": psutil.virtual_memory().total, "threads": 1}
