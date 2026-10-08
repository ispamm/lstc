"""Monotonic stages; GPU synchronization hook without a torch dependency."""
from contextlib import contextmanager
from time import perf_counter

class PipelineTimer:
    def __init__(self, synchronize=None):
        self.synchronize = synchronize or (lambda: None)
        self.stages = {}
        self.total = None

    @contextmanager
    def pipeline(self):
        self.synchronize()
        start = perf_counter()
        try:
            yield self
        finally:
            self.synchronize()
            self.total = perf_counter() - start

    @contextmanager
    def stage(self, name):
        if name in self.stages:
            raise ValueError("Duplicate stage")
        self.synchronize()
        start = perf_counter()
        try:
            yield
        finally:
            self.synchronize()
            self.stages[name] = perf_counter() - start
