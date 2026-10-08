from pathlib import Path
import pytest
from baselines2026.timing import PipelineTimer
from baselines2026.provenance import environment
from baselines2026.registry import BASELINE_ROOT, REPOSITORY
from baselines2026.resources import ResourceMonitor
from baselines2026.util import ContractError, external_output

def test_monotonic_and_sync_boundaries():
    calls=[];t=PipelineTimer(lambda:calls.append(1))
    with t.pipeline():
        with t.stage("fit"):sum(range(1000))
    assert len(calls)==4 and t.total>=t.stages["fit"]>=0

def test_environment_lock_is_exact(tmp_path):
    lock=BASELINE_ROOT/"configs/requirements-wave1.lock.txt"
    info,h=environment(lock)
    assert len(h)==64 and info["python"]=="3.9.13"
    bad=tmp_path/"bad.lock.txt";bad.write_text("numpy==0.0.0\n")
    with pytest.raises(ContractError):environment(bad)

def test_output_cannot_enter_git():
    with pytest.raises(ContractError):external_output(REPOSITORY/"generated",REPOSITORY)

def test_resource_monitor_has_device_and_positive_rss():
    with ResourceMonitor() as monitor:sum(range(1000))
    assert monitor.result()["host_process_tree_peak_rss_sampled_bytes"]>0
    assert monitor.result()["device"]=="CPU" and monitor.result()["peak_gpu_bytes"] is None
