"""Smoke tests for the alternate step methods and the batch optimizer, on the
smallest committed geometry (fast). VolumeFinder's default (newton + shrink) is
covered in test_volume_finder.py; here we exercise the other paths."""
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cytools")

# the benchmark scripts are not an importable package; add their dir to the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from data import PROBLEMS  # noqa: E402
from bench_volume_finder import reconstruct  # noqa: E402
from fanroots.applications.volume_finder import VolumeFinder  # noqa: E402
from fanroots.fanroots import BatchOptimizer  # noqa: E402

_PROB = min(PROBLEMS, key=lambda p: p["h11"])          # smallest (h11=56), fastest
_TARGET = np.asarray(_PROB["target"], dtype=float)


def _vc():
    _, vc = reconstruct(_PROB)
    return vc


@pytest.mark.parametrize("step_proposal", ["newton", "gauss_newton", "lma"])
def test_step_proposals_converge(step_proposal):
    vf = VolumeFinder(target=_TARGET, vc=_vc(), step_proposal=step_proposal)
    vf.optimize()
    assert vf.finished_reason == "converged"
    assert float(np.max(np.abs(vf.tau - _TARGET))) < 1e-3


@pytest.mark.parametrize("step_size_optimizer", ["shrink", "bls", "ternary", "naive"])
def test_step_size_optimizers_converge(step_size_optimizer):
    vf = VolumeFinder(target=_TARGET, vc=_vc(), step_size_optimizer=step_size_optimizer)
    vf.optimize()
    assert vf.finished_reason == "converged"
    assert float(np.max(np.abs(vf.tau - _TARGET))) < 1e-3


def test_batch_optimizer_serial():
    batch = BatchOptimizer(optimizers=[
        VolumeFinder(target=_TARGET, vc=_vc()),
        VolumeFinder(target=_TARGET, vc=_vc()),
    ])
    batch.optimize(serial=True)
    assert all(o.finished_reason == "converged" for o in batch.batch)
