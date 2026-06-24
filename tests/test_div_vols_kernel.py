"""The production div_vols kernel (sparse bincount) must agree with the dense
references it is benchmarked against. The benchmark only times these kernels;
this test pins their *correctness* on the same geometries, so the perf-critical
sparse contraction can't silently diverge from the dense ground truth."""
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cytools")

# the benchmark scripts are not an importable package; add their dir to the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from data import PROBLEMS  # noqa: E402
from bench_div_vols import kappa_nonzeros, make_kernels  # noqa: E402


@pytest.mark.parametrize("prob", PROBLEMS)
def test_div_vols_kernels_agree(prob):
    kappa, nz, vals = kappa_nonzeros(prob)
    kernels = make_kernels(kappa, nz, vals)
    rng = np.random.default_rng(0)
    t = rng.standard_normal(kappa.shape[0])
    ref = kernels["dense-matmul"](t)
    for name, fn in kernels.items():
        np.testing.assert_allclose(
            fn(t), ref, rtol=1e-9, atol=1e-8,
            err_msg=f"div_vols kernel '{name}' disagrees with dense-matmul",
        )
