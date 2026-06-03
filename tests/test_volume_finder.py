import pytest
import numpy as np

pytest.importorskip("cytools")
from cytools import Polytope
from fanroots.applications.volume_finder import VolumeFinder
from fanroots.step_taking.flop import FlopStep

# A 4d reflexive polytope with h11 = 101
_PTS = [[-1, -1, -1, -1], [4, -1, -1, -1], [-1, 4, -1, -1],
        [-1, -1, 4, -1], [-1, -1, -1, 4]]


def _far_target(vc, seed=3):
    # Divisor volumes at the farthest still-fine point along a fixed direction
    # from the Delaunay start
    h0        = VolumeFinder(target=np.ones(1), vc=vc, history_level=0, verbosity=0).heights
    rng       = np.random.default_rng(seed)
    direction = rng.standard_normal(h0.shape)
    direction /= np.linalg.norm(direction)

    far  = None
    step = 0.0
    while step <= 40:
        step += 0.5
        cand  = h0 + step * direction
        tri   = vc.triangulate(heights=cand)
        if not tri.is_fine():
            break
        far = (cand, tri)
    assert far is not None

    cand, tri = far
    kappa     = tri.intersection_numbers(in_basis=True, pushed_down=True, as_np_array=True)
    t         = vc.proj(cand)
    return 0.5 * (kappa @ t) @ t


# VolumeFinder: cross-chamber convergence
# ---------------------------------------
def test_converges_across_many_chambers():
    vc     = Polytope(_PTS).vc()
    target = _far_target(vc)

    # force single-flip steps
    flop_only = [[lambda opt: True, FlopStep(max_num_flips=1)]]
    vf = VolumeFinder(target=target, vc=vc, step_taking_schedule=flop_only,
                      history_level=2, verbosity=0)
    vf.optimize()
    assert vf.finished_reason == "converged"

    # get the number of flips
    n_flips = sum(int(a.get("num_flips", 0)) for a in vf.history_anc)
    assert n_flips >= 5

    # Verify the solution independently from a freshly recomputed kappa
    tri   = vc.triangulate(heights=vf.heights)
    kappa = tri.intersection_numbers(in_basis=True, pushed_down=True, as_np_array=True)
    t     = vc.proj(vf.heights)
    np.testing.assert_allclose(0.5 * (kappa @ t) @ t, target, atol=1e-3)
