"""The top-level FanRoots API with a user-supplied fct/jac -- the path the
README documents. VolumeFinder (tested separately) wraps this; here we exercise
FanRoots directly with a hand-written function and Jacobian."""
import pytest
import numpy as np

pytest.importorskip("cytools")
from cytools import Polytope
from fanroots import FanRoots

# a 4d reflexive polytope with h11 = 101
_PTS = [[-1, -1, -1, -1], [4, -1, -1, -1], [-1, 4, -1, -1],
        [-1, -1, 4, -1], [-1, -1, -1, 4]]


def _reachable_target(vc, glsm, h0, seed=0):
    """Divisor volumes at the farthest still-fine point along a fixed direction
    from the Delaunay start -- a solvable target that requires chamber crossings."""
    rng = np.random.default_rng(seed)
    d = rng.standard_normal(h0.shape)
    d /= np.linalg.norm(d)
    far, step = h0, 0.0
    while step < 40:
        step += 0.5
        cand = h0 + step * d
        if not vc.triangulate(heights=cand).is_fine():
            break
        far = cand
    kappa = vc.triangulate(heights=far).intersection_numbers(
        in_basis=True, pushed_down=True, as_np_array=True)
    t = glsm @ far
    return 0.5 * (kappa @ t) @ t


def test_fanroots_with_user_fct_jac():
    vc = Polytope(_PTS).vc()
    glsm = vc.gale().T
    h0 = np.asarray(vc.subdivide().heights(), dtype=float)
    target = _reachable_target(vc, glsm, h0)

    # fct/jac receive the optimizer as their first argument; opt.glsm and
    # opt.kappa expose the current chamber's data
    def fct(opt, h):
        t = opt.glsm @ h
        return 0.5 * (opt.kappa @ t) @ t - target

    def jac(opt, h):
        t = opt.glsm @ h
        return (opt.kappa @ t) @ opt.glsm

    opt = FanRoots(vc=vc, fct=fct, jac=jac, heights0=h0,
                   step_proposal="newton", step_size_optimizer="shrink",
                   tolerance=1e-4, verbosity=0)
    opt.optimize()

    assert opt.finished_reason == "converged"
    # independently verify the residual at the returned heights
    assert float(np.linalg.norm(fct(opt, opt.heights))) < 1e-3
