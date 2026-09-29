"""FlopStep: the in-cone fast path must return exactly what flop_linear returns, and a
walk must carry the intersection numbers through flips (not recompute them)."""
import warnings

import numpy as np
import pytest

pytest.importorskip("cytools")
from cytools import Polytope

from fanroots.fanroots import KappaRecomputeWarning
from fanroots.step_taking.flop import FlopStep

# a 4d reflexive polytope with h11 = 101
_PTS = [[-1, -1, -1, -1], [4, -1, -1, -1], [-1, 4, -1, -1],
        [-1, -1, 4, -1], [-1, -1, -1, 4]]


def test_fast_path_matches_flop_linear():
    vc = Polytope(_PTS).vc()
    h0 = np.asarray(vc.subdivide().heights(), dtype=float)
    rng = np.random.default_rng(0)
    n_fast = 0
    for scale in (1e-6, 1e-3, 1e-1, 1.0):
        for _ in range(5):
            tri = vc.subdivide(heights=h0)
            ht = h0 + scale * rng.standard_normal(h0.shape)
            fast = FlopStep._walk(_Opt(), tri, h0, ht, 1)
            ref = tri.flop_linear(h_target=ht, h_init=h0, stop_at_deletion=True,
                                  max_N_flips=1, verbosity=-1, check_regularity=False)
            assert (fast[0] == 1) == (ref[0] == 1)
            np.testing.assert_array_equal(fast[1], ref[1])
            assert fast[2] == ref[2]
            assert fast[4] == ref[4]
            n_fast += fast[2] is tri and fast[4] == 0
    assert n_fast > 0   # the fast path was exercised


class _Opt:
    verbosity = -1
    carry_kappa = False


class _CarryOpt(_Opt):
    carry_kappa = True


def test_flips_carry_kappa_through_the_walk():
    vc = Polytope(_PTS).vc()
    h0 = np.asarray(vc.subdivide().heights(), dtype=float)
    checked = 0
    for seed in range(10):
        d = np.random.default_rng(seed).standard_normal(h0.shape)
        d /= np.linalg.norm(d)
        tri = vc.subdivide(heights=h0)
        tri._fanroots_kappa = tri.intersection_numbers(
            in_basis=True, pushed_down=True, as_np_array=True)
        _, _, tri2, _, nf = FlopStep._walk(_CarryOpt(), tri, h0, h0 + d, 5)
        if nf == 0:
            continue
        # the walk handed back the flop-updated intersection numbers ...
        assert isinstance(tri2.kappa, np.ndarray)
        # ... which agree with a fresh computation to round-off
        fresh = tri2.intersection_numbers(in_basis=True, pushed_down=True,
                                          as_np_array=True)
        np.testing.assert_allclose(tri2.kappa, fresh, rtol=0, atol=1e-10)
        checked += 1
    assert checked > 0   # some walk flipped


@pytest.mark.parametrize("carry", [False, True])
def test_carry_kappa_flag(carry):
    from fanroots.applications.volume_finder import VolumeFinder

    vc = Polytope(_PTS).vc()
    h0 = np.asarray(vc.subdivide().heights(), dtype=float)
    d = np.random.default_rng(3).standard_normal(h0.shape)
    d /= np.linalg.norm(d)
    far = h0 + 2.0 * d
    kappa = vc.triangulate(heights=far).intersection_numbers(
        in_basis=True, pushed_down=True, as_np_array=True)
    t = vc.proj(far)
    opt = VolumeFinder(target=0.5 * (kappa @ t) @ t, vc=vc, heights0=h0.copy(),
                       step_taking_schedule=[[lambda o: True, FlopStep(10)]],
                       carry_kappa=carry, verbosity=-1)
    if carry:
        # the flop-updated arrays are used: no recompute warning
        with warnings.catch_warnings():
            warnings.simplefilter("error", KappaRecomputeWarning)
            opt.optimize()
    else:
        # recomputed from scratch in each new chamber, and the user is told
        with pytest.warns(KappaRecomputeWarning, match="carry_kappa=True"):
            opt.optimize()
    assert opt.num_flips
    tri = opt.triang
    fresh = tri.intersection_numbers(in_basis=True, pushed_down=True, as_np_array=True)
    if carry:
        assert opt.kappa is tri.kappa
        np.testing.assert_allclose(opt.kappa, fresh, rtol=0, atol=1e-10)
    else:
        np.testing.assert_array_equal(opt.kappa, fresh)
