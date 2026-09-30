"""FlopStep: the in-cone fast path must return exactly what flop_linear returns, and a
walk must carry the intersection numbers through flips (not recompute them)."""
import numpy as np
import pytest

pytest.importorskip("cytools")
from cytools import Polytope

from fanroots.kappa import NonIntegralKappaWarning, integral_kappa
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


def _fresh(tri):
    return integral_kappa(tri.intersection_numbers(in_basis=True, pushed_down=True,
                                                   as_np_array=True))


def test_walk_carries_kappa_exactly_when_the_start_fan_has_it():
    vc = Polytope(_PTS).vc()
    h0 = np.asarray(vc.subdivide().heights(), dtype=float)
    checked = 0
    for seed in range(10):
        d = np.random.default_rng(seed).standard_normal(h0.shape)
        d /= np.linalg.norm(d)
        tri = vc.subdivide(heights=h0)
        tri._fanroots_kappa = _fresh(tri)
        _, _, tri2, _, nf = FlopStep._walk(_Opt(), tri, h0, h0 + d, 5)
        if nf == 0:
            continue
        # the flop-updated intersection numbers equal a fresh computation
        # bit for bit: both are rounded to the integers they represent
        assert isinstance(tri2.kappa, np.ndarray)
        np.testing.assert_array_equal(tri2.kappa, _fresh(tri2))
        checked += 1
    assert checked > 0   # some walk flipped


def test_walk_does_no_kappa_work_without_a_cached_start():
    vc = Polytope(_PTS).vc()
    h0 = np.asarray(vc.subdivide().heights(), dtype=float)
    for seed in range(10):
        d = np.random.default_rng(seed).standard_normal(h0.shape)
        d /= np.linalg.norm(d)
        _, _, tri2, _, nf = FlopStep._walk(_Opt(), vc.subdivide(heights=h0), h0,
                                           h0 + d, 5)
        if nf:
            assert not isinstance(getattr(tri2, "kappa", None), np.ndarray)
            return
    pytest.skip("no walk flipped")


def test_solve_through_flops_uses_integer_kappa():
    from fanroots.applications.volume_finder import VolumeFinder

    vc = Polytope(_PTS).vc()
    h0 = np.asarray(vc.subdivide().heights(), dtype=float)
    d = np.random.default_rng(3).standard_normal(h0.shape)
    d /= np.linalg.norm(d)
    far = h0 + 2.0 * d
    kappa = _fresh(vc.triangulate(heights=far))
    t = vc.proj(far)
    opt = VolumeFinder(target=0.5 * (kappa @ t) @ t, vc=vc, heights0=h0.copy(),
                       step_taking_schedule=[[lambda o: True, FlopStep(10)]],
                       verbosity=-1)
    opt.optimize()
    assert opt.num_flips
    np.testing.assert_array_equal(opt.kappa, _fresh(opt.triang))
    np.testing.assert_array_equal(opt.kappa, np.rint(opt.kappa))


def test_integral_kappa():
    k = np.array([[1.0 + 3e-13, -2.0 - 1e-13], [-4e-14, 7.0]])
    out = integral_kappa(k)
    np.testing.assert_array_equal(out, [[1.0, -2.0], [0.0, 7.0]])
    assert not np.signbit(out[1, 0])          # no -0.0
    with pytest.warns(NonIntegralKappaWarning):
        kept = integral_kappa(np.array([0.5, 1.0]))
    np.testing.assert_array_equal(kept, [0.5, 1.0])
