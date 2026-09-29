"""Every way a solve ends must say why: ``termination`` is always one of
``FanRoots.TERMINATIONS`` and ``finished_reason`` is never left at "N/A". Also pins
the bookkeeping that diagnoses a solve after the fact: ``history`` aligned with
``history_res_norm``, and the best point seen."""
import numpy as np
import pytest

pytest.importorskip("cytools")
from cytools import Polytope

from fanroots import FanRoots
from fanroots.applications.volume_finder import VolumeFinder
from fanroots.step_taking.flop import FlopStep

# a 4d reflexive polytope with h11 = 101
_PTS = [[-1, -1, -1, -1], [4, -1, -1, -1], [-1, 4, -1, -1],
        [-1, -1, 4, -1], [-1, -1, -1, 4]]


@pytest.fixture(scope="module")
def vc():
    return Polytope(_PTS).vc()


def _target(vc, seed=3):
    """Divisor volumes at the farthest still-fine point along a random direction."""
    h0 = np.asarray(vc.subdivide().heights(), dtype=float)
    d = np.random.default_rng(seed).standard_normal(h0.shape)
    d /= np.linalg.norm(d)
    far, step = h0, 0.0
    while step < 40:
        step += 0.5
        if not vc.triangulate(heights=h0 + step * d).is_fine():
            break
        far = h0 + step * d
    kappa = vc.triangulate(heights=far).intersection_numbers(
        in_basis=True, pushed_down=True, as_np_array=True)
    t = vc.proj(far)
    return 0.5 * (kappa @ t) @ t


def _finder(vc, target, **kw):
    kw.setdefault("step_taking_schedule", [[lambda o: True, FlopStep(max_num_flips=10)]])
    return VolumeFinder(target=target, vc=vc, verbosity=-1, **kw)


def _assert_explained(opt):
    assert opt.finished
    assert opt.termination in FanRoots.TERMINATIONS
    assert opt.finished_reason not in (None, "N/A")
    assert opt.termination_info["num_steps"] == opt.num_steps


def test_converged(vc):
    opt = _finder(vc, _target(vc))
    opt.optimize()
    _assert_explained(opt)
    assert opt.termination == "converged" and opt.success is True
    assert opt.termination_info["res_norm_final"] < opt.tolerance


def test_user_halt_gives_a_reason(vc):
    opt = _finder(vc, _target(vc), user_halting_fct=lambda o: o.num_steps >= 3)
    opt.optimize()
    _assert_explained(opt)
    assert opt.termination == "user_halt"
    assert opt.num_steps == 3


def test_no_progress(vc):
    # an unreachable target (volumes far outside the fan's image) cannot halve
    # the residual within 5 steps
    opt = _finder(vc, -1e3 * np.ones_like(_target(vc)), growth_demand_timescale=5)
    opt.optimize()
    _assert_explained(opt)
    assert opt.termination in ("no_progress", "stalled_at_wall", "stalled")
    assert opt.success is False


def test_numerical_error_names_the_warning(vc):
    from fanroots.fanroots import ResNormError

    def fct(o, h):
        return np.full(o.h11, 1e200)          # its square overflows in res_norm

    def jac(o, h):
        return np.eye(o.h11, len(h))

    opt = FanRoots(vc=vc, fct=fct, jac=jac, verbosity=-1)
    with pytest.raises(ResNormError):
        opt.res_norm()
    assert opt.termination == "numerical_error" and opt.success is False
    assert "overflow" in opt.finished_reason


def test_reckless_mode_continues(vc):
    calls = {"n": 0}

    def fct(o, h):
        calls["n"] += 1
        return np.full(o.h11, np.inf) if calls["n"] == 1 else np.zeros(o.h11)

    def jac(o, h):
        return np.eye(o.h11, len(h))

    opt = FanRoots(vc=vc, fct=fct, jac=jac, reckless_mode=True, verbosity=-1)
    opt.res_norm()                     # the infinite residual is tolerated ...
    assert not opt.finished            # ... and does not end the solve


def test_exception_is_recorded_then_reraised(vc):
    def fct(o, h):
        return np.ones(o.h11)

    def jac(o, h):
        raise ZeroDivisionError("boom")

    opt = FanRoots(vc=vc, fct=fct, jac=jac, verbosity=-1)
    with pytest.raises(ZeroDivisionError):
        opt.step()
    _assert_explained(opt)
    assert opt.termination == "exception"
    assert opt.termination_info["exc_type"] == "ZeroDivisionError"


def test_history_is_aligned_and_best_point_tracked(vc):
    opt = _finder(vc, _target(vc), history_level=1)
    opt.optimize()
    atol = 1e-12 * opt.history_res_norm[0]
    assert len(opt.history) == len(opt.history_res_norm) == opt.num_steps
    # history[k] is the point whose residual is history_res_norm[k]
    for k in (0, len(opt.history) // 2, len(opt.history) - 1):
        # tolerance: the solve carries kappa through flops incrementally,
        # which agrees with a fresh computation to ~1e-14; near convergence
        # the squared residual amplifies that (relatively). Neighbouring
        # history points differ by far more, so this still pins the alignment
        res_k = opt.res_norm(opt.history[k], use_actual_kappa=True)
        assert np.isclose(res_k, opt.history_res_norm[k], rtol=1e-6, atol=atol)
    assert opt.best_res_norm == pytest.approx(min(opt.history_res_norm))
    assert np.isclose(opt.res_norm(opt.best_x, use_actual_kappa=True),
                      opt.best_res_norm, rtol=1e-6, atol=atol)
    # evaluating elsewhere must leave the solver's own state untouched
    assert np.isclose(opt.res_norm(), opt.history_res_norm[-1], rtol=1e-12)


def test_kappa_sparsity_cache_survives_steps_in_one_chamber(vc):
    opt = _finder(vc, _target(vc))
    nz = opt.kappa_nz()
    opt.set_kappa(opt.kappa)           # what _step does after a step without a flip
    assert opt.kappa_nz() is nz
    opt.set_kappa(opt.kappa.copy())    # a new kappa invalidates it
    assert opt.kappa_nz() is not nz


def test_interrupt_is_recorded_then_reraised(vc):
    class Alarm(BaseException):       # like a caller's timeout interrupt
        pass

    def fct(o, h):
        return np.ones(o.h11)

    def jac(o, h):
        raise Alarm()

    opt = FanRoots(vc=vc, fct=fct, jac=jac, verbosity=-1)
    with pytest.raises(Alarm):
        opt.step()
    _assert_explained(opt)
    assert opt.termination_info["exc_type"] == "Alarm"
