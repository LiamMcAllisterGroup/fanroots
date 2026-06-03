#!/usr/bin/env python3
"""Benchmark: FanRoots' VolumeFinder vs. the prior divisor-volume solver.

Runs both solvers on the same realistic Kahler-moduli-stabilization targets
(real dual-Coxeter target volumes for three Calabi-Yau geometries, hardcoded in
``data.py``) and reports wall-clock times and the speedup.

Each problem is rebuilt from cytools only -- no private dependencies and no
network. The polytope is rebuilt from its lattice points, the numeric
GLSM/divisor basis is pinned (cytools' default basis is system-dependent), and
the Delaunay start heights are recomputed from the points. Both solvers start
from those heights and must reach the same target before timing is reported.

Usage:
    python benchmarks/bench_volume_finder.py                # all geometries
    python benchmarks/bench_volume_finder.py 1 2            # selected geometries
    python benchmarks/bench_volume_finder.py --fanroots-only
    python benchmarks/bench_volume_finder.py --trials 5 --prior-max-seconds 3600
"""
import argparse
import os
import platform
import statistics
import sys
import time

import numpy as np
from cytools import Polytope

from fanroots.applications.volume_finder import VolumeFinder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for sibling imports
from prior_method import divisor_to_curve_alt
from data import PROBLEMS


def reconstruct(prob):
    """
    Rebuild the polytope and pin the numeric gale/divisor basis to the stored
    value, since cytools' default basis is system-dependent and the target's
    in_basis labeling must match the basis it was generated in. Returns
    ``(p, vc, basis)``.
    """
    p = Polytope(prob["points"])
    saved_basis = np.asarray(prob["basis"])
    vc = p.vc(include_points_interior_to_facets=False)
    vc._gale_basis = saved_basis      # pin the numeric basis
    vc._gale_in_basis = None          # force the in-basis transform to rebuild from it
    vc._gale = None
    assert np.array_equal(np.asarray(vc.divisor_basis), saved_basis), \
        "failed to pin the gale basis on the vector configuration"
    return p, vc, saved_basis


def divisor_volumes(vc, heights, target_len):
    """
    Primary function.
    """
    kappa = vc.triangulate(heights=heights).intersection_numbers(
        in_basis=True, pushed_down=True, as_np_array=True)
    t = vc.proj(np.asarray(heights, dtype=float))
    return 0.5 * (kappa @ t) @ t


def time_fanroots(vc, target, heights0, trials):
    """
    Warmup once, then `trials` timed runs. Returns (median, min, max, info).
    """
    times = []
    info = None
    for i in range(trials + 1):  # i == 0 is warmup
        tic = time.perf_counter()
        vf = VolumeFinder(target=target, vc=vc, heights0=heights0.copy(),
                          step_size_optimizer="shrink", history_level=0,
                          verbosity=0)
        vf.optimize()
        dt = time.perf_counter() - tic
        if i == 0:
            err = float(np.max(np.abs(
                divisor_volumes(vc, vf.heights, len(target)) - target)))
            info = dict(reason=vf.finished_reason, steps=int(vf.num_steps), err=err)
        else:
            times.append(dt)
    return statistics.median(times), min(times), max(times), info


def time_prior(p, target, max_seconds, basis):
    tic = time.perf_counter()
    kahler, cy, iters = divisor_to_curve_alt(p, target, max_seconds=max_seconds,
                                             divisor_basis=basis)
    dt = time.perf_counter() - tic
    if kahler is None:
        return dt, None, dict(iters=iters)
    err = float(np.max(np.abs(
        np.asarray(cy.compute_divisor_volumes(kahler, in_basis=True)) - target)))
    return dt, err, dict(iters=iters)


def environment():
    import importlib.metadata as md
    mods = {}
    for name in ("numpy", "scipy", "cytools"):
        try:
            mods[name] = getattr(__import__(name), "__version__", None) or md.version(name)
        except Exception:
            mods[name] = "?"
    cores = os.cpu_count()
    return (f"{platform.system()} {platform.machine()} | Python "
            f"{platform.python_version()} | {cores} cores | "
            + ", ".join(f"{k} {v}" for k, v in mods.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("indices", nargs="*", type=int,
                    help="geometry indices to run (default: all in data/)")
    ap.add_argument("--trials", type=int, default=5,
                    help="timed FanRoots runs per geometry (plus one warmup)")
    ap.add_argument("--fanroots-only", action="store_true",
                    help="skip the (slow) prior-method baseline")
    ap.add_argument("--prior-max-seconds", type=float, default=3600.0,
                    help="time budget for each prior-method run")
    args = ap.parse_args()

    indices = args.indices if args.indices else list(range(len(PROBLEMS)))

    print("Environment:", environment())
    print(f"FanRoots: {args.trials} timed runs (+1 warmup) | "
          f"prior method: {'skipped' if args.fanroots_only else '1 run'}\n")
    header = (f"{'geometry':<14}{'h11':>5}{'fanroots(s)':>14}"
              f"{'prior(s)':>12}{'speedup':>10}{'fr_err':>10}{'pr_err':>10}")
    print(header); print("-" * len(header))

    for i in indices:
        prob = PROBLEMS[i]
        name = f"poly{i}"
        h11 = int(prob["h11"])
        target = np.asarray(prob["target"], dtype=float)
        p, vc, basis = reconstruct(prob)
        heights0 = np.asarray(vc.subdivide().heights(), dtype=float)  # Delaunay start

        fr_med, fr_min, fr_max, fr = time_fanroots(vc, target, heights0, args.trials)
        fr_ok = fr["reason"] == "converged" and fr["err"] < 1e-3
        fr_str = f"{fr_med:.2f}" + ("" if fr_ok else "!")

        if args.fanroots_only:
            pr_str, sp_str, pr_err_str = "-", "-", "-"
        else:
            pr_dt, pr_err, pr = time_prior(p, target, args.prior_max_seconds, basis)
            if pr_err is None:
                pr_str, sp_str, pr_err_str = f">{pr_dt:.0f}", "n/a", "FAIL"
            else:
                pr_str = f"{pr_dt:.1f}"
                sp_str = f"{pr_dt / fr_med:.0f}x" if fr_ok else "n/a"
                pr_err_str = f"{pr_err:.1e}"

        print(f"{name:<14}{h11:>5}{fr_str:>14}{pr_str:>12}{sp_str:>10}"
              f"{fr['err']:>10.1e}{pr_err_str:>10}")

    print("\n(! on a fanroots time = did not converge to tol; investigate before trusting.)")


if __name__ == "__main__":
    main()
