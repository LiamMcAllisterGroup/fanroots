#!/usr/bin/env python3
"""Benchmark: FanRoots' VolumeFinder vs. the prior divisor-volume solver.

Runs both solvers on the same KKLT points across all geometries in ``data.py``
(h11 = 56 .. 150) and reports wall-clock time (mean +/- std over repeated runs)
and the speedup. Each problem is rebuilt from cytools only -- no private
dependencies and no network. The polytope is rebuilt from its lattice points, the
numeric GLSM/divisor basis is pinned (cytools' default basis is system-dependent),
and the Delaunay start heights are recomputed. Both solvers start from those
heights and must reach the same target before timing is reported.

Usage:
    python benchmarks/bench_volume_finder.py                     # all geometries
    python benchmarks/bench_volume_finder.py 56 111              # selected h11
    python benchmarks/bench_volume_finder.py --fanroots-only
    python benchmarks/bench_volume_finder.py --trials 5 --prior-trials 3
    python benchmarks/bench_volume_finder.py --plot              # write scaling.png (needs matplotlib)
"""
import argparse
import os
import platform
import statistics
import sys
import time

# Pin BLAS threads BEFORE numpy/scipy import. The prior method is BLAS-bound and,
# left unset, its nested numpy/scipy/cytools thread pools oversubscribe all cores
# -- inflating its time 8-24x and making it unreproducible. Pinning to a fixed
# count gives the baseline a fair, stable time (override via the env vars).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np
from cytools import Polytope

from fanroots.applications.volume_finder import VolumeFinder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling imports
from prior_method import divisor_to_curve_alt
from data import PROBLEMS


def reconstruct(prob):
    """Rebuild the polytope and pin the numeric gale/divisor basis to the stored
    value (cytools' default basis is system-dependent, and the target is only
    meaningful in the basis it was generated in). Returns ``(p, vc)``."""
    p = Polytope(prob["points"])
    saved = np.asarray(prob["basis"])
    vc = p.vc(include_points_interior_to_facets=False)
    vc._gale_basis = saved
    vc._gale_in_basis = None
    vc._gale = None
    assert np.array_equal(np.asarray(vc.divisor_basis), saved), "failed to pin the divisor basis"
    return p, vc


def divisor_volumes(vc, heights):
    kappa = vc.triangulate(heights=heights).intersection_numbers(
        in_basis=True, pushed_down=True, as_np_array=True)
    t = vc.proj(np.asarray(heights, dtype=float))
    return 0.5 * (kappa @ t) @ t


def _mean_std(xs):
    m = statistics.mean(xs)
    s = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return m, s


def time_fanroots(vc, target, heights0, trials):
    """Warmup once, then `trials` timed runs. Returns (mean, std, info)."""
    times, info = [], None
    for i in range(trials + 1):  # i == 0 is warmup
        tic = time.perf_counter()
        vf = VolumeFinder(target=target, vc=vc, heights0=heights0.copy(),
                          step_size_optimizer="shrink", history_level=0, verbosity=0)
        vf.optimize()
        dt = time.perf_counter() - tic
        if i == 0:
            err = float(np.max(np.abs(divisor_volumes(vc, vf.heights) - target)))
            info = dict(reason=vf.finished_reason, err=err)
        else:
            times.append(dt)
    m, s = _mean_std(times)
    return m, s, info


def time_prior(p, target, basis, trials, max_seconds):
    """Up to `trials` timed runs of the prior method. Returns (mean, std, err)
    or (elapsed, None, None) on failure."""
    times, err = [], None
    for _ in range(trials):
        tic = time.perf_counter()
        kahler, cy, _ = divisor_to_curve_alt(p, target, max_seconds=max_seconds, divisor_basis=basis)
        dt = time.perf_counter() - tic
        if kahler is None:
            return dt, None, None
        times.append(dt)
        err = float(np.max(np.abs(np.asarray(cy.compute_divisor_volumes(kahler, in_basis=True)) - target)))
    m, s = _mean_std(times)
    return m, s, err


def environment():
    import importlib.metadata as md
    mods = {}
    for name in ("numpy", "scipy", "cytools"):
        try:
            mods[name] = getattr(__import__(name), "__version__", None) or md.version(name)
        except Exception:
            mods[name] = "?"
    cpu = platform.processor()
    try:  # Linux: read the model name
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip(); break
    except Exception:
        pass
    try:
        ram_gib = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
        ram = f"{ram_gib:.0f} GiB"
    except Exception:
        ram = "?"
    threads = next((f"{v} ({k})" for k, v in os.environ.items()
                    if k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")),
                   "unset (BLAS default)")
    return (f"{cpu} | {os.cpu_count()} cores | {ram} RAM | "
            f"{platform.system()} {platform.release()} | Python {platform.python_version()} | "
            + ", ".join(f"{k} {v}" for k, v in mods.items())
            + f" | BLAS threads: {threads}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h11s", nargs="*", type=int, help="h11 values to run (default: all)")
    ap.add_argument("--trials", type=int, default=5, help="timed FanRoots runs per geometry (+1 warmup)")
    ap.add_argument("--prior-trials", type=int, default=3, help="timed prior-method runs per geometry")
    ap.add_argument("--fanroots-only", action="store_true")
    ap.add_argument("--prior-max-seconds", type=float, default=5000.0)
    ap.add_argument("--plot", action="store_true", help="write scaling.png (needs matplotlib)")
    ap.add_argument("--replot", action="store_true",
                    help="regenerate scaling.png from the saved results JSON, without re-timing")
    args = ap.parse_args()

    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scaling_results.json")
    if args.replot:
        import json
        _plot(json.load(open(results_path)))
        return

    probs = sorted(PROBLEMS, key=lambda p: p["h11"])
    if args.h11s:
        probs = [p for p in probs if p["h11"] in set(args.h11s)]

    print("Environment:", environment(), flush=True)
    print(f"FanRoots: {args.trials} timed runs (+1 warmup) | prior method: "
          f"{'skipped' if args.fanroots_only else f'{args.prior_trials} runs'}\n", flush=True)
    header = (f"{'h11':>5}{'FanRoots(s)':>20}{'prior(s)':>22}{'speedup':>10}{'fr_err':>10}")
    print(header, flush=True); print("-" * len(header), flush=True)

    rows = []
    for prob in probs:
        h11 = prob["h11"]
        target = np.asarray(prob["target"], dtype=float)
        p, vc = reconstruct(prob)
        heights0 = np.asarray(vc.subdivide().heights(), dtype=float)

        fr_m, fr_s, fr = time_fanroots(vc, target, heights0, args.trials)
        fr_ok = fr["reason"] == "converged" and fr["err"] < 1e-3
        fr_str = f"{fr_m:.3f}+/-{fr_s:.3f}" + ("" if fr_ok else "!")

        if args.fanroots_only:
            pr_m = pr_s = None; pr_str, sp_str = "-", "-"
        else:
            pr_m, pr_s, pr_err = time_prior(p, target, np.asarray(prob["basis"]),
                                            args.prior_trials, args.prior_max_seconds)
            if pr_err is None:
                pr_str, sp_str = f">{pr_m:.0f} FAIL", "n/a"; pr_m = None
            else:
                pr_str = f"{pr_m:.1f}+/-{pr_s:.1f}"
                sp_str = f"{pr_m / fr_m:.0f}x" if fr_ok else "n/a"
        rows.append(dict(h11=h11, fr=fr_m, fr_std=fr_s, pr=pr_m, pr_std=pr_s))
        print(f"{h11:>5}{fr_str:>20}{pr_str:>22}{sp_str:>10}{fr['err']:>10.1e}", flush=True)

    print("\n(! on a FanRoots time = did not converge to tol; investigate.)")

    if not args.fanroots_only:
        import json
        json.dump(rows, open(results_path, "w"), indent=2)  # enables --replot
    if args.plot and not args.fanroots_only:
        _plot(rows)
    elif args.plot:
        print("[skipping --plot: --fanroots-only has no prior data to plot]")


def _plot(rows):
    """Scatter of solve time vs h11 (markers only -- one geometry per point).
    Geometries where the prior method failed to converge are ringed. Regenerating
    scaling.png needs `pip install matplotlib`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda r: r["h11"])
    fr = [(r["h11"], r["fr"], r.get("fr_std", 0) or 0) for r in rows if r.get("fr")]
    pr = [(r["h11"], r["pr"], r.get("pr_std", 0) or 0) for r in rows if r.get("pr")]
    fail = [(r["h11"], r["fr"]) for r in rows if r.get("fr") and not r.get("pr")]

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    # error bars are +/- std over repeated runs (usually smaller than the markers --
    # the timings are very reproducible once BLAS threads are pinned)
    ax.errorbar([h for h, _, _ in pr], [t for _, t, _ in pr], yerr=[e for _, _, e in pr],
                fmt="D", ms=6, color="#e4572e", capsize=3, label="prior method")
    ax.errorbar([h for h, _, _ in fr], [t for _, t, _ in fr], yerr=[e for _, _, e in fr],
                fmt="o", ms=6, color="#2a6df4", capsize=3, label="FanRoots")
    for h, f in fail:  # ring geometries where the prior method failed to converge
        ax.scatter([h], [f], s=130, marker="o", facecolors="none",
                   edgecolors="#111", linewidths=1.6, zorder=5)
    if fail:
        ax.scatter([], [], s=60, marker="o", facecolors="none", edgecolors="#111",
                   linewidths=1.6, label="(prior method failed)")
    ax.set_yscale("log")
    ax.set_xlabel("optimization dimension", fontsize=12)
    ax.set_ylabel("solve time [s]", fontsize=12)
    ax.set_title("VolumeFinder benchmarking (KKLT points)", fontsize=13)
    ax.grid(True, which="both", alpha=0.15)
    ax.legend(fontsize=10, loc="upper left")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scaling.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
