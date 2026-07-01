#!/usr/bin/env python3
"""Scaling benchmark: FanRoots' VolumeFinder vs. the prior divisor-volume solver
across a range of dim = h^{1,1} of the search geometry.

Complements bench_volume_finder.py (which uses three fixed geometries) by sweeping
nine KKLT geometries from dim 56 to 111 (see scaling_data.py), reporting each
solver's wall-clock time and the speedup. Both solvers are given the same target
and the same (pinned) divisor basis, and must reach it before timing is reported.

Like bench_volume_finder.py, FanRoots is timed over --trials runs (plus one
warmup); the prior method is run once (it is 1-2 orders of magnitude slower, so a
single run is enough -- but note that means no error bars on the baseline).

Usage:
    python benchmarks/bench_scaling.py                 # all geometries (slow: prior method)
    python benchmarks/bench_scaling.py --fanroots-only # skip the slow baseline
    python benchmarks/bench_scaling.py --trials 5 --prior-max-seconds 600
    python benchmarks/bench_scaling.py --plot          # also write scaling.png (needs matplotlib)
"""
import argparse
import os
import statistics
import sys
import time

import numpy as np
from cytools import Polytope

from fanroots.applications.volume_finder import VolumeFinder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling imports
from prior_method import divisor_to_curve_alt
from scaling_data import PROBLEMS


def reconstruct(prob):
    """Rebuild the geometry and pin the recorded divisor basis (cytools' default
    is system-dependent, and the target is only meaningful in its own basis)."""
    p = Polytope(prob["points"])
    vc = p.vc(include_points_interior_to_facets=False)
    vc._gale_basis = np.asarray(prob["basis"])
    assert np.array_equal(np.asarray(vc.divisor_basis), np.asarray(prob["basis"])), \
        "failed to pin the divisor basis"
    return p, vc


def time_fanroots(vc, target, trials):
    times = []
    for i in range(trials + 1):  # i == 0 is warmup
        tic = time.perf_counter()
        vf = VolumeFinder(target=target, vc=vc)
        vf.optimize()
        if i:
            times.append(time.perf_counter() - tic)
    err = float(np.max(np.abs(np.asarray(vf.tau) - target)))
    return statistics.median(times), vf.finished_reason, err


def time_prior(p, target, basis, max_seconds):
    tic = time.perf_counter()
    kahler, cy, _ = divisor_to_curve_alt(p, target, max_seconds=max_seconds,
                                         divisor_basis=basis)
    dt = time.perf_counter() - tic
    if kahler is None:
        return dt, None
    return dt, dt  # (elapsed, elapsed) -- None signals failure above


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--fanroots-only", action="store_true")
    ap.add_argument("--prior-max-seconds", type=float, default=900.0)
    ap.add_argument("--plot", action="store_true", help="write scaling.png (needs matplotlib)")
    args = ap.parse_args()

    probs = sorted(PROBLEMS, key=lambda p: p["dim"])
    header = f"{'dim':>5}{'FanRoots(s)':>13}{'prior(s)':>12}{'speedup':>10}{'fr_err':>10}"
    print(header); print("-" * len(header))

    rows = []
    for prob in probs:
        target = np.asarray(prob["target"], dtype=float)
        p, vc = reconstruct(prob)
        fr, reason, err = time_fanroots(vc, target, args.trials)
        fr_str = f"{fr:.3f}" + ("" if reason == "converged" and err < 1e-3 else "!")

        if args.fanroots_only:
            pr_str, sp_str, pr = "-", "-", None
        else:
            pr, ok = time_prior(p, target, np.asarray(prob["basis"]), args.prior_max_seconds)
            if ok is None:
                pr_str, sp_str, pr = f">{pr:.0f}", "n/a", None
            else:
                pr_str, sp_str = f"{pr:.1f}", f"{pr / fr:.0f}x"
        rows.append((prob["dim"], fr, pr))
        print(f"{prob['dim']:>5}{fr_str:>13}{pr_str:>12}{sp_str:>10}{err:>10.1e}")

    print("\n(! on a FanRoots time = did not converge to tol; investigate.)")

    if args.plot:
        if args.fanroots_only:
            print("\n[skipping --plot: --fanroots-only has no prior-method data to plot; "
                  "would overwrite scaling.png with a FanRoots-only figure]")
        else:
            _plot(rows)


def _plot(rows):
    """Scatter of solve time vs dim (markers only -- one geometry per point).
    Regenerating the committed scaling.png needs `pip install matplotlib`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fr = [(d, t) for d, t, _ in rows]
    pr = [(d, pt) for d, _, pt in rows if pt]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    if pr:
        ax.scatter([d for d, _ in pr], [t for _, t in pr], s=90, marker="D",
                   color="#e4572e", label="prior method")
    ax.scatter([d for d, _ in fr], [t for _, t in fr], s=90, marker="o",
               color="#2a6df4", label="FanRoots")
    ax.set_yscale("log")
    ax.set_xlabel(r"dim  ($h^{1,1}$ of the search geometry)", fontsize=12)
    ax.set_ylabel("solve time (s)", fontsize=12)
    ax.set_title("Divisor-volume solve time vs dimension (KKLT targets)", fontsize=13)
    ax.grid(True, which="both", alpha=0.15)
    ax.legend(fontsize=11, loc="upper left")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scaling.png")
    fig.savefig(out, dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main()
