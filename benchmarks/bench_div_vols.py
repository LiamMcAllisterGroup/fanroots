#!/usr/bin/env python3
"""Micro-benchmark for the divisor-volume contraction kernel.

``VolumeFinder.div_vols`` evaluates ``0.5 * kappa_{ijk} t_j t_k`` on every
function / Jacobian call, so the way that contraction is computed matters. This
compares dense and sparse variants on the real benchmark geometries (h11 =
93-150, the regime FanRoots targets) and reports the per-call median time.

At these h11 the intersection-number tensor is only ~0.1-0.4% dense, so the
sparse kernels beat dense matmul by ~10x and dense einsum by ~100x, and
``np.bincount`` and ``np.add.at`` are on par -- ``div_vols`` uses the
sparse bincount form.

Usage:
    python benchmarks/bench_div_vols.py [--trials N]
"""
import argparse
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import PROBLEMS
from bench_volume_finder import reconstruct, environment


def kappa_nonzeros(prob):
    """Return (kappa, (i, j, k), vals) for a geometry's Delaunay chamber."""
    _, vc = reconstruct(prob)
    heights0 = np.asarray(vc.subdivide().heights(), dtype=float)
    kappa = np.asarray(vc.triangulate(heights=heights0).intersection_numbers(
        in_basis=True, pushed_down=True, as_np_array=True))
    i, j, k = np.nonzero(kappa)
    return kappa, (i, j, k), kappa[i, j, k]


def make_kernels(kappa, nz, vals):
    """The candidate ways to compute 0.5 * kappa_{ijk} t_j t_k, as t -> tau."""
    i, j, k = nz
    n = kappa.shape[0]

    def add_at(t):
        tau = np.zeros(n)
        np.add.at(tau, i, vals * t[j] * t[k])
        return 0.5 * tau

    return {
        "dense-einsum":    lambda t: 0.5 * np.einsum("abc,b,c", kappa, t, t, optimize=True),
        "dense-matmul":    lambda t: 0.5 * (kappa @ t) @ t,
        "sparse-add.at":   add_at,
        "sparse-bincount": lambda t: 0.5 * np.bincount(i, weights=vals * t[j] * t[k], minlength=n),
    }


def time_kernel(fn, ts):
    fn(ts[0])  # warmup
    times = []
    for t in ts:
        tic = time.perf_counter()
        fn(t)
        times.append(time.perf_counter() - tic)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=25,
                    help="timed evaluations per kernel (random t each)")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    names = ["dense-einsum", "dense-matmul", "sparse-add.at", "sparse-bincount"]

    print("Environment:", environment())
    print(f"contraction 0.5 * kappa_ijk t_j t_k | {args.trials} timed evals per kernel\n")
    header = (f"{'geometry':<14}{'h11':>5}{'nnz':>9}{'density':>10}"
              + "".join(f"{name:>16}" for name in names))
    print(header); print("-" * len(header))

    for idx, prob in enumerate(PROBLEMS):
        kappa, nz, vals = kappa_nonzeros(prob)
        n = kappa.shape[0]
        density = vals.size / n ** 3
        ts = [rng.random(n) + 0.5 for _ in range(args.trials)]
        kernels = make_kernels(kappa, nz, vals)

        ref = kernels["dense-matmul"](ts[0])
        cells = []
        for name in names:
            assert np.allclose(kernels[name](ts[0]), ref, atol=1e-8), f"{name} disagrees"
            cells.append(f"{time_kernel(kernels[name], ts) * 1e3:.3f} ms")

        print(f"{f'poly{idx}':<14}{n:>5}{vals.size:>9}{density:>10.4f}"
              + "".join(f"{c:>16}" for c in cells))


if __name__ == "__main__":
    main()
