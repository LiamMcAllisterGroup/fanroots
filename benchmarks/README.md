# Benchmarks

Performance comparison of FanRoots' `VolumeFinder` against the prior
divisor-volume solver, on realistic targets ('starting guesses').

## What is measured

For each Calabi-Yau geometry in `data.py` (optimization dimension $h^{1,1}$ = 56 to 150), both solvers are given the **same** KKLT point to solve for -- target divisor volumes from [arXiv:2406.13751](https://arxiv.org/pdf/2406.13751) -- and must find Kahler parameters / heights realizing them. We report wall-clock time (mean +/- std over repeated runs) and the speedup.

- **FanRoots** -- the `VolumeFinder` in this repo (`step_size_optimizer="shrink"`, otherwise default: a jump+flop schedule), started from the Delaunay triangulation.
- **Prior method** -- `prior_method.divisor_to_curve_alt`, the solver used in [arXiv:2406.13751](https://arxiv.org/pdf/2406.13751). Its numerical logic is verbatim (5th-order perturbative Newton with a tight-tolerance `least_squares` linear solve, adaptive step control, and flop detection).

**BLAS threads are pinned** -- the harness sets `OMP/OPENBLAS/MKL_NUM_THREADS` before importing numpy. The prior method is BLAS-bound, and left unset its nested numpy/scipy/cytools thread pools oversubscribe all cores, inflating its time ~8-24x and making it irreproducible. Pinning gives the baseline a fair, stable time.

## Results

![VolumeFinder benchmarking (KKLT points)](scaling.png)

Measured on an Intel Core Ultra 7 270K (24 cores, BLAS threads=8), Python 3.14, numpy 2.4 / scipy 1.18 / cytools 1.4.11:

| h11 | FanRoots (s) | prior (s) | speedup |
|----:|:-------------|:----------|--------:|
| 56  | 0.20 +/- 0.00 | 4.5 +/- 0.1   | 22x |
| 70  | 0.27 +/- 0.00 | 10.8 +/- 0.0  | 39x |
| 90  | 0.50 +/- 0.00 | 17.2 +/- 0.0  | 34x |
| 93  | 0.57 +/- 0.00 | 21.9 +/- 0.0  | 39x |
| 111 | 0.88 +/- 0.03 | 35.6 +/- 0.1  | 41x |
| 150 | 1.57 +/- 0.03 | 105.4 +/- 0.1 | 67x |

(Representative rows; the full 12-geometry sweep is in `data.py`.) FanRoots is **~20-70x faster, and the advantage grows with dimension** -- it scales gently (0.2 -> 1.6 s over $h^{1,1}$ 56 -> 150) while the prior method grows from seconds to ~2 minutes. On some geometries (e.g. $h^{1,1}=86$, ringed in the plot) the prior method **fails to converge** while FanRoots succeeds -- a robustness gap on top of the speed. All runs reach matched accuracy (max |volume - target| ~ 1e-6 - 1e-5). Times are means over repeated runs (FanRoots 5, prior 3); the error bars are +/- std but are **smaller than the plot markers** -- once BLAS threads are pinned the timings are reproducible to ~0.1-3%. At h11=150 the prior method is also memory-intensive (it can exhaust a ~16 GB machine); use `--fanroots-only` there if memory-constrained.

## Running

```bash
python benchmarks/bench_volume_finder.py                     # all geometries
python benchmarks/bench_volume_finder.py 56 150              # selected h11
python benchmarks/bench_volume_finder.py --fanroots-only     # skip the slow baseline
python benchmarks/bench_volume_finder.py --trials 5 --prior-trials 3
python benchmarks/bench_volume_finder.py --plot              # also write scaling.png (needs matplotlib)
python benchmarks/bench_volume_finder.py --replot            # regenerate scaling.png from saved results
```

FanRoots is timed over `--trials` runs (+1 warmup), the prior method over `--prior-trials`; results are saved to `scaling_results.json` (used by `--replot`). With BLAS threads pinned the prior method on the h11=150 geometry takes ~2 min (it was ~1 hour with threads oversubscribed); use `--fanroots-only` for a quick check.

## Kernel micro-benchmark

`bench_div_vols.py` times the inner contraction `VolumeFinder.div_vols` runs on every function/Jacobian call -- `0.5 * kappa_{ijk} t_j t_k` -- comparing dense (`einsum`, `matmul`) against sparse (`np.add.at`, `np.bincount`) forms on the same geometries.

```bash
python benchmarks/bench_div_vols.py [--trials N]
```

At the targeted h11 (90+) the intersection-number tensor is only ~0.1-0.4 % dense, so the sparse kernels win by ~10x over dense matmul and ~100x over einsum, and `np.bincount` and `np.add.at` are on par. `div_vols` uses the sparse `bincount` form (no density-adaptive branch -- small-h11, where dense would win, is out of scope).

## Data and reproducibility

`data.py` holds the pre-extracted problems as plain literals: the cut-polytope lattice **points**, the numeric GLSM/**divisor basis**, and the **target** volumes. The Delaunay start heights are recomputed from the points, so they are not stored. The harness rebuilds each geometry from the points using **cytools only** -- no private dependencies and no network.

**Divisor basis.** All `in_basis` quantities (target, intersection numbers) depend on the gale/divisor basis, and cytools' default basis is *system-dependent*. So the saved numeric basis is pinned explicitly -- on the vector configuration for FanRoots and via `set_divisor_basis` on the prior method's CalabiYau -- rather than trusting the default to reproduce it. This is essential: the target is only meaningful in the basis it was generated in.

The targets themselves are generated by the (private) KMS pipeline
(`kklt_lib`: cut polytope -> conifold -> dual-Coxeter target volumes); only the
resulting numeric problems are stored here, so the benchmark is self-contained.
