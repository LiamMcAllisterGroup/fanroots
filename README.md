# fanroots
*Original author: [Nate MacFadden](https://github.com/natemacfadden), Liam McAllister Group, Cornell*

*Contributors: —*

Root-finding and optimization for vector-valued functions defined piecewise over the secondary fan of a point or vector configuration. Designed for Kähler moduli stabilization (KMS) in string compactifications, where it delivers **order-of-magnitude speedups** over prior methods ([arXiv:2406.13751](https://arxiv.org/abs/2406.13751)).

## The Problem

Given a vector/point configuration with $N$ elements, the secondary fan partitions $\mathbb{R}^N$ (or, for vector configurations, a convex subregion of this space) into convex 'secondary' cones. Call $\mathbb{R}^N$ 'height space'. This software is designed for continuous, differentiable functions whose analytic form may vary chamber-by-chamber. One example is KMS, which depends on the intersection numbers of the toric variety associated to each triangulation. These numbers change discretely as one crosses walls of the secondary fan, but the function remains smooth.

The major complication in practice is the moderate-to-high dimension $\mathbb{R}^N$ as well as the large number of chambers (depending roughly exponentially on N - see [arXiv:2008.01730](https://arxiv.org/abs/2008.01730), [arXiv:2309.10855](https://arxiv.org/abs/2309.10855), and [arXiv:2602.16909](https://arxiv.org/abs/2602.16909)). At $N$ of interest, there are far too many chambers to enumerate, so operations are instead taken locally.

## Algorithm

**fanroots** solves this by moving through the fan adaptively:

- **Large steps** (`JumpStep`): jump directly to target heights, recomputing the triangulation from scratch. Effective when the function varies slowly chamber-by-chamber, as is typical for finding locations in Kähler moduli space where the divisors take certain volumes.
- **Small steps** (`FlopStep`): walk along the step direction via `regfans`' `flip_linear`, flipping through chamber walls one at a time. Efficient for fine-grained convergence once near a solution.

A schedule can mix both strategies dynamically based on step size. Step proposals include Newton's method, Gauss-Newton, gradient descent, and Levenberg-Marquardt. Step sizes are tuned via backtracking line search, ternary search, or 'shrinking'.

For functions depending on the intersection numbers, performance is further boosted by a recently developed fast intersection number kernel in [CYTools](https://github.com/LiamMcAllisterGroup/cytools), since computation of the intersection numbers typically becomes the bottleneck for such cases.

## Installation

```
pip install -e .
```

Requires [regfans](https://github.com/natemacfadden/regfans). For string-theoretic applications, [CYTools](https://github.com/LiamMcAllisterGroup/cytools) is also required.

## Usage

This operates via an optimization class `FanRoots`:

```python
from fanroots import FanRoots

optimizer = FanRoots(
    vc=my_vector_configuration,
    fct=fct,
    jac=jac,
    step_proposal="newton",       # "newton", "gauss_newton", "grad", "lma"
    step_size_optimizer="shrink", # "shrink", "backtracking", "ternary", "naive"
    step_taking_method="jump",    # "jump", "flop"
    tolerance=1e-6,
)
optimizer.run()
```

Key arguments (see `help(FanRoots)` for the full list):

| Argument | Options | Description |
|---|---|---|
| `step_proposal` | `"newton"`, `"gauss_newton"`, `"grad"`, `"lma"` | Step direction method |
| `step_size_optimizer` | `"shrink"`, `"backtracking"`, `"ternary"`, `"naive"` | Step size tuning |
| `step_taking_method` | `"jump"`, `"flop"` | How to move through the fan; overridden by `step_taking_schedule` for mixed strategies |
| `learning_rate` | float | Scales the proposed step before size optimization |
| `tolerance` | float | Halt when `\|fct(h)\|_2 < tolerance` |
| `min_step_size` | float | Halt if step shrinks below this |
| `verbosity` | int | Controls diagnostic output |

See `demo/volume_finder.py` for a complete example finding Kähler parameters that realize prescribed divisor volumes.
