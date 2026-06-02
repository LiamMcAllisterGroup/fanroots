# The algorithm

## Problem statement

Given:

- A vector configuration with $N$ elements defining a secondary fan, whose chambers correspond to distinct triangulations of the associated point configuration.
- A smooth function $F: \mathbb{R}^N \to \mathbb{R}^m$ whose analytic form varies chamber by chamber (e.g. depending on intersection numbers $\kappa$ that change discontinuously at chamber walls).
- Optionally, a Jacobian $J: \mathbb{R}^N \to \mathbb{R}^{m \times N}$.

Find: heights $h^*$ such that

$$\|F(h^*)\|_2 < \text{tolerance}.$$

## Two modes of traversal

The core difficulty is that $F$ depends on quantities such as intersection numbers $\kappa$ that change discontinuously whenever $h$ crosses a wall between adjacent chambers of the secondary fan. fanroots handles this via two complementary traversal strategies.

**JumpStep** (large steps): jump directly to $h + \Delta h$, then recompute the triangulation from scratch. Effective when the function varies slowly chamber by chamber and a single step spans many chambers. If the recomputation fails, the step is halved and retried.

**FlopStep** (small steps): walk along the step direction through chamber walls one flip at a time using `regfans.flop_linear`. The triangulation and $\kappa$ update incrementally at each wall crossing. This is efficient near convergence when the proposed step is small and only a few flips are needed.

A `step_taking_schedule` can mix both strategies dynamically, dispatching based on `last_step_size`. For example, `VolumeFinder` uses `JumpStep` when `last_step_size` $\geq 1$ and `FlopStep` (with up to ten flips) otherwise.

## Step proposals

All step proposals solve for a direction $p$ that minimises the linearised residual $\|J p + F\|_2$. The available methods differ in how they handle the condition number of $J$ and the trade-off between speed and stability.

**Gauss-Newton** (default): solve $J\, p = -F$ via QR-pivoted least squares (`scipy gelsy`). Because $J$ is in general rectangular ($m \neq N$), a literal matrix inverse is undefined; the least-squares solve gives the minimum-norm solution. For square, well-conditioned $J$ this reduces to Newton's method. Complex-valued $F$ and $J$ are handled by stacking real and imaginary rows.

**Levenberg-Marquardt**: solve the augmented system

$$\begin{pmatrix} J \\ \sqrt{\lambda}\, D \end{pmatrix} p = \begin{pmatrix} -F \\ 0 \end{pmatrix},$$

where $D$ is a diagonal scaling matrix and $\lambda \geq 0$ is a damping parameter adjusted dynamically: $\lambda$ is divided by $\nu$ after a successful step and multiplied by $\nu$ after a failed one. At $\lambda = 0$ the method reduces to Gauss-Newton; as $\lambda \to \infty$ it approaches gradient descent. This interpolation makes it robust when $J$ is ill-conditioned.

**Gradient descent**: $p = -J^T F$, i.e. steepest descent of $S(h) = \tfrac{1}{2}\|F(h)\|_2^2$. Reliable but slow; useful as a fallback.

## Step size and momentum

After a direction $p$ is computed, a scalar $\alpha \in [0, 1]$ is selected by the step-size optimiser. The actual update applied to $h$ is

$$\Delta h = \alpha \cdot m \cdot \eta \cdot p,$$

where $\eta$ is the learning rate and $m$ is a momentum factor. Four step-size strategies are available:

- **naive**: always $\alpha = 1$.
- **shrink**: halve $\alpha$ repeatedly until $\|F(h + \Delta h)\|_2 \leq \|F(h)\|_2$; return $\alpha = 0$ if $\alpha$ falls below `tol=1e-8`.
- **backtracking line search**: reduce $\alpha$ by factor $\beta$ until the Armijo sufficient-decrease condition is satisfied.
- **ternary search**: minimise $\|F(h + \alpha p)\|_2$ over $\alpha \in [0, 1]$ by ternary search, assuming a unimodal residual profile.

The momentum factor $m$ adjusts adaptively across steps. Let the overestimation ratio be the ratio of the proposed step size to the actual step size taken:

- If the overestimation ratio exceeds 2 (the step had to be cut substantially), $m$ is penalised: $m \leftarrow m \times m_{\text{penalty}}$ with $m_{\text{penalty}} < 1$.
- If the overestimation ratio is below 1.01 (the full proposal was accepted), $m$ is rewarded: $m \leftarrow m \times m_{\text{reward}}$ with $m_{\text{reward}} > 1$.

$m$ is clipped to the interval $[m_{\min}, m_{\max}]$ at each step.

## Halting conditions

`FanRoots` stops iteration when any of the following hold:

- **Convergence**: $\|F(h)\|_2^2 < \text{tolerance}^2$.
- **Step failure**: the last step size fell below `min_step_size`, indicating the optimiser can no longer make progress.
- **Growth demand**: the residual did not decrease by 50% within `growth_demand_timescale` steps.
- **Numerical overflow**: a `RuntimeWarning` is raised during the residual evaluation.
- **User condition**: a user-supplied function `user_halting_fct(optimizer)` returns `True`.
