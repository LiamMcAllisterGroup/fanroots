fanroots.step\_proposal
=======================

.. currentmodule:: fanroots.step_proposal

.. automodule:: fanroots.step_proposal

Computational graph
-------------------

The step-proposal layer computes a direction p such that the optimiser
moves h to h + alpha * (momentum * lr * p). All four methods derive p
from the residual vector F(h) and its Jacobian J, but differ in how
they regularise or weight the linear system.

For rectangular Jacobians (m rows, n columns, m != n), Newton's method
is not directly applicable because J is not square. The Gauss-Newton
form replaces the exact Newton system with a least-squares problem:

.. math::
   :nowrap:

   \begin{align*}
     \text{Gauss-Newton:}& \quad J p = -F, \\
     \text{LMA:}& \quad \begin{bmatrix} J \\ \sqrt{\lambda}\,D^{1/2} \end{bmatrix} p
                        = -\begin{bmatrix} F \\ 0 \end{bmatrix}, \\
     \text{Gradient descent:}& \quad p = -J^T F.
   \end{align*}

where F(h) is the residual vector and J its Jacobian.

When F and J are complex, the system is split into real and imaginary
parts before solving: [Re(J); Im(J)] p = -[Re(F); Im(F)]. This
doubles the number of rows and keeps p real.

``propose_newton`` is an alias for ``propose_gauss_newton``; the name
reflects the conceptual intent (Newton's method) while the
implementation uses the least-squares form because Jacobians are
generally rectangular.

Levenberg-Marquardt uses an augmented least-squares form rather than
the normal equations, which avoids squaring cond(J). The damping matrix
D is either the identity (unscaled) or diag(J^T J) (scaled). Lambda is
updated dynamically: it is divided by nu after a successful step
(residual decreases) and multiplied by nu after a failed step (residual
increases), and is clamped to [1e-12, 1e12] at all times.

.. raw:: html
   :file: _static/figures/f3_step_proposal.html

Newton / Gauss-Newton
---------------------

.. autosummary::
   :toctree: _autosummary

   propose_newton
   propose_gauss_newton

Levenberg-Marquardt
-------------------

.. autosummary::
   :toctree: _autosummary

   propose_lma

LMA helpers
~~~~~~~~~~~

.. currentmodule:: fanroots.step_proposal.lma

.. autosummary::
   :toctree: _autosummary

   lma
   lma_idk

.. currentmodule:: fanroots.step_proposal

Gradient descent
----------------

.. autosummary::
   :toctree: _autosummary

   propose_gradient_descent
