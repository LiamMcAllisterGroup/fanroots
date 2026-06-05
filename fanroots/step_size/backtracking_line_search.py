# =============================================================================
#    Copyright (C) 2026  Nate MacFadden and contributors
#    Originally developed in the Liam McAllister Group at Cornell University.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description: Apply backtracking line search to optimize step sizes
# -----------------------------------------------------------------------------

import numpy as np

def backtracking_line_search(optimizer, step, tau=0.5, c=0.5, beta=0.8):
    """
    Perform an Armijo backtracking line search to find an acceptable step size.

    Reduces alpha by beta at each iteration until the Armijo sufficient-decrease
    condition r(x + alpha*step) <= r(x) + c*alpha*dot(grad, step) is satisfied.
    Returns 0 if alpha falls below 1e-16.

    Parameters
    ----------
    optimizer : FanRoots
        The FanRoots instance owning the current residual and gradient.
    step : ndarray of shape (n,)
        The proposed step direction.
    tau : float, optional
        Unused legacy parameter, kept for API compatibility. Defaults to 0.5.
    c : float, optional
        Sufficient-decrease constant for the Armijo condition. Defaults to 0.5.
    beta : float, optional
        Multiplicative reduction factor applied to alpha at each iteration.
        Defaults to 0.8.

    Returns
    -------
    alpha : float
        Scale factor in (0, 1] satisfying the Armijo condition,
        or 0 if no such alpha >= 1e-16 was found.
    """
    res0 = optimizer.res_norm()
    grad = optimizer.grad()

    alpha = 1
    while True:
        res = optimizer.res_norm(optimizer.x()+alpha*step)
        gradient_term = c*alpha*np.dot(grad,step)
        if res<=res0+gradient_term:
            return alpha

        alpha *= beta
        if alpha < 1e-16:
            return 0
