# =============================================================================
#    Copyright (C) 2026  Liam McAllister Group
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
# Description: Propose an optimization step h->h+step in a fan using gradient
#              descent.
# -----------------------------------------------------------------------------

import numpy as np

def propose_gradient_descent(optimizer):
    """
    **Description:**
    We solve F(h, x) = 0 using gradient descent, where
        - h are the heights (point in the fan) and
        - x are some optional other parameters.

    (
    In case F is complex, we split the real/imaginary components, effectively
    solving
        F'(h, x) = [Re(F(h,x)); Im(F(h,x))] = 0.
    This requires modifying
        J'(h, x) = [Re(J(h,x)); Im(J(h,x))]
    )

    This really solves the least squares problem
        argmin S = argmin \\sum_i F_i(x)^2.
    It does so via stepping
        step = - lr \\grad S = - lr jac.T @ F(x)

    **Arguments:**
    - `optimizer`: The FanRoots containing the current state.

    **Returns:**
    - `step_t`: The step in K\"ahler parameters.
    - `step_x`: If other_params is not None, then the step in other_params.
    """
    # compute the step
    step = -optimizer.grad()
    return step#, None

    #if not optimizer.only_heights:
    #    h11 = len(optimizer.heights)
    #    step_h     = step[:h11]
    #    step_other = step[h11:]
    #    return step_h, step_other
    #else:
    #    step_h     = step
    #    return step_h