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
# Description: Propose an optimization step h->h+step in a fan using Newton's
#              method.
# -----------------------------------------------------------------------------

import numpy as np
import scipy as sp

def propose_newton(optimizer):
    """
    Propose a step h->h+step using Newton's method for root finding.

    Solve F(h, x) = 0 using Newton's method, where h are the heights
    (point in the fan) and x are some optional other parameters.

    In case F is complex, we split the real/imaginary components,
    effectively solving
        F'(h, x) = [Re(F(h,x)); Im(F(h,x))] = 0.
    This requires modifying
        J'(h, x) = [Re(J(h,x)); Im(J(h,x))]

    This can be derived as:
        F(h+step_h, x+step_x) = F(h, x) + J(h, x)@[step_h, step_x] + ...
    Then, to linear order, F has a root at h+step_h, x+step_x if
        J(h, x)@[step_h, step_x] = -F(h, x).
    This is simple to solve for via least squares:
        step = lstsq(J, -F).

    Parameters
    ----------
    optimizer : FanRoots
        The FanRoots instance containing the current state.

    Returns
    -------
    step : ndarray of shape (n,)
        The proposed step, where n = len(heights) + len(other).
        Contains the step in heights (and optionally other parameters,
        concatenated).
    """
    # fetch the value of the function of interest F (and its Jacobian, J)
    F_h = optimizer.fct()
    J_h = optimizer.jac()

    # if there are other variables, split by the Jacobian
    if not optimizer.only_heights:
        J_h, J_other = J_h
    else:
        J_other = np.zeros(shape=(J_h.shape[0],0))

    # split by real, imaginary components
    if np.any(np.iscomplex(F_h)) or np.any(np.iscomplex(J_h))\
        or np.any(np.iscomplex(J_other)):
        F_h = np.concatenate((F_h.real, F_h.imag))
        J_h = np.block([
                [J_h.real, J_other.real],
                [J_h.imag, J_other.imag]
            ])
    else:
        J_h = np.hstack([J_h, J_other])

    # solve via least squares
    step,res = sp.linalg.lstsq(J_h, -F_h, lapack_driver='gelsy')[0:2]

    return step
