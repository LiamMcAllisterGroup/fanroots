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
# Description: Apply heuristic shortening of step size until residual decreases
# -----------------------------------------------------------------------------

def shrink(optimizer, step, tol=1e-8):
    """
    Halve the step scaling until the residual does not increase.

    Repeatedly multiplies the scaling factor alpha by 0.5 until
    r(x + alpha*step) <= r(x). Returns 0 if alpha falls below tol.

    Parameters
    ----------
    optimizer : FanRoots
        The FanRoots instance owning the current residual.
    step : ndarray of shape (n,)
        The proposed step direction (already scaled by momentum and lr).
    tol : float, optional
        Minimum allowed alpha before returning 0. Defaults to 1e-8.

    Returns
    -------
    alpha : float
        Scale factor in (0, 1] such that the residual does not increase,
        or 0 if no such alpha >= tol was found.
    """
    alpha = 1
    res0 = optimizer.res_norm(optimizer.x())

    while True:
        res = optimizer.res_norm(optimizer.x() + alpha*step)
        if res<=res0:
            # step decreases residual... accept it!
            break
        else:
            # stepped too far... shrink it!
            alpha /= 2
            if alpha < tol:
                return 0

    return alpha
