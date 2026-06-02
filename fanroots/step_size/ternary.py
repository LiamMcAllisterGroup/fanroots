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
# Description: Apply ternary search to optimize step sizes
# -----------------------------------------------------------------------------

def ternary_raw(f, left, right, absolute_precision):
    """
    Find the minimum of a unimodal function on [left, right] by ternary search.

    Recursively bisects [left, right] into thirds and discards the third whose
    endpoint has the higher function value, converging to within
    absolute_precision of the true minimum. Adapted from
    https://en.wikipedia.org/wiki/Ternary_search (adjusted to minimise, not
    maximise).

    Parameters
    ----------
    f : Callable
        Univariate, unimodal function to minimise on [left, right].
    left : float
        Left endpoint of the search interval.
    right : float
        Right endpoint of the search interval.
    absolute_precision : float
        Convergence threshold; search stops when right - left < absolute_precision.

    Returns
    -------
    x : float
        Approximate location of the minimum of f on [left, right].
    """
    if abs(right - left) < absolute_precision:
        return (left + right) / 2

    left_third = (2 * left + right) / 3
    right_third = (left + 2 * right) / 3

    if f(left_third) < f(right_third):
        # minimum occurs in range [left, right_third]
        return ternary_raw(f, left, right_third, absolute_precision)
    else:
        # minimum occurs in range [left_third, right]
        return ternary_raw(f, left_third, right, absolute_precision)

def ternary(optimizer, step, absolute_precision=1e-1):
    """
    Find the alpha in [0, 1] minimising r(x + alpha*step) by ternary search.

    Assumes the residual profile along step is unimodal. Emits a warning if
    absolute_precision is small enough that the expected recursion depth may
    exceed Python's recursion limit.

    Parameters
    ----------
    optimizer : FanRoots
        The FanRoots instance owning the current residual.
    step : ndarray of shape (n,)
        The proposed step direction.
    absolute_precision : float, optional
        Search precision passed to ternary_raw; controls how tightly the
        minimising alpha is located. Defaults to 0.1.

    Returns
    -------
    alpha : float
        Scale factor in [0, 1] that approximately minimises
        r(x + alpha*step) along the given step direction.
    """
    import sys, math, warnings
    expected_depth = math.log(1 / absolute_precision) / math.log(1.5)
    if expected_depth > sys.getrecursionlimit() - 50:
        warnings.warn(
            f"ternary: absolute_precision={absolute_precision} may cause"
            f" RecursionError (expected depth ~{int(expected_depth)})"
        )

    f = lambda alpha: optimizer.res_norm(optimizer.x() + alpha*step)

    return ternary_raw(f, 0, 1, absolute_precision)