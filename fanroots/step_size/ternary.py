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

import math
import sys
import warnings


def ternary_raw(f, left, right, absolute_precision):
    """
    Ternary search to minimize f over [left, right].

    Taken from https://en.wikipedia.org/wiki/Ternary_search

    Adjusted to minimize f, not maximize.

    Assumes f is unimodal in interval [left, right].
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
    ASSUME UNIMODULAR
    """
    expected_depth = math.log(1 / absolute_precision) / math.log(1.5)
    if expected_depth > sys.getrecursionlimit() - 50:
        warnings.warn(
            f"ternary: absolute_precision={absolute_precision} may cause"
            f" RecursionError (expected depth ~{int(expected_depth)})"
        )

    f = lambda alpha: optimizer.res_norm(optimizer.x() + alpha*step)

    return ternary_raw(f, 0, 1, absolute_precision)