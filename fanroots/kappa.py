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
# Description: Canonical (integer-valued) intersection numbers.
# -----------------------------------------------------------------------------

import warnings

import numpy as np


class NonIntegralKappaWarning(UserWarning):
    """
    Intersection numbers were not within tolerance of integers, so they were
    left unrounded (e.g. a basis that is not integral).
    """


def integral_kappa(kappa, tol=1e-6):
    """
    Round intersection numbers to the integers they represent.

    The intersection numbers are integers, but CYTools computes them in
    floating point, and different routes (a fresh computation, or an
    incremental update through a flop) carry different ~1e-13 round-off.
    Rounding makes the result independent of the route, so the solve does
    not depend on how (or whether) the numbers were cached.

    Parameters
    ----------
    kappa : ArrayLike
        The intersection numbers.
    tol : float, optional
        Largest allowed distance of an entry from an integer. If any entry is
        farther, a NonIntegralKappaWarning is raised and kappa is returned
        unrounded. Defaults to 1e-6.

    Returns
    -------
    kappa : ndarray of float
        Integer-valued (as floats), or the input unchanged if not integral.
    """
    kappa = np.asarray(kappa, dtype=float)
    rounded = np.rint(kappa)
    err = float(np.max(np.abs(kappa - rounded))) if kappa.size else 0.0
    if err > tol:
        warnings.warn(
            f"intersection numbers are up to {err:.3g} from integers; "
            "leaving them unrounded",
            NonIntegralKappaWarning, stacklevel=2)
        return kappa
    # + 0.0 turns the -0.0 that rint gives small negatives into 0.0
    return rounded + 0.0
