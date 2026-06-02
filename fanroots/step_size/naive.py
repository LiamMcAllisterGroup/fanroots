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
# Description: Accept the naive step size (i.e, use a scaling of 1).
# -----------------------------------------------------------------------------

def naive_scaling(optimizer, step):
    """
    Return a step scaling of 1, accepting the full proposed step.

    Used when the caller wants to skip any line search and take the
    complete step as proposed by the step-proposal module.

    Parameters
    ----------
    optimizer : FanRoots
        The FanRoots instance owning the current state.
    step : ndarray of shape (n,)
        The proposed step direction.

    Returns
    -------
    alpha : float
        Always 1.
    """
    return 1
