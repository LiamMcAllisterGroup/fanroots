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
#
# Note: fanroots' Jacobians are typically rectangular (m != n), so the literal
# square-system Newton step J^{-1} F isn't well-defined. The standard fix is to
# solve via least squares (lstsq), which makes Newton mathematically identical
# to Gauss-Newton in this setting. We just alias to propose_gauss_newton.
# -----------------------------------------------------------------------------

from fanroots.step_proposal.gauss_newton import propose_gauss_newton

propose_newton = propose_gauss_newton
