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
