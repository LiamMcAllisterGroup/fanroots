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
# Description: Attempt an optimization step t->t+step in a fan, for which the
#              objective function depends on the interesection numbers, kappa.
#
#              Update the objective function/kappa by flopping.
# -----------------------------------------------------------------------------

import numpy as np

class FlopStep:
    """
    Step method that advances through the fan by flopping.

    Attempt to take a step t->t+r*step for r=1.

    Restrict t to live in the (pushed down) secondary subfan of fine,
    regular triangulations (i.e., valid Kahler parameters). Failure
    modes are:
        1) t is outside the secondary subfan (doesn't define a
           subdivision)
        2) t defines a non-triangulation subdivision
        3) t defines a non-fine triangulation

    If a step fails for any of the above reasons, return last valid
    location.

    Parameters
    ----------
    max_num_flips : int, optional
        Limit to taking <= this number of flips. Defaults to 1.
    check_triang : bool, optional
        Whether to check that triang is indeed defined by kahler,
        i.e., triang=vc.subdivide(heights=vc.jorp(kahler)).
        Defaults to False.

    Notes
    -----
    When called, the instance accepts:
        optimizer : FanRoots
            The FanRoots instance with current state (heights, triang,
            min_step_size, verbosity, etc.).
        step : ndarray of shape (N_vecs,)
            The requested step h->h+step.

    And returns:
        success : bool
            Whether some r>0 was found such that t->t+r*step is valid.
        h : ndarray of shape (N_vecs,)
            The heights after the step.
        triang : Fan
            The triangulation at the new location. Kappa accessible
            via triang.kappa (precomputed by flop_linear hooks).
        anc : dict
            Ancillary data.
    """
    def __init__(self, max_num_flips=1, check_triang=False):
        self.max_num_flips = max_num_flips
        self.check_triang = check_triang

    def __call__(self, optimizer, step, project=False, tau=1e-4):
        # current, target heights
        h_curr   = optimizer.heights
        h_target = optimizer.heights + step

        # check triangulation
        if self.check_triang:
            assert optimizer.triang.secondary_cone().contains(h_curr)

        # try to walk along the step
        out = optimizer.triang.flop_linear(
            h_target=h_target,
            h_init=h_curr,
            stop_at_deletion=True,
            max_N_flips=self.max_num_flips,
            verbosity=optimizer.verbosity-1,
            check_regularity=False
        )
        
        # read the data
        status, h_curr, triang, sc, num_flips = out

        # check how far along the step we actually moved
        denom = np.dot(step,step)
        if denom == 0:
            success   = False
            r         = 0
            fail_mode = "hit wall of BG and projection led to non-fine triangulation"
            anc = {
                'num_flips': num_flips,
                'step_scaling': r,
                'failure_mode': fail_mode, # None indicates success
                }

            return success, h_curr, triang, anc

        r = np.dot(h_curr-optimizer.heights,step)/denom

        # project the initial step if r != 1
        if project:
            # get the stopping hyperplane
            n = sc[np.argmin(sc@h_curr)]

            # step h_curr + projected_step, instead of h_curr + step
            # the modifications ensure
            #     dot(n, h_curr + projected_step) = tau * np.linalg.norm(h_curr) > 0
            projected_step   = (
                step
                + (tau*np.linalg.norm(h_curr) - np.dot(n, h_target))
                * n / np.dot(n, n)
            )
            projected_target = h_curr + projected_step

            # might have moved a triangulation...
            triang    = optimizer.vc.triangulate(heights=projected_target)
            success   = triang.is_fine()
            if success:
                h_curr    = projected_target
                fail_mode = None
            else:
                fail_mode = "hit wall of BG and projection led to non-fine triangulation"
                if not optimizer.last_step_success:
                    optimizer.finished = True
        else:
            # determine if the step was a success
            if (np.linalg.norm(optimizer.heights-h_curr)
                    < optimizer.min_step_size):
                success = False
            else:
                success = True

            # save success/fail info
            if success:
                fail_mode = None
            else:
                fail_mode = "step too small"
                if not optimizer.last_step_success:
                    optimizer.finished = True

        anc = {
            'num_flips': num_flips,
            'step_scaling': r,
            'failure_mode': fail_mode, # None indicates success
            }

        return success, h_curr, triang, anc
