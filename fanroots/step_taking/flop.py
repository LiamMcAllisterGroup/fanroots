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
#              objective function depends on the intersection numbers, kappa.
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
            Ancillary data: num_flips, step_scaling, failure_mode,
            walk_status (the wall the walk stopped at, if any) and
            step_norm (the distance walked).
    """
    def __init__(self, max_num_flips=1, check_triang=False):
        self.max_num_flips = max_num_flips
        self.check_triang = check_triang

    @staticmethod
    def _hyperplanes(triang):
        """The secondary cone's hyperplanes, as flip_linear computes them,
        cached on the triangulation (flip_linear recomputes them per call)."""
        H = getattr(triang, '_fanroots_sc', None)
        if H is None:
            H = np.array(triang.secondary_cone_hyperplanes(via_circuits=True,
                                                           verbosity=-1))
            triang._fanroots_sc = H
        return H

    @staticmethod
    def _kappa_hooks():
        """
        flip_linear hooks that carry the intersection numbers through the
        walk: the start fan's are taken from its cache (computed only if it
        has none) and each flip updates them incrementally. None if CYTools'
        flop update is unavailable.
        """
        try:
            from cytools.vector_config.fan import flop as kappa_flop
        except ImportError:
            return None

        def hook_init(fan):
            k = getattr(fan, '_fanroots_kappa', None)
            if k is None:
                k = fan._fanroots_kappa = fan.intersection_numbers(
                    pushed_down=True, in_basis=True, as_np_array=True)
            fan.kappa = k

        def hook_flip(fanpre, fanpost, circ):
            fanpost.kappa = kappa_flop(fanpre, fanpre.kappa, circ)

        return hook_init, hook_flip

    @staticmethod
    def _walk(optimizer, triang, h_from, h_to, max_flips):
        # fast path: most steps stay inside the current cone. This returns
        # exactly what flip_linear returns in that case (status 1, h_to, the
        # same fan, its hyperplanes, 0 flips), using flip_linear's own tests
        # (h_from strictly inside, h_to inside with >= 0); anything else
        # takes the full walk, which raises where flip_linear raises
        H = FlopStep._hyperplanes(triang)
        if np.all(H @ h_from > 0) and np.all(H @ h_to >= 0):
            return 1, np.array(h_to), triang, H, 0
        walk = dict(
            h_target=h_to,
            h_init=h_from,
            stop_at_deletion=True,
            max_N_flips=max_flips,
            verbosity=optimizer.verbosity-1,
            check_regularity=False,
        )
        if not getattr(optimizer, 'carry_kappa', False):
            # the intersection numbers are not carried through the walk (see
            # FanRoots.carry_kappa): walk without them
            return triang.flip_linear(**walk)
        hooks = FlopStep._kappa_hooks()
        if hooks is None:
            return triang.flop_linear(**walk)
        hook_init, hook_flip = hooks
        return triang.flip_linear(**walk, hook_init=hook_init,
                                  hook_flip=hook_flip)

    def __call__(self, optimizer, step, project=False, tau=1e-4):
        # current, target heights
        h_start  = optimizer.heights

        # check triangulation
        if self.check_triang:
            assert optimizer.triang.secondary_cone().contains(h_start)

        # try to walk along the step
        h_target = h_start + step
        status, h_curr, triang, sc, num_flips = self._walk(
            optimizer, optimizer.triang, h_start, h_target, self.max_num_flips)

        # flip_linear gives 1 on success, else an Exception naming the wall hit
        walk_status = str(status) if isinstance(status, Exception) else None

        # what the walk covered; min_step_size is compared against this
        walk_dist = np.linalg.norm(h_curr - h_start)

        extra = {
            'walk_status': walk_status,
            'step_norm': float(walk_dist),
            }

        # check how far along the step we actually moved
        denom = np.dot(step,step)
        if denom == 0:
            success   = False
            r         = 0
            # reached only when the step is exactly zero, i.e. the step-size
            # optimizer found no scaling that lowered the residual
            fail_mode = "step scaled to zero: no step size reduced the residual"
            anc = {
                'num_flips': num_flips,
                'step_scaling': r,
                'failure_mode': fail_mode, # None indicates success
                **extra,
                }

            return success, h_curr, triang, anc

        r = np.dot(h_curr-h_start,step)/denom

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
        else:
            # determine if the step was a success
            success = bool(walk_dist >= optimizer.min_step_size)

            # save success/fail info. FanRoots decides whether a failed step
            # ends the solve (and records why), so nothing is halted here
            fail_mode = None if success else "step too small"

        anc = {
            'num_flips': num_flips,
            'step_scaling': r,
            'failure_mode': fail_mode, # None indicates success
            **extra,
            }

        return success, h_curr, triang, anc
