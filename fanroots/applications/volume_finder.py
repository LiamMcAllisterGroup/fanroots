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

from __future__ import annotations

#import sys; sys.path.append('..')
from fanroots.fanroots import FanRoots
from fanroots.step_taking import flop, jump
import numpy as np
from numpy.typing import ArrayLike

class VolumeFinder(FanRoots):
    """
    Root-finder that inverts the divisor-volume map :math:`\\tau(h) = \\tau^{\\text{target}}`.

    Subclass of :class:`~fanroots.FanRoots` for the inverse-volume problem:
    given a toric variety and target divisor volumes
    :math:`\\tau^{\\text{target}} \\in \\mathbb{R}^{h^{1,1}}`, find height
    parameters :math:`h^*` such that

    .. math::

       \\tau_i(h^*) = \\tfrac{1}{2}\\,\\kappa_{ijk}\\,t^j\\,t^k
                    = \\tau^{\\text{target}}_i, \\qquad t = \\mathrm{GLSM}\\cdot h.

    Uses a hybrid :attr:`~fanroots.FanRoots.step_taking_schedule` by default:
    :class:`~fanroots.step_taking.JumpStep` when ``last_step_size >= 1``
    and :class:`~fanroots.step_taking.FlopStep` (max 10 flips) otherwise.
    """
    def __init__(self,
        # required for function definition
        target: np.ndarray,

        # initial parameters
        heights0: ArrayLike = None,

        # step proposal/taking
        step_taking_schedule  = None,
        learning_rate: float  = None,
        
        # all other arguments (see FanRoots)
        **kwargs):
        """
        Construct a VolumeFinder to solve for divisor_volumes = target.

        Construct a VolumeFinder object (derived from FanRoots) to
        solve for heights/kahler parameters satisfying
        ``divisor_volumes = target``.

        Every other keyword argument is passed along to FanRoots.

        Parameters
        ----------
        target : np.ndarray of shape (h11,)
            The target divisor volumes.
        heights0 : ArrayLike of shape (N_vecs,), optional
            Starting value of h to use.
        step_taking_schedule : list, optional
            A schedule setting the step taking method. Requires a list
            of tuples. Use method tuple[1] if check tuple[0] passes.
            Checks in order of the list.
        learning_rate : float, optional
            Scale down the proposed step by this factor, before doing
            any step size optimization. Normally paired with trivial
            step size optimization (always accept proposed step).

        Returns
        -------
        VolumeFinder
            The VolumeFinder instance for optimizing divisor volumes.
        """

        self._target = target
        if step_taking_schedule is None:
            step_taking_schedule = [
                [jump_rule, jump.JumpStep()],
                [flop_rule, flop.FlopStep(max_num_flips=10)] 
            ]
        if learning_rate is None:
            learning_rate=0.9

        super().__init__(
            fct=fct,
            jac=jac,
            step_taking_schedule=step_taking_schedule,
            learning_rate=learning_rate,
            heights0 = heights0,
            **kwargs)

    def get_status(self):
        """
        Extend FanRoots.get_status() with the target volume vector.

        Calls the parent get_status() method and appends the current target
        divisor volumes under the key ``'target'``.

        Returns
        -------
        status : dict
            Status dict from FanRoots.get_status() with an additional
            ``'target'`` key containing ``self.target.tolist()``.
        """
        status = super().get_status()
        status['target'] = self.target.tolist()
        return status

    # updates
    # -------
    @property
    def target(self):
        """
        Target divisor volumes.

        Returns
        -------
        target : ndarray of shape (h11,)
            The target divisor volume vector.
        """
        return self._target

    @target.setter
    def target(self, value):
        """Set a new target and clear the local cache."""
        self.clear_local_cache(clear_momentum=True, clear_finished_state=True)
        self._target = value

    # misc
    # ----
    def div_vols(self, h, extrapolate=False):
        """
        Evaluate the divisor volume vector tau_i(h).

        Computes ``tau_i = 0.5 * kappa_{ijk} * t^j * t^k`` where
        ``t = GLSM @ h``.  If ``h`` lies inside the current secondary cone
        and ``extrapolate`` is False, the cached intersection numbers are used;
        otherwise the triangulation and kappa are recomputed at ``h``.  When
        the cached kappa is used, evaluation proceeds via the sparse helpers
        ``kappa_nz()`` and ``kappa_vals()`` for efficiency.

        Parameters
        ----------
        h : ndarray of shape (N_vecs,)
            Height vector.
        extrapolate : bool, optional
            If True, use the current kappa even when ``h`` is outside the
            current secondary cone. Defaults to False.

        Returns
        -------
        tau : ndarray of shape (h11,)
            Divisor volumes evaluated at ``h``.
        """
        # compute tau
        # i.e., solve 0.5 kappa_{ijk} (glsm@h)^j (glsm@h)^k
        t = self.glsm@h

        # evaluate at the current location using correct kappa
        if (not extrapolate) and (not self.triang.secondary_cone().contains(h)):
            tri = self.vc.triangulate(heights=h)
            kappa = tri.intersection_numbers(in_basis=True,
                                             pushed_down=True,
                                             as_np_array=True)
            return 0.5 * (kappa@t)@t

        # evaluate with the current chamber's cached kappa (self.kappa); if h is
        # outside this chamber's Kahler cone, this is a deliberate smooth
        # extrapolation of the chamber's volume polynomial (the exact /
        # new-chamber case is the branch above). the 0.5 * kappa_{ijk} t_j t_k
        # contraction is done sparsely -- fastest by far at the high h11 (>~90)
        # targeted here, where kappa is ~0.1-0.4% dense (see benchmarks/);
        # bincount beats add.at
        i, j, k = self.kappa_nz()
        vals = self.kappa_vals()
        tau_curr = 0.5 * np.bincount(i, weights=vals * t[j] * t[k], minlength=len(t))

        return tau_curr

# step taking schedules
# ---------------------
crossover_step_size = 1
def jump_rule(opt):
    """Return True when the step size warrants a JumpStep."""
    if opt.last_step_size is None:
        return True
    elif opt.last_step_size>=crossover_step_size:
        return True
    else:
        return False
def flop_rule(opt):
    """Return True when the step size warrants a FlopStep."""
    if opt.last_step_size<crossover_step_size:
        return True
    else:
        return False
def always_true(opt):
    """Always return True (unconditional schedule criterion)."""
    return True

# helper functions
# ----------------
def fct(optimizer, h):
    """
    Residual function passed to FanRoots: F(h) = tau(h) - tau_target.

    Evaluates the divisor volumes at ``h`` with extrapolation enabled (so the
    current kappa is always used) and returns the difference from the target.

    Parameters
    ----------
    optimizer : VolumeFinder
        The optimiser instance providing ``kappa``, ``GLSM``, and ``target``.
    h : ndarray of shape (N_vecs,)
        Current height vector.

    Returns
    -------
    residual : ndarray of shape (h11,)
        Residual F(h) = tau(h) - tau_target.
    """
    return optimizer.div_vols(h,extrapolate=True)-optimizer.target

def jac(optimizer, h, extrapolate=False):
    """
    Jacobian of fct with respect to h.

    Returns ``(kappa @ t) @ GLSM`` where ``t = GLSM @ h``.  If ``h`` is
    outside the current secondary cone and ``extrapolate`` is False, the
    triangulation and kappa are recomputed at ``h``; otherwise the cached
    kappa is used.

    Parameters
    ----------
    optimizer : VolumeFinder
        The optimiser instance providing ``kappa``, ``GLSM``, and the
        current triangulation.
    h : ndarray of shape (N_vecs,)
        Current height vector.
    extrapolate : bool, optional
        If True, use the current kappa even when ``h`` is outside the
        current secondary cone. Defaults to False.

    Returns
    -------
    J : ndarray of shape (h11, N_vecs)
        Jacobian dF/dh evaluated at ``h``.
    """
    if (not extrapolate) and (
        not optimizer.triang.secondary_cone().contains(h)
    ):
        tri   = optimizer.vc.triangulate(heights=h)
        kappa = tri.intersection_numbers(in_basis=True,
                                         pushed_down=True,
                                         as_np_array=True)
    else:
        kappa = optimizer.kappa

    t = optimizer.glsm@h
    A = (kappa@t)

    return A@optimizer.glsm
