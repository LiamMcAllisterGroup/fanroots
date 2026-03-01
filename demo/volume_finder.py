# =============================================================================
# Unless this comment is removed, this code is meant solely for members of Liam
# McAllister's group or other people associated with the 'private' variant of
# CYTools.
#
# Those people should feel free to use/modify this code as they see fit.
# =============================================================================

#import sys; sys.path.append('..')
from lib.util.fan_root.src.fan_root import FanRoots
from lib.util.fan_root.src.step_taking import flop, jump
import numpy as np

class VolumeFinder(FanRoots):
    def __init__(self,
        # required for function definition
        target: np.ndarray,

        # initial parameters
        heights0: "ArrayLike" = None,

        # step proposal/taking
        step_taking_schedule  = None,
        learning_rate: float  = None,
        
        # all other arguments (see FanRoots)
        **kwargs):
        """
        **Description:**
        Construct a VolumeFinder object (derived from FanRoots) to solve for
        heights/kahler parameters satisfying `divisor_volumes = target`.

        **Arguments:**
        - `target`: The target divisor volumes
        - `heights0`: Starting value of h to use.
        - `step_taking_schedule`: A schedule setting the step taking method.
            Requires a list of tuples. Use method tuple[1] if check tuple[0]
            passes. Checks in order of the list.
        - `learning_rate`: Scale down the proposed step by this factor, before
            doing any step size optimization. Normally paired with trivial step
            size optimization (always accept proposed step).
        - `include_corrections: Whether to include general (non-GV) corrections.
        - `include_ws_instantons`: Whether to include WSI corrections.
        - `gs`: The value of the string coupling to use.
        - `bfields`: The value of the B-fields.
        - `precomputed_gvs`: Precomputed values of the GVs to use.
        - `correction_scaling`: Scale the corrections by this value. Typically
            want this 0<=correction_scaling<=1. Useful for homotopy studies.

        Also see FanRoots - every other keyword argument passed to VolumeFinder
        will be passed along to FanRoots.

        **Returns:**
        The VolumeFinder class object to handle optimizing the divisor volumes.
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
        status = super().get_status()
        status['target'] = self.target
        return status

    # updates
    # -------
    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, value):
        self.clear_local_cache(clear_momentum=True, clear_finished_state=True)
        self._target = value

# step taking schedules
# ---------------------
crossover_step_size = 1
def jump_rule(opt):
    if opt.last_step_size is None:
        return True
    elif opt.last_step_size>=crossover_step_size:
        return True
    else:
        return False
def flop_rule(opt):
    if opt.last_step_size<crossover_step_size:
        return True
    else:
        return False
def always_true(opt):
    return True

# helper functions
# ----------------
def fct(optimizer, h):
    """
    Function to find a root of,
        f(h) = \\tau(h) - target.
    """
    # compute tau
    # i.e., solve 0.5 kappa_{ijk} (GLSM@h)^j (GLSM@h)^k
    t = optimizer.glsm@h

    if False:
        tau_curr = 0.5*np.einsum(
            'ijk,j,k', 
            optimizer.kappa, t, t,
            optimize=True) # this is unrelated to the optimizer...
    elif False:
        tau_curr = 0.5 * (optimizer.kappa@t)@t
    else:
        # do things in an explicitly-sparse manner...
        # much faster, if we already have kappa_nz and kappa_vals
        i,j,k = optimizer.kappa_nz()
        vals = optimizer.kappa_vals()

        prod = vals * t[j] * t[k]

        tau_curr = np.zeros_like(t)
        np.add.at(tau_curr, i, prod)
        tau_curr *= 0.5
             
    # return tau-target
    return tau_curr-optimizer.target

def jac(optimizer, h):
    t = optimizer.glsm@h
    A = (optimizer.kappa@t)

    return A@optimizer.glsm
