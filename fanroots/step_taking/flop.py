# =============================================================================
# Unless this comment is removed, this code is meant solely for members of Liam
# McAllister's group or other people associated with the 'private' variant of
# CYTools.
#
# Those people should feel free to use/modify this code as they see fit.
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
    Method to make a `flop` step method, described below.

    **Description:**
    Attempt to take a step t->t+r*step for r=1.

    Restrict t to live in the (pushed down) secondary subfan of fine, regular
    triangulations. I.e., it is valid K\"ahler parameters. Failure modes are:
        1) t is outside the secondary subfan (doesn't define a subdivision)
        2) t defines a non-triangulation subdivision
        3) t defines a non-fine triangulation

    If a step fails for any of the above reasons, return last valid location.

    **Arguments:**
    - `vc`:     The vector configuration (defines the secondary subfan).
    - `kahler`: The K\"ahler parameters to step from.
    - `step`:   The requested step t->t+step.
    - `min_step_size`: The minimum allowed step size (in terms of the requested
                       step... i.e., in terms of r).
    - `triang`: The triangulation defined by kahler.
    - `kappa`:  The intersection numbers defined by triang. If none are
                provided, then don't compute updates.
    - `verbosity`: The verbosity level. Higher means more verbosity.

    Custom:
    - `max_num_flips`: Limit to taking <= this number of flips.
    - `heights`: The initial heights = vc.jorp(kahler).
    - `check_triang`: Whether to check that triang is indeed defined by kahler,
                      triang=vc.subdivide(heights=vc.jorp(kahler))
    - `check_kappa`: Whether to check intersection numbers after each flop.

    **Returns:**
    - `success`: Whether the step succeeded. I.e., whether some r>0 was found
                 such that t->t+r*step is a valid step.
    - `kahler`:  The K\"ahler parameters after the step.
    - `triang`:  The triangulation defined by t (after the step).
    - `kappa`:   The intersection numbers of triang.
    - `anc`:     The anciliary data.
    """
    def __init__(self, max_num_flips=1, check_triang=False, check_kappa=False):
        self.max_num_flips = max_num_flips
        self.check_triang = check_triang
        self.check_kappa = check_kappa

    def __call__(self, optimizer, step):
        # current, target heights
        if False:
            h_curr   = optimizer.vc.jorp(optimizer.kahler)
            h_target = optimizer.vc.jorp(optimizer.kahler + step)
        else:
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
            check_kappa=self.check_kappa,
            kappa_init=optimizer.kappa,
            verbosity=int(optimizer.verbosity>1),
            print_progress=int(optimizer.verbosity>5)
        )
        
        # read the data
        status, h_curr, triang, sc, num_flips, kappa = out

        # parse the data
        if False:
            t = optimizer.vc.proj(h_curr)
            r = np.dot(t-optimizer.kahler,step)/np.dot(step,step)
        else:
            r = np.dot(h_curr-optimizer.heights,step)/np.dot(step,step)

        # determine if the step was a success
        #if isinstance(status, Exception):
        if np.linalg.norm(optimizer.heights-h_curr) < optimizer.min_step_size:
            success = False
        else:
            success = True

        # save success/fail info
        if success:
            fail_mode = None
        else:
            #fail_mode = status.args
            fail_mode = "step too small"
            if not optimizer.last_step_success:
                optimizer.finished = True

        anc = {
            'num_flips': num_flips,
            #'heights': h_curr,
            #'secondary_cone': sc,
            'step_scaling': r,
            'failure_mode': fail_mode, # None indicates success
            }

        return success, h_curr, triang, kappa, anc
