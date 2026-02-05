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
#              Update the objective function/kappa by recomputing from scratch
#              kappa at t+step.
# -----------------------------------------------------------------------------

import numpy as np

class JumpStep:
    """
    Method to make a `jump` step method, described below.

    **Description:**
    Attempt to take a step t->t+r*step for r=1.

    Restrict t to live in the (pushed down) secondary subfan of fine, regular
    triangulations. I.e., it is valid K\"ahler parameters. Failure modes are:
        1) t is outside the secondary subfan (doesn't define a subdivision)
        2) t defines a non-triangulation subdivision
        3) t defines a non-fine triangulation

    If a step fails for any of the above reasons, try
        r->0.5*r
        t->t+r*step
    Quit if r<min_step_size.

    **Arguments:**
    - `vc`:     The vector configuration (defines the secondary subfan).
    - `kahler`: The K\"ahler parameters to step from.
    - `step`:   The requested step t->t+step.
    - `min_step_size`: The minimum allowed step size (in terms of the requested
                       step... i.e., in terms of r).
    - `triang`: The triangulation defined by kahler.
    - `kappa`:  The intersection numbers defined by triang. Only used to know
                if we should recompute the intersection numbers after stepping.
    - `verbosity`: The verbosity level. Higher means more verbosity.

    **Returns:**
    - `success`: Whether the step succeeded. I.e., whether some r>0 was found
                 such that t->t+r*step is a valid step.
    - `kahler`:  The K\"ahler parameters after the step.
    - `triang`:  The triangulation defined by t (after the step).
    - `kappa`:   The intersection numbers of triang.
    - `anc`:     The anciliary data.
    """

    def __init__(self):
        return

    def __call__(self, optimizer, step):
        kappa  = None
        triang = None
        h = optimizer.heights + step

        # keep on try to step t->t+r*step
        # for decreasing r, until the step is accepted (or r<min_step_size)
        half_r             = 0.5
        half_min_step_size = 0.5*optimizer.min_step_size

        while True:
            # try the step,
            # see if the step leaves us somwhere valid
            success   = True
            fail_mode = None
            try:
                triang = optimizer.vc.subdivide(heights=h)
                # if we aren't an F(R)T, need to walk back
                if not triang.is_fine():
                    triang    = None
                    #fail_mode = "non-fine"
                    success   = False
                elif not triang.is_triangulation():
                    triang    = None
                    #fail_mode = "non-triangulation"
                    success   = False

            except ValueError:
                # heights don't even define a regular fan
                triang    = None
                #fail_mode = "outside fan"
                success   = False

            # break if the step was OK
            if success:
                break

            # step was NOT OK - step back by 0.5*r*step
            h -= half_r*step
            half_r /= 2
            if half_r<half_min_step_size:
                fail_mode = f"step scaling={2*half_r} is too small"
                if optimizer.verbosity >= 1:
                    print(f"Failed because {fail_mode}")
                success   = False
                break

        # all set :)
        anc = {
            'step_scaling': 2*half_r,
            'failure_mode': fail_mode, # None indicates success
            }
        
        if triang is None:
            h      = optimizer.heights
            triang = optimizer.triang
            kappa  = optimizer.kappa
        else:
            triang = triang
            kappa = triang.intersection_numbers(in_basis=True,
                                                pushed_down=True,
                                                as_np_array=True)

        return success, h, triang, kappa, anc
