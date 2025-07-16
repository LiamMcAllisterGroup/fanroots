# =============================================================================
# Unless this comment is removed, this code is meant solely for members of Liam
# McAllister's group or other people associated with the 'private' variant of
# CYTools.
#
# Those people should feel free to use/modify this code as they see fit.
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description: Propose an optimization step h->h+step in a fan using gradient
#              descent.
# -----------------------------------------------------------------------------

import numpy as np

def propose_gradient_descent(optimizer):
    """
    **Description:**
    We solve F(h, x) = 0 using gradient descent, where
        - h are the heights (point in the fan) and
        - x are some optional other parameters.

    (
    In case F is complex, we split the real/imaginary components, effectively
    solving
        F'(h, x) = [Re(F(h,x)); Im(F(h,x))] = 0.
    This requires modifying
        J'(h, x) = [Re(J(h,x)); Im(J(h,x))]
    )

    This really solves the least squares problem
        argmin S = argmin \\sum_i F_i(x)^2.
    It does so via stepping
        step = - lr \\grad S = - lr jac.T @ F(x)

    **Arguments:**
    - `optimizer`: The FanRoots containing the current state.

    **Returns:**
    - `step_t`: The step in K\"ahler parameters.
    - `step_x`: If other_params is not None, then the step in other_params.
    """
    # compute the step
    step = -optimizer.grad()
    return step, None

    #if not optimizer.only_heights:
    #    h11 = len(optimizer.heights)
    #    step_h     = step[:h11]
    #    step_other = step[h11:]
    #    return step_h, step_other
    #else:
    #    step_h     = step
    #    return step_h