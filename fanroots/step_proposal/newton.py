# =============================================================================
# Unless this comment is removed, this code is meant solely for members of Liam
# McAllister's group or other people associated with the 'private' variant of
# CYTools.
#
# Those people should feel free to use/modify this code as they see fit.
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description: Propose an optimization step h->h+step in a fan using Newton's
#              method.
# -----------------------------------------------------------------------------

import numpy as np
import scipy as sp

def propose_newton(optimizer):
    """
    **Description:**
    We solve F(h, x) = 0 using Newton's method for root finding, where
        - h are the heights (point in the fan) and
        - x are some optional other parameters.

    (
    In case F is complex, we split the real/imaginary components, effectively
    solving
        F'(h, x) = [Re(F(h,x)); Im(F(h,x))] = 0.
    This requires modifying
        J'(h, x) = [Re(J(h,x)); Im(J(h,x))]
    )

    This can be derived as:
        F(h+step_h, x+step_x) = F(h, x) + J(h, x)@[step_h, step_x] + ...
    Then, to linear order, F has a root at h+step_h, x+step_x if
        J(h, x)@[step_h, step_x] = -F(h, x).
    This is simple to solve for via least squares:
        step = lstsq(J, -F).

    **Arguments:**
    - `optimizer`: The FanRoots containing the current state.

    **Returns:**
    - `step_t`: The step in K\"ahler parameters.
    - `step_x`: If other_params is not None, then the step in other_params.
    """
    # fetch the value of the function of interest F (and its Jacobian, J)
    F_h = optimizer.fct()
    J_h = optimizer.jac()

    # if there are other variables, split by the Jacobian
    if not optimizer.only_heights:
        J_h, J_other = J_h
    else:
        J_other = np.zeros(shape=(J_h.shape[0],0))

    # split by real, imaginary components
    if np.any(np.iscomplex(F_h)) or np.any(np.iscomplex(J_h))\
        or np.any(np.iscomplex(J_other)):
        F_h = np.concatenate((F_h.real, F_h.imag))
        J_h = np.block([
                [J_h.real, J_other.real],
                [J_h.imag, J_other.imag]
            ])
    else:
        J_h = np.hstack([J_h, J_other])

    # solve via least squares
    #step,res = np.linalg.lstsq(J_h, -F_h, rcond=None)[0:2]
    step,res = sp.linalg.lstsq(J_h, -F_h, lapack_driver='gelsy')[0:2]
    cond = np.linalg.cond(J_h)
    return step, cond

    #if not optimizer.only_heights:
    #    h11 = len(optimizer.heights)
    #    step_h     = step[:h11]
    #    step_other = step[h11:]
    #    return step_h, step_other
    #else:
    #    step_h     = step
    #    return step_h
