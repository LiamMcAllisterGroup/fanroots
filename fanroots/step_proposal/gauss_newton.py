# =============================================================================
# Unless this comment is removed, this code is meant solely for members of Liam
# McAllister's group or other people associated with the 'private' variant of
# CYTools.
#
# Those people should feel free to use/modify this code as they see fit.
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description: Propose an optimization step h->h+step in a fan using the
#              Gauss-Newton algorithm (LMA).
# -----------------------------------------------------------------------------

import numpy as np
import scipy as sp

import warnings
import time
    
def propose_gauss_newton(optimizer):
    """
    **Description:**
    See https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm.

    We solve F(h, x) = 0 using the Gauss-Newton algorithm for
        - h the heights (point in the fan) and
        - x some optional other parameters.

    (
    In case F is complex, we split the real/imaginary components, effectively
    solving
        F'(h, x) = [Re(F(h,x)); Im(F(h,x))] = 0.
    This requires modifying
        J'(h, x) = [Re(J(h,x)); Im(J(h,x))]
    )

    In case F is complex, we split the real/imaginary components, effectively
    solving
        F'(h, x) = [Re(F(h,x)); Im(F(h,x))] = 0.
    This requires modifying
        J'(h, x) = [Re(J(h,x)); Im(J(h,x))]

    Agrees with Levenberg–Marquardt algorithm (LMA) when that algorithm has
    lambda=0.

    **Arguments:**
    - `optimizer`: The FanRoots optimizer containing the current state.

    **Returns:**
    - `step`: The step in parameters. If there are more parameters than just
        heights, then we concatenate the step in these parameters.
    - `cond`: The condition number arising in the computation.
    """
    # fetch the value of the function of interest F (and its Jacobian, J)
    F = optimizer.fct()
    J_h = optimizer.jac()

    # if there are other variables, split by the Jacobian
    if not optimizer.only_heights:
        J_h, J_other = J_h
    else:
        J_other = np.zeros(shape=(J_h.shape[0],0))

    # split by real, imaginary components
    if np.any(np.iscomplex(F)) or np.any(np.iscomplex(J_h))\
        or np.any(np.iscomplex(J_other)):
        F = np.concatenate((F.real, F.imag))
        J = np.block([
                [J_h.real, J_other.real],
                [J_h.imag, J_other.imag]
            ])
    else:
        J = np.hstack([J_h, J_other])

    # compute the step
    # (solve J.T@J@step = -J.T@F via least squares)
    JTJ = J.T@J
    JTF = optimizer.grad()

    if False:
        step,res = np.linalg.lstsq(JTJ, -JTF, rcond=None)[0:2]
    else:
        step,res = sp.linalg.lstsq(JTJ, -JTF, lapack_driver='gelsy')[0:2]

    cond = np.linalg.cond(JTJ)
    return step, cond
