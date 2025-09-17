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
#              Levenberg–Marquardt algorithm (LMA).
# -----------------------------------------------------------------------------

import numpy as np
import scipy as sp

import warnings

def lma_idk(F, J, lmbda, scaled):
    """
    Compute Levenberg-Marquardt step using augmented least squares formulation
    """
    m, n = J.shape

    if scaled == False:
        D_sqrt = np.sqrt(lmbda) * np.eye(n)
    else:
        # assumes D is diagonal or symmetric positive-definite
        raise NotImplementedError()
        D_sqrt = np.sqrt(lmbda) * sp.sqrtm(D)

    A = np.vstack([J, D_sqrt])
    b = -np.concatenate([F, np.zeros(n)])

    # Solve the least squares problem ||A s - b||^2
    step, *_ = np.linalg.lstsq(A, b, rcond=None)
    cond = np.linalg.cond(A)
    return step, cond

warnings.warn("lambda setting isn't correct yet... must be dynamic")
def lma(F, J, JTF, lmbda, scaled):
    """
    **Description:**
    We solve F = 0 using the Levenberg–Marquardt algorithm (LMA), disjoint from
    any fan considerations.
    
    Subsumes both Gauss-Newton (lmbda=0) and gradient descent (lmbda->inf;
    lr=1/lmbda).

    **Method:**
    (See https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm.)

    This really solves the least squares problem
        argmin S = argmin \\sum_i F_i(x)^2.
    It does so via stepping
        (J^T@J + lmbda L)@step = J^T@F
    for L a matrix either
        - L = 1 or         # Levenberg
        - L = diag(J^T@J). # Marquardt

    This can be derived as (see wiki)
        F(x+step) = F(x) + J(x)@step + ...
    So
        S(x+step) = \\sum_i (F_i(x) + (J(x)@step)_i + ...)**2
            ~= = \\sum_i (F_i(x) + (J(x)@step)_i)**2
            = (F + J@step).T @ (F + J@step)
            =  F.T @ F
             + F.T @ (J@step)
             + (J@step).T @ F
             + (J@step).T @ (J@step)
            =  F.T @ F
             + 2 F.T @ J @ step 
             + step.T @ J.T @ J @ step
    This has dS/dstep given by
        dS/dstep = 2 F.T @ J + 2 J.T @ J @ step
    and, if dS/dstep=0, we get
        2 J.T @ J @ step == -F.T @ J
    ** AT THIS POINT, WE HAVE REDERIVED GAUSS-NEWTON **

    The additive lmbda*L matrix is a semi-adhoc damping term, making this LMA.

    **Arguments:**
    - `F`: The value of the function at the current location.
    - `J`: The value of the Jacobian at the current location.
    - `lmbda`: The damping factor/Marquardt parameter.
    - `scaled`: If True, use D=diag(J.T@J). Otherwise, use D=1.

    **Returns:**
    - `step`: The proposed step to take.
    """
    JTJ = J.T@J
    if scaled:
        D = np.diag(np.diag(JTJ))
    else:
        D = np.identity(JTJ.shape[0])
    
    # solve (J.T@J + lmbda*L)@step = -J.T@F via least squares
    M = JTJ + lmbda*D
    if False:
        step,res = np.linalg.lstsq(M, -JTF, rcond=None)[0:2]
    else:
        step,res = sp.linalg.lstsq(M, -JTF, lapack_driver='gelsy')[0:2]
    cond = np.linalg.cond(M)
    return step, cond
    
def propose_lma(optimizer, lmbda=0, scaled=False):
    """
    **Description:**
    See https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm.
    See https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm.

    We solve F(h, x) = 0 using the Levenberg–Marquardt algorithm (LMA) for
        - h the heights (point in the fan) and
        - x some optional other parameters.
    This is also known as damped least squares. This is a generalization of the
    Gauss-Newton algorithm (lmbda=0) and gradient descent (lmbda>>0).

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

    **Arguments:**
    - `optimizer`: The FanRoots optimizer containing the current state.

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

    # compute the step
    step,cond = lma(F_h, J_h, optimizer.grad(), lmbda=lmbda, scaled=scaled)

    return step, cond

    #if not optimizer.only_heights:
    #    h11 = len(optimizer.heights)
    #    step_h     = step[:h11]
    #    step_other = step[h11:]
    #    return step_h, step_other
    #else:
    #    step_h     = step
    #    return step_h
