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
# Description: Propose an optimization step h->h+step in a fan using the
#              Levenberg–Marquardt algorithm (LMA).
# -----------------------------------------------------------------------------

import numpy as np
import scipy as sp

import warnings

def lma_idk(F, J, lmbda, scaled):
    """
    Compute Levenberg-Marquardt step using augmented least squares.
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
    return step

def lma(F, J, JTF, lmbda, scaled):
    """
    Solve F = 0 using the Levenberg-Marquardt algorithm (LMA).

    Disjoint from any fan considerations. Subsumes both Gauss-Newton
    (lmbda=0) and gradient descent (lmbda->inf; lr=1/lmbda).

    See https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm.

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

    The additive lmbda*L matrix is a semi-adhoc damping term, making
    this LMA.

    Parameters
    ----------
    F : ndarray of shape (m,)
        The value of the function at the current location.
    J : ndarray of shape (m, n)
        The value of the Jacobian at the current location.
    JTF : ndarray of shape (n,)
        The product J.T @ F.
    lmbda : float
        The damping factor/Marquardt parameter.
    scaled : bool
        If True, use D=diag(J.T@J). Otherwise, use D=1.

    Returns
    -------
    step : ndarray of shape (n,)
        The proposed step to take.
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
    elif False:
        step,res = sp.linalg.lstsq(M, -JTF, lapack_driver='gelsy')[0:2]
    else:
        step = np.linalg.solve(M, -JTF)

    return step

def propose_lma(optimizer, lmbda=0, scaled=False):
    """
    Propose a step h->h+step using the Levenberg-Marquardt algorithm.

    See https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm.
    See https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm.

    Solve F(h, x) = 0 using the Levenberg-Marquardt algorithm (LMA),
    where h are the heights (point in the fan) and x are some optional
    other parameters. Also known as damped least squares. Generalizes
    the Gauss-Newton algorithm (lmbda=0) and gradient descent
    (lmbda>>0).

    In case F is complex, we split the real/imaginary components,
    effectively solving
        F'(h, x) = [Re(F(h,x)); Im(F(h,x))] = 0.
    This requires modifying
        J'(h, x) = [Re(J(h,x)); Im(J(h,x))]

    Parameters
    ----------
    optimizer : FanRoots
        The FanRoots optimizer containing the current state.
    lmbda : float, optional
        The damping factor. Defaults to 0.
    scaled : bool, optional
        If True, use D=diag(J.T@J). Otherwise, use D=1.
        Defaults to False.

    Returns
    -------
    step : ndarray of shape (n,)
        The proposed step, where n = len(heights) + len(other).
        Contains the step in heights (and optionally other parameters,
        concatenated).
    """
    raise NotImplementedError("lambda setting isn't correct yet... must be dynamic")
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
    step = lma(F_h, J_h, optimizer.grad(), lmbda=lmbda, scaled=scaled)

    return step
