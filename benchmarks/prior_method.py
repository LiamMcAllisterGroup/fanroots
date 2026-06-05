"""Prior divisor-volume solver used as the benchmark baseline.

Extracted, with permission, from the McAllister group's KKLT code as the "prior
method" against which FanRoots is benchmarked (the ``divisor_to_curve_alt``
solver and its ``matrix_inv`` helper).

The body below is reproduced essentially verbatim so the comparison reflects the
original implementation (same variables, control flow, and per-iteration cost --
including the otherwise-unused ``AAinv`` inverse the original computes each
step). The only changes, made for packaging/benchmarking, are:
  - removed the progress ``print`` statements and a stray developer
    ``warnings.warn`` note;
  - added an optional ``max_seconds`` budget (a timeout returns ``None``, like
    the routine's other failure exits) and an iteration counter;
  - added an optional ``divisor_basis`` pinned on each CalabiYau via
    ``set_divisor_basis`` so in_basis quantities match a fixed basis;
  - on success, return ``(kahler_current, cy, n_iter)`` instead of the heights,
    so the benchmark can verify volumes and report iterations.

It solves the same task FanRoots' ``VolumeFinder`` solves: given target divisor
volumes ``tau_target`` (in the GLSM basis), find Kahler parameters realizing
them. The algorithm is a 5th-order perturbative Newton step with adaptive
step-size control and flop detection across secondary-fan walls.

Requires only ``cytools`` (and numpy/scipy).
"""
import time

import numpy as np
from scipy.optimize import least_squares
from cytools import utils


def list_flat(l):
    return [x for y in l for x in y]


def matrix_inv(M, v):
    def F(x):
        return v - M @ x
    return least_squares(F, np.linalg.inv(M) @ v,
                         ftol=1e-14, xtol=1e-14, gtol=1e-14).x


def divisor_to_curve_alt(p, tau_target, kahler_initial=None,
                         maximum_step_size=0.5, target_residual=1e-5,
                         control_target=1e-1, fast_track=True,
                         max_seconds=float("inf"), divisor_basis=None):
    t_start = time.time()

    def _get_cy(triang):
        cy = triang.get_cy()
        if divisor_basis is not None:
            cy.set_divisor_basis(divisor_basis)
        return cy

    phases_traversed = 0
    basis = p.glsm_basis()
    if type(kahler_initial) == type(None):
        t = p.triangulate()
        cy = _get_cy(t)
        toric_k = cy.toric_kahler_cone()
        kahler_initial = toric_k.tip_of_stretched_cone(1)
        hts = utils.kahler_to_heights(p, kahler_initial)
    else:
        hts = utils.kahler_to_heights(p, kahler_initial)
        t = p.triangulate(heights=hts, check_heights=False)
        cy = _get_cy(t)
        toric_k = cy.toric_kahler_cone()
    kahler_current = kahler_initial.copy()
    kahler_last_valid = kahler_current.copy()

    intnums = cy.intersection_numbers()
    if fast_track:
        toric_curves = set(list_flat([[tuple(np.delete(s, i, 0)) for i in range(3)] for s in t.simplices(on_faces_dim=2)]))
        toric_curve_charges = np.array([[intnums.get(tuple(np.sort(list(c) + [b])), 0) for b in basis] for c in toric_curves])
    else:
        toric_curve_charges = cy.toric_mori_cone(in_basis=True).rays()

    tau_current = cy.compute_divisor_volumes(kahler_current, in_basis=True)
    residual = sum(abs((tau_target - tau_current)))

    steps_since_last_transition = 0
    control = 0
    step_size = 0
    n_iter = 0

    while residual > target_residual:
        n_iter += 1
        if time.time() - t_start > max_seconds:
            return None, None, n_iter
        control = 10
        nflops = 2
        step_size = maximum_step_size
        while control > control_target or nflops > 1:
            tau_step = (tau_target - tau_current) * step_size
            AA = cy.compute_AA(kahler_current)
            AAinv = np.linalg.inv(AA)
            # compute perturbation of kahler parameters to fifth order
            kahler_step1 = matrix_inv(AA, tau_step)
            div_vol_kahler_step1 = cy.compute_divisor_volumes(kahler_step1, in_basis=True)
            kahler_step2 = -matrix_inv(AA, div_vol_kahler_step1)
            AA_kahler_step1 = cy.compute_AA(kahler_step1)
            kahler_step3 = -matrix_inv(AA, (AA_kahler_step1 @ kahler_step2))
            div_vol_kahler_step2 = cy.compute_divisor_volumes(kahler_step2, in_basis=True)
            kahler_step4 = -matrix_inv(AA, (div_vol_kahler_step2 + AA_kahler_step1 @ kahler_step3))
            AA_kahler_step2 = cy.compute_AA(kahler_step2)
            kahler_step5 = -matrix_inv(AA, (AA_kahler_step1 @ kahler_step4 + AA_kahler_step2 @ kahler_step3))
            control = sum(abs(kahler_step2)) / sum(abs(kahler_step1))
            kahler_step = kahler_step1 + kahler_step2 + kahler_step3 + kahler_step4 + kahler_step5
            nflops = len(np.where(toric_curve_charges @ (kahler_current + kahler_step) < 0)[0])
            step_size = step_size / 2
            if step_size < 1e-10:
                return None, None, n_iter  # got stuck on a singularity
        step_size = step_size * 2
        kahler_current = kahler_current + kahler_step
        if min(toric_curve_charges @ kahler_current) > 0:
            hts = utils.kahler_to_heights(p, kahler_current)
            t = p.triangulate(heights=hts, check_heights=False)
            if t.is_fine() and t.is_regular() and t.is_star():
                kahler_last_valid = kahler_current.copy()
            else:
                return None, None, n_iter  # ran into Gerald's wall
            tau_current = cy.compute_divisor_volumes(kahler_current, in_basis=True)
            residual = sum(abs((tau_target - tau_current)))

            steps_since_last_transition = steps_since_last_transition + 1
            if steps_since_last_transition > 1:
                steps_since_last_transition = 0
        else:
            hts = utils.kahler_to_heights(p, kahler_current)
            t = p.triangulate(heights=hts, check_heights=False)
            if t.is_fine() and t.is_regular() and t.is_star():
                kahler_last_valid = kahler_current.copy()
                cy = _get_cy(t)
                intnums = cy.intersection_numbers()
                if fast_track:
                    toric_curves = set(list_flat([[tuple(np.delete(s, i, 0)) for i in range(3)] for s in t.simplices(on_faces_dim=2)]))
                    toric_curve_charges = np.array([[intnums.get(tuple(np.sort(list(c) + [b])), 0) for b in basis] for c in toric_curves])
                else:
                    toric_curve_charges = cy.toric_mori_cone(in_basis=True).rays()
                tau_current = cy.compute_divisor_volumes(kahler_current, in_basis=True)
                residual = sum(abs((tau_target - tau_current)))
                steps_since_last_transition = 0
                phases_traversed = phases_traversed + 1
            else:
                return None, None, n_iter  # ran into Gerald's wall

    hts = utils.kahler_to_heights(p, kahler_current)
    t = p.triangulate(heights=hts, check_heights=False)
    cy = _get_cy(t)
    toric_mori_rays = cy.toric_mori_cone(in_basis=True).rays()
    if min(toric_mori_rays @ kahler_current) > 0:
        return kahler_current, cy, n_iter
    else:
        return None, None, n_iter  # ran into Gerald's wall
