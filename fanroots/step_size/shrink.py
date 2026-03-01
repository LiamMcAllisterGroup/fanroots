# =============================================================================
# Unless this comment is removed, this code is meant solely for members of Liam
# McAllister's group or other people associated with the 'private' variant of
# CYTools.
#
# Those people should feel free to use/modify this code as they see fit.
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description: Apply heuristic shortening of step size until residual decreases
# -----------------------------------------------------------------------------

def shrink(optimizer, step, tol=1e-8):
    alpha = 1
    res0 = optimizer.res_norm(optimizer.x())

    while True:
        res = optimizer.res_norm(optimizer.x() + alpha*step)
        if res<=res0:
            # step decreases residual... accept it!
            break
        else:
            # stepped too far... shrink it!
            alpha /= 2
            if alpha < tol:
                return 0

    return alpha
