# =============================================================================
# Unless this comment is removed, this code is meant solely for members of Liam
# McAllister's group or other people associated with the 'private' variant of
# CYTools.
#
# Those people should feel free to use/modify this code as they see fit.
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description: Apply ternary search to optimize step sizes
# -----------------------------------------------------------------------------

def ternary_raw(f, left, right, absolute_precision):
    """
    Taken from https://en.wikipedia.org/wiki/Ternary_search

    Adjusted to minimize f, not maximize.

    Assumes f is unimodular in interval [left, right].
    """
    if abs(right - left) < absolute_precision:
        return (left + right) / 2

    left_third = (2 * left + right) / 3
    right_third = (left + 2 * right) / 3

    if f(left_third) < f(right_third):
        # minimum occurs in range [left, right_third]
        return ternary_raw(f, left, right_third, absolute_precision)
    else:
        # minimum occurs in range [left_third, right]
        return ternary_raw(f, left_third, right, absolute_precision)

def ternary(optimizer, step, absolute_precision=1e-1):
    """
    ASSUME UNIMODULAR
    """
    f = lambda alpha: optimizer.res_norm(optimizer.x() + alpha*step)

    return ternary_raw(f, 0, 1, absolute_precision)