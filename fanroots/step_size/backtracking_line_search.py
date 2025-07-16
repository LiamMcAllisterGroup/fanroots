# =============================================================================
# Unless this comment is removed, this code is meant solely for members of Liam
# McAllister's group or other people associated with the 'private' variant of
# CYTools.
#
# Those people should feel free to use/modify this code as they see fit.
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description: Apply backtracking line search to optimize step sizes
# -----------------------------------------------------------------------------

import numpy as np

def backtracking_line_search(optimizer, step, tau=0.5, c=0.5, beta=0.8):
	res0 = optimizer.res_norm()
	grad = optimizer.grad()

	alpha = 1
	while True:
		res = optimizer.res_norm(optimizer.x()+alpha*step)
		gradient_term = c*alpha*np.dot(grad,step)
		if res<=res0+gradient_term:
			return alpha

		alpha *= beta
