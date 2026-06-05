fanroots.step_size
==================

.. currentmodule:: fanroots.step_size

.. automodule:: fanroots.step_size

Computational graph
-------------------

After a step direction p is proposed, the step-size optimiser selects a scalar
α ∈ [0, 1] so the actual step is α·p (α = 0 signals that no acceptable step
was found). The four available optimisers span a spectrum from always accepting
the full step (``naive_scaling``) to searching for the minimum residual on
[0, 1] (``ternary``). ``shrink`` and ``backtracking_line_search`` both
guarantee the residual does not increase; ``ternary`` finds the approximate
minimum assuming the residual is unimodal along the direction.

The Armijo condition used by ``backtracking_line_search`` is

.. math::

   r(h + \alpha p) \leq r(h) + c \cdot \alpha \cdot \nabla r(h) \cdot p

.. raw:: html
   :file: _static/figures/f4_step_size.html

No scaling
----------

.. autosummary::
   :toctree: _autosummary

   naive_scaling

Shrink
------

.. autosummary::
   :toctree: _autosummary

   shrink

Backtracking line search
------------------------

.. autosummary::
   :toctree: _autosummary

   backtracking_line_search

Ternary search
--------------

.. autosummary::
   :toctree: _autosummary

   ternary
   ternary_raw
