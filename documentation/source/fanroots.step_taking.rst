fanroots.step\_taking
=====================

.. currentmodule:: fanroots.step_taking

.. automodule:: fanroots.step_taking

Computational graph
-------------------

The step-taking layer decides how the optimiser actually moves through the
secondary fan from h to h + step. The critical difference between the two
methods is the cost of updating the triangulation and intersection numbers:

- JumpStep recomputes the triangulation from scratch at each step via
  ``vc.subdivide(h + step)``. This is more expensive per step but makes no
  assumption about the current cone and handles large jumps.

- FlopStep walks through chamber walls flip by flip using regfans'
  ``flip_linear(h_target, ...)``, taking at most ``max_num_flips`` flips per
  step. Each flip is cheap because only adjacent simplices change; intersection
  numbers update incrementally. Efficient near convergence when steps are small.

The ``step_taking_schedule`` parameter lets a single run switch between both
strategies depending on the current step size. VolumeFinder uses JumpStep on
the first step and whenever the last accepted step size is >= 1, and
FlopStep (``max_num_flips=10``) once the step size drops below 1.

.. raw:: html
   :file: _static/figures/f5_step_taking.html

Jump step
---------

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst

   JumpStep

Flop step
---------

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst

   FlopStep
