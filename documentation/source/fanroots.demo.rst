fanroots.demo
=============

.. currentmodule:: fanroots.demo.volume_finder

.. automodule:: fanroots.demo.volume_finder

Overview
--------

``VolumeFinder`` is a ready-made subclass of ``FanRoots`` for the
inverse-volume problem: given a toric variety and target divisor volumes
:math:`\tau^\text{target} \in \mathbb{R}^{h^{1,1}}`, find heights h such that

.. math::

   \tau_i(h) = \tfrac{1}{2} \kappa_{ijk}\, t^j t^k = \tau^\text{target}_i,
   \qquad t = \text{GLSM} \cdot h.

The residual is :math:`F_i(h) = \tau_i(h) - \tau^\text{target}_i` and the
Jacobian is

.. math::

   J_{ia}(h) = \kappa_{ijk}\, t^k\, \text{GLSM}_{ja},

i.e., contract the last index of :math:`\kappa` with :math:`t`, then
right-multiply by :math:`\text{GLSM}`.

It uses a hybrid step-taking schedule: ``JumpStep`` on the first step
and whenever ``last_step_size`` >= 1, and ``FlopStep`` with up to 10
flips otherwise.

Usage
-----

.. code-block:: python

   from fanroots.demo.volume_finder import VolumeFinder
   import numpy as np

   finder = VolumeFinder(
       vc=my_vc,
       target=np.array([vol1, vol2, vol3]),
       step_proposal="gauss_newton",
       step_size_optimizer="shrink",
       tolerance=1e-6,
       verbosity=1,
   )
   finder.optimize()
   print(finder.get_status())

VolumeFinder class
------------------

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst

   VolumeFinder

Helper functions
----------------

.. autosummary::
   :toctree: _autosummary

   fct
   jac
