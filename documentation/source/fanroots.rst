fanroots
========

.. currentmodule:: fanroots


Package architecture
--------------------

fanroots is organised around one core class :class:`FanRoots` that owns the
optimisation state and dispatches three pluggable components: a step-proposal
method, a step-size optimiser, and a step-taking method.  Each component is
interchangeable at construction time via the ``step_proposal``,
``step_size_optimizer``, and ``step_taking_method`` constructor arguments.

When two qualitatively different step-taking behaviours are needed -- for
example, flopping through chamber walls at small step sizes while jumping
directly to a new triangulation at large ones -- the optional
``step_taking_schedule`` argument accepts a list of ``[criteria_fn, method]``
pairs.  At each iteration the list is traversed in order and the first pair
whose ``criteria_fn`` returns ``True`` determines the active method, enabling
hybrid strategies without subclassing.

.. raw:: html
   :file: _static/figures/f1_workflow.html

Module dependency graph
-----------------------

The figure below shows how the sub-packages depend on each other and on the
external libraries ``regfans`` and ``cytools``.  Solid arrows indicate a
required code dependency; dashed arrows indicate an optional "used-by"
relationship.

.. raw:: html
   :file: _static/figures/f2_architecture.html


Core optimiser
--------------

.. autosummary::
    :toctree: _autosummary
    :template: custom-class-template.rst

    FanRoots
    BatchOptimizer

Utilities
~~~~~~~~~

.. autosummary::
    :toctree: _autosummary

    fanroots_from_state


Key methods
-----------

Running the optimiser
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

    FanRoots.step
    FanRoots.optimize
    FanRoots.__next__

Evaluating the function
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

    FanRoots.fct
    FanRoots.jac
    FanRoots.res_norm
    FanRoots.grad

State and diagnostics
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

    FanRoots.get_status
    FanRoots.get_state
    FanRoots.load_state
    FanRoots.load_heights_other
    FanRoots.timing_fct
    FanRoots.timing_jac

Swarm mode
~~~~~~~~~~

.. autosummary::

    FanRoots.swarm


Subpackages
-----------

.. toctree::
    :maxdepth: 1

    fanroots.step_proposal
    fanroots.step_size
    fanroots.step_taking
    fanroots.demo
