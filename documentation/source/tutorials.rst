Tutorials
=========

This page collects the executable notebooks.  Use it as the entry point once
you want to run code rather than read background material.

Choosing a path
---------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Volume finder
      :link: notebooks/01_volume_finder
      :link-type: doc

      Work through the VolumeFinder notebook to see the full fanroots workflow
      end-to-end: geometry setup, optimisation, diagnostics, step-taking strategy
      comparison, and batch mode.

   .. grid-item-card:: API reference
      :link: fanroots
      :link-type: doc

      Go to the module reference for precise class and function signatures.

Tutorial catalogue
------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Notebook
     - Use it for
   * - :doc:`VolumeFinder: prescribing divisor volumes <notebooks/01_volume_finder>`
     - | End-to-end demonstration: geometry setup, optimiser construction, convergence
       | diagnostics, step-taking strategy comparison, and batch mode via swarm().

.. toctree::
   :hidden:
   :maxdepth: 2

   notebooks/01_volume_finder
