fanroots -- Root-finding on secondary fans
==========================================

**fanroots** is a Python library for root-finding and optimisation of
functions that are piecewise-defined on the secondary fan of a toric
variety.  It is aimed at practitioners in string compactification who
need to locate Kahler moduli satisfying prescribed conditions -- such as
fixed divisor volumes -- while respecting the combinatorial chamber
structure encoded by the secondary fan.  The library provides a
modular optimizer (``FanRoots``) whose step-proposal, step-size, and
step-taking strategies are independently configurable, together with a
batch wrapper and a self-contained demo (``VolumeFinder``) that
illustrates a complete research workflow.

How to navigate
---------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: New to the maths
      :link: intro/index
      :link-type: doc

      Start with the introduction chapters.  They explain the secondary
      fan, how the algorithm navigates chamber walls, and the
      mathematical background for the root-finding procedure.

   .. grid-item-card:: Looking for the API
      :link: fanroots
      :link-type: doc

      Go to the module reference when you already know which class or
      function you need.  It covers ``FanRoots``, ``BatchOptimizer``,
      all step-proposal, step-size, and step-taking helpers, and the
      ``VolumeFinder`` demo subclass.

   .. grid-item-card:: Tutorials
      :link: tutorials
      :link-type: doc

      Work through the VolumeFinder notebook for an end-to-end demonstration:
      geometry setup, running the optimiser, inspecting diagnostics, comparing
      step strategies, and batch mode.

   .. grid-item-card:: Source / citing
      :link: https://github.com/LiamMcAllisterGroup/fanroots
      :link-type: url

      Browse the source on GitHub.  If you use fanroots in published
      work, please cite arXiv:2406.13751 (bibtex below).

Recommended first path
-----------------------

For a first pass through the documentation, read:

1. :doc:`Introduction <intro/index>` for the conceptual map of the
   secondary fan and the algorithm design.
2. :doc:`Tutorials <tutorials>` for executable notebooks with worked examples.
3. :doc:`API documentation <fanroots>` once you need precise class and
   function signatures.
4. ``demo/volume_finder.py`` for a complete, self-contained working
   example using ``VolumeFinder``.

Citing fanroots
---------------

If you find this work useful, please cite::

    @article{MacFadden:2024qob,
        author = "MacFadden, Nate and McAllister, Liam",
        title = "{Root-finding on Secondary Fans}",
        eprint = "2406.13751",
        archivePrefix = "arXiv",
        primaryClass = "hep-th",
        year = "2024"
    }

Reference lookup
----------------

* :ref:`genindex`
* :ref:`modindex`

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Start here

   intro/index
   fanroots
   tutorials

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Introduction

   intro/secondary_fan
   intro/algorithm
