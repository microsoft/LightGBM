Documentation
=============

Documentation for LightGBM is generated using `Sphinx <http://www.sphinx-doc.org/>`__
and `Breathe <https://breathe.readthedocs.io/>`__, which works on top of `Doxygen <http://www.doxygen.nl/index.html>`__ output.

List of parameters and their descriptions in `Parameters.rst <./Parameters.rst>`__
is generated automatically from comments in `config file <https://github.com/microsoft/LightGBM/blob/master/include/LightGBM/config.h>`__
by `this script <https://github.com/microsoft/LightGBM/blob/master/helpers/parameter_generator.py>`__.

After each commit on ``master``, documentation is updated and published to `Read the Docs <https://lightgbm.readthedocs.io/>`__.

Build
-----

You can build the documentation locally. Just install Doxygen and run in ``docs`` folder

.. code:: sh

    pip install -r requirements.txt
    make html

Unfortunately, documentation for R code is built only on our site, and commands above will not build it for you locally.
Consider using common R utilities for documentation generation, if you need it.

If you faced any problems with Doxygen installation or you simply do not need documentation for C code, it is possible to build the documentation without it:

.. code:: sh

    pip install -r requirements_base.txt
    export C_API=NO || set C_API=NO
    make html
