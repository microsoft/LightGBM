Documentation
=============

Documentation for LightGBM is generated using `Sphinx <https://www.sphinx-doc.org/>`__
and `Breathe <https://breathe.readthedocs.io/>`__, which works on top of `Doxygen <https://www.doxygen.nl/index.html>`__ output.

List of parameters and their descriptions in `Parameters.rst <./Parameters.rst>`__
is generated automatically from comments in `config file <https://github.com/microsoft/LightGBM/blob/master/include/LightGBM/config.h>`__
by `this script <https://github.com/microsoft/LightGBM/blob/master/helpers/parameter_generator.py>`__.

After each commit on ``master``, documentation is updated and published to `Read the Docs <https://lightgbm.readthedocs.io/>`__.

Build
-----

It is not necessary to re-build this documentation while modifying LightGBM's source code.
The HTML files generated using ``Sphinx`` are not checked into source control.
However, you may want to build them locally during development to test changes.

Docker
^^^^^^

The most reliable way to build the documentation locally is with Docker, using `the same images readthedocs uses <https://hub.docker.com/r/readthedocs/build>`_.

Run the following from the root of this repository to pull the relevant image and run a container locally.

.. code:: sh

    docker run \
        --rm \
        --user=0 \
        -v $(pwd):/opt/LightGBM \
        --env C_API=true \
        --env CONDA=/opt/conda \
        --env PYTHONUNBUFFERED=1 \
        --env READTHEDOCS=true \
        --workdir=/opt/LightGBM/docs \
        --entrypoint="" \
        -it readthedocs/build:ubuntu-20.04-2021.09.23 \
        /bin/bash build-docs.sh

When that code completes, open ``docs/html/index.html`` in your browser.

.. note::

    The navigation in locally-built docs does not link to the local copy of the R documentation. To view the local version of the R docs, open ``docs/_build/html/R/index.html`` in your browser.

Locally
^^^^^^^

You can build the documentation locally. Just install Doxygen and run in ``docs`` folder

.. code:: sh

    pip install breathe sphinx 'sphinx_rtd_theme>0.5'
    make html

Unfortunately, documentation for R code is built only on our site, and commands above will not build it for you locally.
Consider using common R utilities for documentation generation, if you need it.

If you faced any problems with Doxygen installation or you simply do not need documentation for C code, it is possible to build the documentation without it:

.. code:: sh

    pip install -r requirements_base.txt
    export C_API=NO || set C_API=NO
    make html
