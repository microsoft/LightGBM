Documentation
=============

Documentation for LightGBM is generated using `Sphinx <http://www.sphinx-doc.org/>`__.

List of parameters and their descriptions in `Parameters.rst <./Parameters.rst>`__
is generated automatically from comments in `config file <https://github.com/Microsoft/LightGBM/blob/master/include/LightGBM/config.h>`__
by `this script <https://github.com/Microsoft/LightGBM/blob/master/helper/parameter_generator.py>`__.

After each commit on ``master``, documentation is updated and published to `Read the Docs <https://lightgbm.readthedocs.io/>`__.

Build
-----

You can build the documentation locally. Just run in ``docs`` folder

for Python 3.x:

.. code:: sh

    pip install sphinx "sphinx_rtd_theme>=0.3"
    make html

 
for Python 2.x:

.. code:: sh

    pip install mock sphinx "sphinx_rtd_theme>=0.3"
    make html
