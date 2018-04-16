Documentation
=============

Documentation for LightGBM is generated using `Sphinx <http://www.sphinx-doc.org/>`__.

After each commit on ``master``, documentation is updated and published to `Read the Docs <https://lightgbm.readthedocs.io/>`__.

Build
-----

You can build the documentation locally. Just run in ``docs`` folder

for Python 3.x:

.. code:: sh

    pip install sphinx sphinx_rtd_theme
    make html

 
for Python 2.x:

.. code:: sh

    pip install mock sphinx sphinx_rtd_theme
    make html
