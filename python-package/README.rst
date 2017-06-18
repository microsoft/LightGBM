LightGBM Python Package
=======================

Installation
------------

Preparation
'''''''''''

You need to install `cmake <https://cmake.org/>`_ and `setuptools <https://pypi.python.org/pypi/setuptools>`_ first. 

For Windows users, Visual Studio (or `MS Build <https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017>`_) is needed. You also can use MinGW instead if installing from GitHub.

For Mac OS X users, gcc with OpenMP support must be installed first. Refer to `wiki <https://github.com/Microsoft/LightGBM/wiki/Installation-Guide#osx>`_ for installing gcc with OpenMP support.

Note: 32-bit python is not supported. Please install 64-bit version.

Install from pip
''''''''''''''''

``pip install lightgbm``

For the MinGW build in Windows and GPU support, please install the latest version from GitHub.

Install from GitHub
'''''''''''''''''''

.. code:: sh

    git clone --recursive https://github.com/Microsoft/LightGBM
    cd LightGBM/python-package
    python setup.py install

You may need to use ``sudo`` (or administrator rights in Windows) to perform ``python setup.py install``.

Use ``python setup.py install --mingw`` to use MinGW in Windows.

Use ``python setup.py install --gpu`` to enable GPU support. You will need to install Boost and OpenCL first: details for installation can be found in `gpu-support <https://github.com/Microsoft/LightGBM/wiki/Installation-Guide#with-gpu-support>`_.

Examples
--------

Refer to the walk through examples in `python-guide folder <https://github.com/Microsoft/LightGBM/tree/master/examples/python-guide>`__


Troubleshooting
---------------

Refer to `FAQ <https://github.com/Microsoft/LightGBM/tree/master/docs/FAQ.md>`__ 

Developments
--------

The code style of python package follows `pep8 <https://www.python.org/dev/peps/pep-0008/>`__. If you would like to make a contribution and not familiar with pep-8, please check the pep8 style guide first. Otherwise, you won't pass the check. You should be careful about:

- E1 Indentation (check pep8 link above)
- E202 whitespace before and after brackets
- E225 missing whitespace around operator
- E226 missing whitespace around arithmetic operator
- E261 at least two spaces before inline comment
- E301 expected 1 blank line in front of and at the end of a method
- E302 expected 2 blank lines in front of and at the end of a function or a class

You can ignore E501 (line too long).
