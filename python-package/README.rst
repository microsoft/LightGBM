LightGBM Python Package
=======================

|PyPI version|


Installation
------------

Preparation
'''''''''''

`setuptools <https://pypi.python.org/pypi/setuptools>`_ is needed. 

For Mac OS X users, gcc with OpenMP support must be installed first. Refer to `wiki <https://github.com/Microsoft/LightGBM/wiki/Installation-Guide#osx>`_ for installing gcc with OpenMP support.

Note: 32-bit python is not supported. Please install 64-bit version.

Install from pip
''''''''''''''''

Install `wheel <http://pythonwheels.com>`_ via ``pip install wheel`` first. For windows user, `VC runtime <https://go.microsoft.com/fwlink/?LinkId=746572>`_ is needed if Visual Studio(2015 or 2017) is not installed.


``pip install lightgbm``


Install source package from pip
*******************************

``pip install --no-binary :all: lightgbm``


Note: Installation from source package require installing `cmake <https://cmake.org/>`_ first.

For Windows user, Visual Studio (or `MS Build <https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017>`_) is needed, and `cmake <https://cmake.org/>`_ must be version 3.8 or higher.

For OSX user, you need to run ```export CXX=g++-7 CC=gcc-7``` before running ```pip install ... ```.

Install GPU version:

``pip install lightgbm --install-option=--gpu``

Note: Boost and OpenCL are needed: details for installation can be found in `gpu-support <https://github.com/Microsoft/LightGBM/wiki/Installation-Guide#with-gpu-support>`_. Need to add OpenCL_INCLUDE_DIR to PATH and export BOOST_ROOT before installation.

Install with MinGW on Windows:

``pip install lightgbm --install-option=--mingw``

Install from GitHub
'''''''''''''''''''

Installation from GitHub require installing `cmake <https://cmake.org/>`_ first. 

For Windows user, Visual Studio (or `MS Build <https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017>`_) is needed, and `cmake <https://cmake.org/>`_ must be version 3.8 or higher.

.. code:: sh

    git clone --recursive https://github.com/Microsoft/LightGBM
    cd LightGBM/python-package
    # export CXX=g++-7 CC=gcc-7 # for OSX
    python setup.py install

``sudo`` (or administrator rights in Windows) may is needed to perform ``python setup.py install``.

Use ``python setup.py install --mingw`` to use MinGW in Windows.

Use ``python setup.py install --gpu`` to enable GPU support. Boost and OpenCL are needed: details for installation can be found in `gpu-support <https://github.com/Microsoft/LightGBM/wiki/Installation-Guide#with-gpu-support>`_.

Examples
--------

Refer to the walk through examples in `python-guide folder <https://github.com/Microsoft/LightGBM/tree/master/examples/python-guide>`_


Troubleshooting
---------------

Refer to `FAQ <https://github.com/Microsoft/LightGBM/tree/master/docs/FAQ.md>`_ 

Developments
------------

The code style of python package follows `pep8 <https://www.python.org/dev/peps/pep-0008/>`_. If you would like to make a contribution and not familiar with pep-8, please check the pep8 style guide first. Otherwise, the check won't pass. You should be careful about:

- E1 Indentation (check pep8 link above)
- E202 whitespace before and after brackets
- E225 missing whitespace around operator
- E226 missing whitespace around arithmetic operator
- E261 at least two spaces before inline comment
- E301 expected 1 blank line in front of and at the end of a method
- E302 expected 2 blank lines in front of and at the end of a function or a class

E501 can be ignored (line too long).

.. |PyPI version| image:: https://badge.fury.io/py/lightgbm.svg
    :target: https://badge.fury.io/py/lightgbm
