LightGBM Python Package
=======================

Installation
------------

1. Following `Installation Guide <https://github.com/Microsoft/LightGBM/wiki/Installation-Guide>`__ to build first.
   For the windows user, please change the build config to ``DLL``.
2. Install with ``cd python-package; python setup.py install`` 

Note: Make sure you have `setuptools <https://pypi.python.org/pypi/setuptools>`__


Examples
--------

-  Refer also to the walk through examples in `python-guide
   folder <https://github.com/Microsoft/LightGBM/tree/master/examples/python-guide>`__


Troubleshooting
--------

- **Trouble 1**: I see error messages like this when install from github using `python setup.py install`.

    error: Error: setup script specifies an absolute path:

    /Users/Microsoft/LightGBM/python-package/lightgbm/../../lib_lightgbm.so

    setup() arguments must *always* be /-separated paths relative to the
    setup.py directory, *never* absolute paths.

- **Solution 1**: please check `here <http://stackoverflow.com/questions/18085571/pip-install-error-setup-script-specifies-an-absolute-path>`__.


Developments
--------

The code style of python package follows `pep-8 <https://www.python.org/dev/peps/pep-0008/>`__. If you would like to make a contribution and not familiar with pep-8, please check the pep-8 style guide first. You can ignore E501 (line too long).
