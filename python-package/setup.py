# pylint: disable=invalid-name, exec-used
"""Setup lightgbm package."""
from __future__ import absolute_import
import sys
import os
from setuptools import setup, find_packages
# import subprocess
sys.path.insert(0, '.')

CURRENT_DIR = os.path.dirname(__file__)

libpath_py = os.path.join(CURRENT_DIR, 'lightgbm/libpath.py')
libpath = {'__file__': libpath_py}
exec(compile(open(libpath_py, "rb").read(), libpath_py, 'exec'), libpath, libpath)

LIB_PATH = libpath['find_lib_path']()
print("Install lib_lightgbm from: %s" % LIB_PATH)

# Please use setup_pip.py for generating and deploying pip installation
# detailed instruction in setup_pip.py
setup(name='lightgbm',
      version=open(os.path.join(CURRENT_DIR, 'lightgbm/VERSION')).read().strip(),
      description="LightGBM Python Package",
      install_requires=[
          'numpy',
          'scipy',
      ],
      maintainer='Guolin Ke',
      maintainer_email='guolin.ke@microsoft.com',
      zip_safe=False,
      packages=find_packages(),
      include_package_data=True,
      data_files=[('lightgbm', LIB_PATH)],
      url='hhttps://github.com/Microsoft/LightGBM')
