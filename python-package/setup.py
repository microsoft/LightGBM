# coding: utf-8
# pylint: disable=invalid-name, exec-used
"""Setup lightgbm package."""
from __future__ import absolute_import

import os
import sys
import distutils
from distutils import dir_util
from distutils import file_util
from setuptools import find_packages, setup

if not os.path.isfile("_IS_FULL_PACKAGE.txt"):
    distutils.dir_util.copy_tree("../include", "./include")
    distutils.dir_util.copy_tree("../src", "./src")
    distutils.dir_util.copy_tree("../compute", "./compute")
    distutils.file_util.copy_file("../CMakeLists.txt", ".")
    file_flag = open("_IS_FULL_PACKAGE.txt", 'w')
    file_flag.close()

if not os.path.exists("build"):
    os.makedirs("build")
os.chdir("build")

use_mingw = False
use_gpu = False
cmake_cmd = "cmake"
build_cmd = "make"

if os.name == "nt":
    if use_mingw:
        cmake_cmd = cmake_cmd + " -G \"MinGW Makefiles\" "
        build_cmd = "mingw32-make.exe"
    else:
        cmake_cmd = cmake_cmd + " -DCMAKE_GENERATOR_PLATFORM=x64 "
        build_cmd = "cmake --build . --target _lightgbm  --config Release"
if use_gpu:
    cmake_cmd = cmake_cmd + " -DUSE_GPU=1 "
print("Start to build libarary.")
os.system(cmake_cmd + " ..")
os.system(build_cmd)

os.chdir("..")
sys.path.insert(0, '.')

CURRENT_DIR = os.path.dirname(__file__)

libpath_py = os.path.join(CURRENT_DIR, 'lightgbm/libpath.py')
libpath = {'__file__': libpath_py}
exec(compile(open(libpath_py, "rb").read(), libpath_py, 'exec'), libpath, libpath)

LIB_PATH = [os.path.relpath(path, CURRENT_DIR) for path in libpath['find_lib_path']()]
print("Install lib_lightgbm from: %s" % LIB_PATH)

setup(name='lightgbm',
      version=0.2,
      description="LightGBM Python Package",
      install_requires=[
          'numpy',
          'scipy',
          'scikit-learn'
      ],
      maintainer='Guolin Ke',
      maintainer_email='guolin.ke@microsoft.com',
      zip_safe=False,
      packages=find_packages(),
      include_package_data=True,
      data_files=[('lightgbm', LIB_PATH)],
      url='https://github.com/Microsoft/LightGBM')
