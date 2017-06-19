# coding: utf-8
# pylint: disable=invalid-name, exec-used
"""Setup lightgbm package."""
from __future__ import absolute_import

import struct
import os
import sys
import getopt
import distutils
from distutils import dir_util
from distutils import file_util
from setuptools import find_packages, setup

if __name__ == "__main__":
    if (8 * struct.calcsize("P")) != 64:
        raise Exception('Cannot install LightGBM in 32-bit python, please use 64-bit python instead.')
    use_gpu = False
    use_mingw = False
    use_precompile = False
    try:
        opts, args = getopt.getopt(sys.argv[2:], 'mgp', ['mingw', 'gpu', 'precompile'])
        for opt, arg in opts:
            if opt in ('-m', '--mingw'):
                use_mingw = True
            elif opt in ('-g', '--gpu'):
                use_gpu = True
            elif opt in ('-p', '--precompile'):
                use_precompile = True
    except getopt.GetoptError as err:
        pass
    sys.argv = sys.argv[0:2]
    if not use_precompile:
        if not os.path.isfile("_IS_FULL_PACKAGE.txt"):
            if os.path.exists("../include"):
                distutils.dir_util.copy_tree("../include", "./lightgbm/include")
            else:
                raise Exception('Cannot copy ../include folder')
            if os.path.exists("../src"):
                distutils.dir_util.copy_tree("../src", "./lightgbm/src")
            else:
                raise Exception('Cannot copy ../src folder')
            if use_gpu:
                if os.path.exists("../compute"):
                    distutils.dir_util.copy_tree("../compute", "./lightgbm/compute")
                else:
                    raise Exception('Cannot copy ../compute folder')
            distutils.file_util.copy_file("../CMakeLists.txt", "./lightgbm/")
            file_flag = open("_IS_FULL_PACKAGE.txt", 'w')
            file_flag.close()

        if not os.path.exists("build"):
            os.makedirs("build")
        os.chdir("build")

        cmake_cmd = "cmake -DBUILD_EXE=OFF -DBUILD_LIB=ON "
        build_cmd = "make"

        if os.name == "nt":
            if use_mingw:
                cmake_cmd = cmake_cmd + " -G \"MinGW Makefiles\" "
                build_cmd = "mingw32-make.exe"
            else:
                cmake_cmd = cmake_cmd + " -DCMAKE_GENERATOR_PLATFORM=x64 "
                build_cmd = "cmake --build . --target _lightgbm  --config Release"
        if use_gpu:
            cmake_cmd = cmake_cmd + " -DUSE_GPU=ON "
        print("Start to build libarary.")
        os.system(cmake_cmd + " ../lightgbm/")
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
          version='0.2a0',
          description='LightGBM Python Package',
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
          license='The MIT License(https://github.com/Microsoft/LightGBM/blob/master/LICENSE)',
          url='https://github.com/Microsoft/LightGBM')
