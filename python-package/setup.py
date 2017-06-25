# coding: utf-8
# pylint: disable=invalid-name, exec-used
"""Setup lightgbm package."""
from __future__ import absolute_import

import struct
import os
import sys
import getopt
import distutils
import shutil
from distutils import dir_util
from distutils import file_util
from setuptools import find_packages, setup

if __name__ == "__main__":
    build_sdist = sys.argv[1] == 'sdist'
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
        sys.argv = sys.argv[0:2]
    except getopt.GetoptError as err:
        pass
    if not use_precompile or build_sdist:
        if not os.path.isfile('./_IS_SOURCE_PACKAGE.txt'):
            if os.path.exists("../include"):
                if os.path.exists("./lightgbm/include"):
                    shutil.rmtree('./lightgbm/include')
                distutils.dir_util.copy_tree("../include", "./lightgbm/include")
            else:
                raise Exception('Cannot copy ../include folder')
            if os.path.exists("../src"):
                if os.path.exists("./lightgbm/src"):
                    shutil.rmtree('./lightgbm/src')
                distutils.dir_util.copy_tree("../src", "./lightgbm/src")
            else:
                raise Exception('Cannot copy ../src folder')
            if use_gpu:
                if os.path.exists("../compute"):
                    if os.path.exists("./lightgbm/compute"):
                        shutil.rmtree('./lightgbm/compute')
                    distutils.dir_util.copy_tree("../compute", "./lightgbm/compute")
                else:
                    raise Exception('Cannot copy ../compute folder')
            distutils.file_util.copy_file("../CMakeLists.txt", "./lightgbm/")
            distutils.file_util.copy_file("../VERSION.txt", "./lightgbm/")
            if build_sdist:
                file_flag = open("./_IS_SOURCE_PACKAGE.txt", 'w')
                file_flag.close()

        if not os.path.exists("build"):
            os.makedirs("build")
        os.chdir("build")

        cmake_cmd = "cmake "
        build_cmd = "make _lightgbm"

        if os.name == "nt":
            if use_mingw:
                cmake_cmd = cmake_cmd + " -G \"MinGW Makefiles\" "
                build_cmd = "mingw32-make.exe _lightgbm"
            else:
                cmake_cmd = cmake_cmd + " -DCMAKE_GENERATOR_PLATFORM=x64 "
                build_cmd = "cmake --build . --target _lightgbm  --config Release"
        if use_gpu:
            cmake_cmd = cmake_cmd + " -DUSE_GPU=ON "
        if not build_sdist:
            print("Start to compile libarary.")
            os.system(cmake_cmd + " ../lightgbm/")
            os.system(build_cmd)
        os.chdir("..")

    data_files = []

    if build_sdist:
        print("remove library when building source distribution")
        if os.path.exists("./lightgbm/Release/"):
            shutil.rmtree('./lightgbm/Release/')
        if os.path.isfile('./lightgbm/lib_lightgbm.so'):
            os.remove('./lightgbm/lib_lightgbm.so')
    else:
        sys.path.insert(0, '.')
        CURRENT_DIR = os.path.dirname(__file__)
        libpath_py = os.path.join(CURRENT_DIR, 'lightgbm/libpath.py')
        libpath = {'__file__': libpath_py}
        exec(compile(open(libpath_py, "rb").read(), libpath_py, 'exec'), libpath, libpath)

        LIB_PATH = [os.path.relpath(path, CURRENT_DIR) for path in libpath['find_lib_path']()]
        print("Install lib_lightgbm from: %s" % LIB_PATH)
        data_files = [('lightgbm', LIB_PATH)]
    version = '2.0.1'
    if os.path.isfile('./lightgbm/VERSION.txt'):
        version = open('./lightgbm/VERSION.txt').read().strip()
    elif os.path.isfile('../VERSION.txt'):
        version = open('../VERSION.txt').read().strip()
    setup(name='lightgbm',
          version=version,
          description='LightGBM Python Package',
          install_requires=[
              'wheel',
              'numpy',
              'scipy',
              'scikit-learn'
          ],
          maintainer='Guolin Ke',
          maintainer_email='guolin.ke@microsoft.com',
          zip_safe=False,
          packages=find_packages(),
          include_package_data=True,
          data_files=data_files,
          license='The MIT License(https://github.com/Microsoft/LightGBM/blob/master/LICENSE)',
          url='https://github.com/Microsoft/LightGBM')
    if build_sdist and os.path.isfile('./_IS_SOURCE_PACKAGE.txt'):
        os.remove('./_IS_SOURCE_PACKAGE.txt')
