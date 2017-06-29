# coding: utf-8
# pylint: disable=invalid-name, exec-used
"""Setup lightgbm package."""
from __future__ import absolute_import

import distutils
import os
import shutil
import struct
import sys
from distutils.command.install import install
from distutils.command.install_lib import install_lib
from distutils.command.sdist import sdist

from setuptools import find_packages, setup


def find_lib():
    CURRENT_DIR = os.path.dirname(__file__)
    libpath_py = os.path.join(CURRENT_DIR, 'lightgbm/libpath.py')
    libpath = {'__file__': libpath_py}
    exec(compile(open(libpath_py, "rb").read(), libpath_py, 'exec'), libpath, libpath)

    LIB_PATH = [os.path.relpath(path, CURRENT_DIR) for path in libpath['find_lib_path']()]
    print("Install lib_lightgbm from: %s" % LIB_PATH)
    return LIB_PATH


def compile_cpp(use_mingw=False, use_gpu=False, build_sdist=False):

    def copy_files(folder_name):
        src = os.path.join('..', folder_name)
        if os.path.exists(src):
            dst = os.path.join('./lightgbm', folder_name)
            shutil.rmtree(dst, ignore_errors=True)
            distutils.dir_util.copy_tree(src, dst)
        else:
            raise Exception('Cannot copy {} folder'.format(src))

    if not os.path.isfile('./_IS_SOURCE_PACKAGE.txt'):
        copy_files('include')
        copy_files('src')
        if use_gpu:
            copy_files('compute')
        distutils.file_util.copy_file("../CMakeLists.txt", "./lightgbm/")

    if build_sdist:
        open("./_IS_SOURCE_PACKAGE.txt", 'w').close()
    else:
        if not os.path.exists("build"):
            os.makedirs("build")
        os.chdir("build")

        cmake_cmd = "cmake "
        build_cmd = "make _lightgbm"

        if os.name == "nt":
            if use_mingw:
                cmake_cmd += " -G \"MinGW Makefiles\" "
                build_cmd = "mingw32-make.exe _lightgbm"
            else:
                cmake_cmd += " -DCMAKE_GENERATOR_PLATFORM=x64 "
                build_cmd = "cmake --build . --target _lightgbm  --config Release"
        if use_gpu:
            cmake_cmd += " -DUSE_GPU=ON "
        print("Start to compile libarary.")
        os.system(cmake_cmd + " ../lightgbm/")
        os.system(build_cmd)
        os.chdir("..")


class CustomInstallLib(install_lib):

    def install(self):
        outfiles = install_lib.install(self)
        src = find_lib()[0]
        dst = os.path.join(self.install_dir, 'lightgbm')
        dst, _ = self.copy_file(src, dst)
        outfiles.append(dst)
        return outfiles

class CustomInstall(install):

    user_options = install.user_options + [
        ('mingw', 'm', 'compile with mingw'),
        ('gpu', 'g', 'compile gpu version'),
        ('precompile', 'p', 'use precompile library')
    ]

    def initialize_options(self):
        install.initialize_options(self)
        self.mingw = 0
        self.gpu = 0
        self.precompile = 0

    def run(self):
        if not self.precompile:
            compile_cpp(use_mingw=self.mingw, use_gpu=self.gpu)
        self.distribution.data_files = [('lightgbm', find_lib())]
        install.run(self)


class CustomSdist(sdist):

    user_options = sdist.user_options + [
        ('gpu', 'g', 'compile gpu version')
    ]

    def initialize_options(self):
        sdist.initialize_options(self)
        self.gpu = 0

    def run(self):
        compile_cpp(use_gpu=self.gpu, build_sdist=True)
        if os.path.exists("./lightgbm/Release/"):
            shutil.rmtree('./lightgbm/Release/')
        if os.path.isfile('./lightgbm/lib_lightgbm.so'):
            os.remove('./lightgbm/lib_lightgbm.so')
        sdist.run(self)
        if os.path.isfile('./_IS_SOURCE_PACKAGE.txt'):
            os.remove('./_IS_SOURCE_PACKAGE.txt')

if __name__ == "__main__":
    if (8 * struct.calcsize("P")) != 64:
        raise Exception('Cannot install LightGBM in 32-bit python, please use 64-bit python instead.')
    if os.path.isfile('../VERSION.txt'):
        distutils.file_util.copy_file("../VERSION.txt", "./lightgbm/")
    version = '2.0.3'
    if os.path.isfile('./lightgbm/VERSION.txt'):
        with open('./lightgbm/VERSION.txt') as file_version:
            version = file_version.readline().strip()
    sys.path.insert(0, '.')
    data_files = []
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
          cmdclass={
              'install': CustomInstall,
              'install_lib': CustomInstallLib,
              'sdist': CustomSdist,
          },
          packages=find_packages(),
          include_package_data=True,
          data_files=data_files,
          license='The MIT License(https://github.com/Microsoft/LightGBM/blob/master/LICENSE)',
          url='https://github.com/Microsoft/LightGBM')
