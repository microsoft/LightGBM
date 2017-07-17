# coding: utf-8
# pylint: disable=invalid-name, exec-used, C0111
"""Setup lightgbm package."""
from __future__ import absolute_import

import distutils
import os
import shutil
import struct
import sys

from setuptools import find_packages, setup
from setuptools.command.install import install
from setuptools.command.install_lib import install_lib
from setuptools.command.sdist import sdist


def find_lib():
    CURRENT_DIR = os.path.dirname(__file__)
    libpath_py = os.path.join(CURRENT_DIR, 'lightgbm/libpath.py')
    libpath = {'__file__': libpath_py}
    exec(compile(open(libpath_py, "rb").read(), libpath_py, 'exec'), libpath, libpath)

    LIB_PATH = [os.path.relpath(path, CURRENT_DIR) for path in libpath['find_lib_path']()]
    print("Install lib_lightgbm from: %s" % LIB_PATH)
    return LIB_PATH


def copy_files(use_gpu=False):

    def copy_files_helper(folder_name):
        src = os.path.join('..', folder_name)
        if os.path.exists(src):
            dst = os.path.join('./lightgbm', folder_name)
            shutil.rmtree(dst, ignore_errors=True)
            distutils.dir_util.copy_tree(src, dst)
        else:
            raise Exception('Cannot copy {} folder'.format(src))

    if not os.path.isfile('./_IS_SOURCE_PACKAGE.txt'):
        copy_files_helper('include')
        copy_files_helper('src')
        if use_gpu:
            copy_files_helper('compute')
        distutils.file_util.copy_file("../CMakeLists.txt", "./lightgbm/")
        distutils.file_util.copy_file("../LICENSE", "./")


def clear_path(path):
    contents = os.listdir(path)
    for file in contents:
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
        else:
            shutil.rmtree(file_path)


def compile_cpp(use_mingw=False, use_gpu=False):

    if os.path.exists("build_cpp"):
        shutil.rmtree("build_cpp")
    os.makedirs("build_cpp")
    os.chdir("build_cpp")

    cmake_cmd = "cmake "
    build_cmd = "make _lightgbm"
    if use_gpu:
        cmake_cmd += " -DUSE_GPU=ON "
    if os.name == "nt":
        if use_mingw:
            cmake_cmd += " -G \"MinGW Makefiles\" "
            os.system(cmake_cmd + " ../lightgbm/")
            build_cmd = "mingw32-make.exe _lightgbm"
        else:
            vs_versions = ["Visual Studio 15 2017 Win64", "Visual Studio 14 2015 Win64", "Visual Studio 12 2013 Win64"]
            try_vs = 1
            for vs in vs_versions:
                tmp_cmake_cmd = "%s -G \"%s\"" % (cmake_cmd, vs)
                try_vs = os.system(tmp_cmake_cmd + " ../lightgbm/")
                if try_vs == 0:
                    cmake_cmd = tmp_cmake_cmd
                    break
                else:
                    clear_path("./")
            if try_vs != 0:
                raise Exception('Please install Visual Studio or MS Build first')

            build_cmd = "cmake --build . --target _lightgbm  --config Release"
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
            copy_files(use_gpu=self.gpu)
            compile_cpp(use_mingw=self.mingw, use_gpu=self.gpu)
        self.distribution.data_files = [('lightgbm', find_lib())]
        install.run(self)


class CustomSdist(sdist):

    def run(self):
        copy_files(use_gpu=True)
        open("./_IS_SOURCE_PACKAGE.txt", 'w').close()
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
          license='The MIT License (Microsoft)',
          url='https://github.com/Microsoft/LightGBM')
