# coding: utf-8
# pylint: disable=invalid-name, exec-used, C0111
"""Setup lightgbm package."""
from __future__ import absolute_import

import distutils
import logging
import os
import shutil
import struct
import subprocess
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
    logging.info("Installing lib_lightgbm from: %s" % LIB_PATH)
    return LIB_PATH


def copy_files(use_gpu=False):

    def copy_files_helper(folder_name):
        src = os.path.join('..', folder_name)
        if os.path.exists(src):
            dst = os.path.join('./compile', folder_name)
            shutil.rmtree(dst, ignore_errors=True)
            distutils.dir_util.copy_tree(src, dst)
        else:
            raise Exception('Cannot copy {} folder'.format(src))

    if not os.path.isfile('./_IS_SOURCE_PACKAGE.txt'):
        copy_files_helper('include')
        copy_files_helper('src')
        if not os.path.exists("./compile/windows/"):
            os.makedirs("./compile/windows/")
        distutils.file_util.copy_file("../windows/LightGBM.sln", "./compile/windows/LightGBM.sln")
        distutils.file_util.copy_file("../windows/LightGBM.vcxproj", "./compile/windows/LightGBM.vcxproj")
        if use_gpu:
            copy_files_helper('compute')
        distutils.file_util.copy_file("../CMakeLists.txt", "./compile/")
        distutils.file_util.copy_file("../LICENSE", "./")


def clear_path(path):
    if os.path.isdir(path):
        contents = os.listdir(path)
        for file_name in contents:
            file_path = os.path.join(path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
            else:
                shutil.rmtree(file_path)


def silent_call(cmd, raise_error=False, error_msg=''):
    try:
        with open(os.devnull, "w") as shut_up:
            subprocess.check_output(cmd, stderr=shut_up)
            return 0
    except Exception:
        if raise_error:
            raise Exception(error_msg)
        return 1


def compile_cpp(use_mingw=False, use_gpu=False):

    if os.path.exists("build_cpp"):
        shutil.rmtree("build_cpp")
    os.makedirs("build_cpp")
    os.chdir("build_cpp")

    logger.info("Starting to compile the library.")

    cmake_cmd = ["cmake", "../compile/"]
    if use_gpu:
        cmake_cmd.append("-DUSE_GPU=ON")
    if os.name == "nt":
        if use_mingw:
            logger.info("Starting to compile with CMake and MinGW.")
            silent_call(cmake_cmd + ["-G", "MinGW Makefiles"], raise_error=True,
                        error_msg='Please install CMake first')
            silent_call(["mingw32-make.exe", "_lightgbm"], raise_error=True,
                        error_msg='Please install MinGW first')
        else:
            status = 1
            lib_path = "../compile/windows/x64/DLL/lib_lightgbm.dll"
            if not use_gpu:
                logger.info("Starting to compile with MSBuild from existing solution file.")
                platform_toolsets = ("v141", "v140")
                for pt in platform_toolsets:
                    status = silent_call(["MSBuild", "../compile/windows/LightGBM.sln",
                                          "/p:Configuration=DLL",
                                          "/p:Platform=x64",
                                          "/p:PlatformToolset={0}".format(pt)])
                    if status == 0 and os.path.exists(lib_path):
                        break
                    else:
                        clear_path("../compile/windows/x64")
                if status != 0 or not os.path.exists(lib_path):
                    logger.warning("Compilation with MSBuild from existing solution file failed.")
            if status != 0 or not os.path.exists(lib_path):
                vs_versions = ("Visual Studio 15 2017 Win64", "Visual Studio 14 2015 Win64")
                for vs in vs_versions:
                    logger.info("Starting to compile with %s." % vs)
                    status = silent_call(cmake_cmd + ["-G", vs])
                    if status == 0:
                        break
                    else:
                        clear_path("./")
                if status != 0:
                    raise Exception('Please install Visual Studio or MS Build first')
                silent_call(["cmake", "--build", ".", "--target", "_lightgbm", "--config", "Release"], raise_error=True,
                            error_msg='Please install CMake first')
    else:  # Linux, Darwin (OS X), etc.
        logger.info("Starting to compile with CMake.")
        silent_call(cmake_cmd, raise_error=True, error_msg='Please install CMake first')
        silent_call(["make", "_lightgbm"], raise_error=True,
                    error_msg='An error has occurred while building lightgbm library file')
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
        ('precompile', 'p', 'use precompiled library')
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
        install.run(self)


class CustomSdist(sdist):

    def run(self):
        copy_files(use_gpu=True)
        open("./_IS_SOURCE_PACKAGE.txt", 'w').close()
        if os.path.exists("./lightgbm/Release/"):
            shutil.rmtree('./lightgbm/Release/')
        if os.path.exists("./lightgbm/windows/x64/"):
            shutil.rmtree('./lightgbm/windows/x64/')
        if os.path.isfile('./lightgbm/lib_lightgbm.so'):
            os.remove('./lightgbm/lib_lightgbm.so')
        sdist.run(self)
        if os.path.isfile('./_IS_SOURCE_PACKAGE.txt'):
            os.remove('./_IS_SOURCE_PACKAGE.txt')


if __name__ == "__main__":
    if (8 * struct.calcsize("P")) != 64:
        raise Exception('Cannot install LightGBM in 32-bit python, please use 64-bit python instead.')

    dir_path = os.path.dirname(os.path.realpath(__file__))
    if os.path.isfile(os.path.join('..', 'VERSION.txt')):
        distutils.file_util.copy_file(os.path.join('..', 'VERSION.txt'),
                                      os.path.join('.', 'lightgbm'))
    version = open(os.path.join(dir_path, 'lightgbm', 'VERSION.txt')).read().strip()

    sys.path.insert(0, '.')

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('LightGBM')

    setup(name='lightgbm',
          version=version,
          description='LightGBM Python Package',
          long_description=open('README.rst').read(),
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
          license='The MIT License (Microsoft)',
          url='https://github.com/Microsoft/LightGBM',
          classifiers=['Development Status :: 5 - Production/Stable',
                       'Intended Audience :: Science/Research',
                       'License :: OSI Approved :: MIT License',
                       'Natural Language :: English',
                       'Operating System :: MacOS',
                       'Operating System :: Microsoft :: Windows',
                       'Operating System :: POSIX',
                       'Operating System :: Unix',
                       'Programming Language :: Python :: 2',
                       'Programming Language :: Python :: 2.7',
                       'Programming Language :: Python :: 3',
                       'Programming Language :: Python :: 3.4',
                       'Programming Language :: Python :: 3.5',
                       'Programming Language :: Python :: 3.6',
                       'Topic :: Scientific/Engineering :: Artificial Intelligence'])
