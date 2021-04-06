# coding: utf-8
"""Setup lightgbm package."""
import logging
import os
import struct
import subprocess
import sys
from distutils.dir_util import copy_tree, create_tree, remove_tree
from distutils.file_util import copy_file
from platform import system

from setuptools import find_packages, setup
from setuptools.command.install import install
from setuptools.command.install_lib import install_lib
from setuptools.command.sdist import sdist
from wheel.bdist_wheel import bdist_wheel

LIGHTGBM_OPTIONS = [
    ('mingw', 'm', 'Compile with MinGW'),
    ('integrated-opencl', None, 'Compile integrated OpenCL version'),
    ('gpu', 'g', 'Compile GPU version'),
    ('cuda', None, 'Compile CUDA version'),
    ('mpi', None, 'Compile MPI version'),
    ('nomp', None, 'Compile version without OpenMP support'),
    ('hdfs', 'h', 'Compile HDFS version'),
    ('bit32', None, 'Compile 32-bit version'),
    ('precompile', 'p', 'Use precompiled library'),
    ('boost-root=', None, 'Boost preferred installation prefix'),
    ('boost-dir=', None, 'Directory with Boost package configuration file'),
    ('boost-include-dir=', None, 'Directory containing Boost headers'),
    ('boost-librarydir=', None, 'Preferred Boost library directory'),
    ('opencl-include-dir=', None, 'OpenCL include directory'),
    ('opencl-library=', None, 'Path to OpenCL library')
]


def find_lib():
    libpath_py = os.path.join(CURRENT_DIR, 'lightgbm', 'libpath.py')
    libpath = {'__file__': libpath_py}
    exec(compile(open(libpath_py, "rb").read(), libpath_py, 'exec'), libpath, libpath)

    LIB_PATH = [os.path.relpath(path, CURRENT_DIR) for path in libpath['find_lib_path']()]
    logger.info(f"Installing lib_lightgbm from: {LIB_PATH}")
    return LIB_PATH


def copy_files(integrated_opencl=False, use_gpu=False):

    def copy_files_helper(folder_name):
        src = os.path.join(CURRENT_DIR, os.path.pardir, folder_name)
        if os.path.exists(src):
            dst = os.path.join(CURRENT_DIR, 'compile', folder_name)
            if os.path.exists(dst):
                if os.path.isdir:
                    # see https://github.com/pypa/distutils/pull/21
                    remove_tree(dst)
                else:
                    os.remove(dst)
            create_tree(src, dst, verbose=0)
            copy_tree(src, dst, verbose=0)
        else:
            raise Exception(f'Cannot copy {src} folder')

    if not os.path.isfile(os.path.join(CURRENT_DIR, '_IS_SOURCE_PACKAGE.txt')):
        copy_files_helper('include')
        copy_files_helper('src')
        for submodule in os.listdir(os.path.join(CURRENT_DIR, os.path.pardir, 'external_libs')):
            if submodule == 'compute' and not use_gpu:
                continue
            copy_files_helper(os.path.join('external_libs', submodule))
        if not os.path.exists(os.path.join(CURRENT_DIR, "compile", "windows")):
            os.makedirs(os.path.join(CURRENT_DIR, "compile", "windows"))
        copy_file(os.path.join(CURRENT_DIR, os.path.pardir, "windows", "LightGBM.sln"),
                  os.path.join(CURRENT_DIR, "compile", "windows", "LightGBM.sln"),
                  verbose=0)
        copy_file(os.path.join(CURRENT_DIR, os.path.pardir, "windows", "LightGBM.vcxproj"),
                  os.path.join(CURRENT_DIR, "compile", "windows", "LightGBM.vcxproj"),
                  verbose=0)
        copy_file(os.path.join(CURRENT_DIR, os.path.pardir, "LICENSE"),
                  os.path.join(CURRENT_DIR, "LICENSE"),
                  verbose=0)
        copy_file(os.path.join(CURRENT_DIR, os.path.pardir, "CMakeLists.txt"),
                  os.path.join(CURRENT_DIR, "compile", "CMakeLists.txt"),
                  verbose=0)
        if integrated_opencl:
            if not os.path.exists(os.path.join(CURRENT_DIR, "compile", "cmake")):
                os.makedirs(os.path.join(CURRENT_DIR, "compile", "cmake"))
            copy_file(os.path.join(CURRENT_DIR, os.path.pardir, "cmake", "IntegratedOpenCL.cmake"),
                      os.path.join(CURRENT_DIR, "compile", "cmake", "IntegratedOpenCL.cmake"),
                      verbose=0)


def clear_path(path):
    if os.path.isdir(path):
        contents = os.listdir(path)
        for file_name in contents:
            file_path = os.path.join(path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
            else:
                remove_tree(file_path)


def silent_call(cmd, raise_error=False, error_msg=''):
    try:
        with open(LOG_PATH, "ab") as log:
            subprocess.check_call(cmd, stderr=log, stdout=log)
        return 0
    except Exception as err:
        if raise_error:
            raise Exception("\n".join((error_msg, LOG_NOTICE)))
        return 1


def compile_cpp(use_mingw=False, use_gpu=False, use_cuda=False, use_mpi=False,
                use_hdfs=False, boost_root=None, boost_dir=None,
                boost_include_dir=None, boost_librarydir=None,
                opencl_include_dir=None, opencl_library=None,
                nomp=False, bit32=False, integrated_opencl=False):

    if os.path.exists(os.path.join(CURRENT_DIR, "build_cpp")):
        remove_tree(os.path.join(CURRENT_DIR, "build_cpp"))
    os.makedirs(os.path.join(CURRENT_DIR, "build_cpp"))
    os.chdir(os.path.join(CURRENT_DIR, "build_cpp"))

    logger.info("Starting to compile the library.")

    cmake_cmd = ["cmake", "../compile/"]
    if integrated_opencl:
        use_gpu = False
        cmake_cmd.append("-D__INTEGRATE_OPENCL=ON")
    if use_gpu:
        cmake_cmd.append("-DUSE_GPU=ON")
        if boost_root:
            cmake_cmd.append(f"-DBOOST_ROOT={boost_root}")
        if boost_dir:
            cmake_cmd.append(f"-DBoost_DIR={boost_dir}")
        if boost_include_dir:
            cmake_cmd.append(f"-DBoost_INCLUDE_DIR={boost_include_dir}")
        if boost_librarydir:
            cmake_cmd.append(f"-DBOOST_LIBRARYDIR={boost_librarydir}")
        if opencl_include_dir:
            cmake_cmd.append(f"-DOpenCL_INCLUDE_DIR={opencl_include_dir}")
        if opencl_library:
            cmake_cmd.append(f"-DOpenCL_LIBRARY={opencl_library}")
    elif use_cuda:
        cmake_cmd.append("-DUSE_CUDA=ON")
    if use_mpi:
        cmake_cmd.append("-DUSE_MPI=ON")
    if nomp:
        cmake_cmd.append("-DUSE_OPENMP=OFF")
    if use_hdfs:
        cmake_cmd.append("-DUSE_HDFS=ON")

    if system() in {'Windows', 'Microsoft'}:
        if use_mingw:
            if use_mpi:
                raise Exception('MPI version cannot be compiled by MinGW due to the miss of MPI library in it')
            logger.info("Starting to compile with CMake and MinGW.")
            silent_call(cmake_cmd + ["-G", "MinGW Makefiles"], raise_error=True,
                        error_msg='Please install CMake and all required dependencies first')
            silent_call(["mingw32-make.exe", "_lightgbm"], raise_error=True,
                        error_msg='Please install MinGW first')
        else:
            status = 1
            lib_path = os.path.join(CURRENT_DIR, "compile", "windows", "x64", "DLL", "lib_lightgbm.dll")
            if not any((use_gpu, use_cuda, use_mpi, use_hdfs, nomp, bit32, integrated_opencl)):
                logger.info("Starting to compile with MSBuild from existing solution file.")
                platform_toolsets = ("v142", "v141", "v140")
                for pt in platform_toolsets:
                    status = silent_call(["MSBuild",
                                          os.path.join(CURRENT_DIR, "compile", "windows", "LightGBM.sln"),
                                          "/p:Configuration=DLL",
                                          "/p:Platform=x64",
                                          f"/p:PlatformToolset={pt}"])
                    if status == 0 and os.path.exists(lib_path):
                        break
                    else:
                        clear_path(os.path.join(CURRENT_DIR, "compile", "windows", "x64"))
                if status != 0 or not os.path.exists(lib_path):
                    logger.warning("Compilation with MSBuild from existing solution file failed.")
            if status != 0 or not os.path.exists(lib_path):
                arch = "Win32" if bit32 else "x64"
                vs_versions = ("Visual Studio 16 2019", "Visual Studio 15 2017", "Visual Studio 14 2015")
                for vs in vs_versions:
                    logger.info(f"Starting to compile with {vs} ({arch}).")
                    status = silent_call(cmake_cmd + ["-G", vs, "-A", arch])
                    if status == 0:
                        break
                    else:
                        clear_path(os.path.join(CURRENT_DIR, "build_cpp"))
                if status != 0:
                    raise Exception("\n".join(('Please install Visual Studio or MS Build and all required dependencies first',
                                    LOG_NOTICE)))
                silent_call(["cmake", "--build", ".", "--target", "_lightgbm", "--config", "Release"], raise_error=True,
                            error_msg='Please install CMake first')
    else:  # Linux, Darwin (macOS), etc.
        logger.info("Starting to compile with CMake.")
        silent_call(cmake_cmd, raise_error=True, error_msg='Please install CMake and all required dependencies first')
        silent_call(["make", "_lightgbm", "-j4"], raise_error=True,
                    error_msg='An error has occurred while building lightgbm library file')
    os.chdir(CURRENT_DIR)


class CustomInstallLib(install_lib):

    def install(self):
        outfiles = install_lib.install(self)
        src = find_lib()[0]
        dst = os.path.join(self.install_dir, 'lightgbm')
        dst, _ = self.copy_file(src, dst)
        outfiles.append(dst)
        return outfiles


class CustomInstall(install):

    user_options = install.user_options + LIGHTGBM_OPTIONS

    def initialize_options(self):
        install.initialize_options(self)
        self.mingw = 0
        self.integrated_opencl = 0
        self.gpu = 0
        self.cuda = 0
        self.boost_root = None
        self.boost_dir = None
        self.boost_include_dir = None
        self.boost_librarydir = None
        self.opencl_include_dir = None
        self.opencl_library = None
        self.mpi = 0
        self.hdfs = 0
        self.precompile = 0
        self.nomp = 0
        self.bit32 = 0

    def run(self):
        if (8 * struct.calcsize("P")) != 64:
            if self.bit32:
                logger.warning("You're installing 32-bit version. "
                               "This version is slow and untested, so use it on your own risk.")
            else:
                raise Exception("Cannot install LightGBM in 32-bit Python, "
                                "please use 64-bit Python instead.")
        open(LOG_PATH, 'wb').close()
        if not self.precompile:
            copy_files(integrated_opencl=self.integrated_opencl, use_gpu=self.gpu)
            compile_cpp(use_mingw=self.mingw, use_gpu=self.gpu, use_cuda=self.cuda, use_mpi=self.mpi,
                        use_hdfs=self.hdfs, boost_root=self.boost_root, boost_dir=self.boost_dir,
                        boost_include_dir=self.boost_include_dir, boost_librarydir=self.boost_librarydir,
                        opencl_include_dir=self.opencl_include_dir, opencl_library=self.opencl_library,
                        nomp=self.nomp, bit32=self.bit32, integrated_opencl=self.integrated_opencl)
        install.run(self)
        if os.path.isfile(LOG_PATH):
            os.remove(LOG_PATH)


class CustomBdistWheel(bdist_wheel):

    user_options = bdist_wheel.user_options + LIGHTGBM_OPTIONS

    def initialize_options(self):
        bdist_wheel.initialize_options(self)
        self.mingw = 0
        self.integrated_opencl = 0
        self.gpu = 0
        self.cuda = 0
        self.boost_root = None
        self.boost_dir = None
        self.boost_include_dir = None
        self.boost_librarydir = None
        self.opencl_include_dir = None
        self.opencl_library = None
        self.mpi = 0
        self.hdfs = 0
        self.precompile = 0
        self.nomp = 0
        self.bit32 = 0

    def finalize_options(self):
        bdist_wheel.finalize_options(self)

        install = self.reinitialize_command('install')

        install.mingw = self.mingw
        install.integrated_opencl = self.integrated_opencl
        install.gpu = self.gpu
        install.cuda = self.cuda
        install.boost_root = self.boost_root
        install.boost_dir = self.boost_dir
        install.boost_include_dir = self.boost_include_dir
        install.boost_librarydir = self.boost_librarydir
        install.opencl_include_dir = self.opencl_include_dir
        install.opencl_library = self.opencl_library
        install.mpi = self.mpi
        install.hdfs = self.hdfs
        install.precompile = self.precompile
        install.nomp = self.nomp
        install.bit32 = self.bit32


class CustomSdist(sdist):

    def run(self):
        copy_files(integrated_opencl=True, use_gpu=True)
        open(os.path.join(CURRENT_DIR, '_IS_SOURCE_PACKAGE.txt'), 'w').close()
        if os.path.exists(os.path.join(CURRENT_DIR, 'lightgbm', 'Release')):
            remove_tree(os.path.join(CURRENT_DIR, 'lightgbm', 'Release'))
        if os.path.exists(os.path.join(CURRENT_DIR, 'lightgbm', 'windows', 'x64')):
            remove_tree(os.path.join(CURRENT_DIR, 'lightgbm', 'windows', 'x64'))
        if os.path.isfile(os.path.join(CURRENT_DIR, 'lightgbm', 'lib_lightgbm.so')):
            os.remove(os.path.join(CURRENT_DIR, 'lightgbm', 'lib_lightgbm.so'))
        sdist.run(self)
        if os.path.isfile(os.path.join(CURRENT_DIR, '_IS_SOURCE_PACKAGE.txt')):
            os.remove(os.path.join(CURRENT_DIR, '_IS_SOURCE_PACKAGE.txt'))


if __name__ == "__main__":
    CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
    LOG_PATH = os.path.join(os.path.expanduser('~'), 'LightGBM_compilation.log')
    LOG_NOTICE = f"The full version of error log was saved into {LOG_PATH}"
    if os.path.isfile(os.path.join(CURRENT_DIR, os.path.pardir, 'VERSION.txt')):
        copy_file(os.path.join(CURRENT_DIR, os.path.pardir, 'VERSION.txt'),
                  os.path.join(CURRENT_DIR, 'lightgbm', 'VERSION.txt'),
                  verbose=0)  # type:ignore
    version = open(os.path.join(CURRENT_DIR, 'lightgbm', 'VERSION.txt'), encoding='utf-8').read().strip()
    readme = open(os.path.join(CURRENT_DIR, 'README.rst'), encoding='utf-8').read()

    sys.path.insert(0, CURRENT_DIR)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('LightGBM')

    setup(name='lightgbm',
          version=version,
          description='LightGBM Python Package',
          long_description=readme,
          install_requires=[
              'wheel',
              'numpy',
              'scipy',
              'scikit-learn!=0.22.0'
          ],
          extras_require={
              'dask': [
                  'dask[array]>=2.0.0',
                  'dask[dataframe]>=2.0.0',
                  'dask[distributed]>=2.0.0',
                  'pandas',
              ],
          },
          maintainer='Guolin Ke',
          maintainer_email='guolin.ke@microsoft.com',
          zip_safe=False,
          cmdclass={
              'install': CustomInstall,
              'install_lib': CustomInstallLib,
              'bdist_wheel': CustomBdistWheel,
              'sdist': CustomSdist,
          },
          packages=find_packages(),
          include_package_data=True,
          license='The MIT License (Microsoft)',
          url='https://github.com/microsoft/LightGBM',
          classifiers=['Development Status :: 5 - Production/Stable',
                       'Intended Audience :: Science/Research',
                       'License :: OSI Approved :: MIT License',
                       'Natural Language :: English',
                       'Operating System :: MacOS',
                       'Operating System :: Microsoft :: Windows',
                       'Operating System :: POSIX',
                       'Operating System :: Unix',
                       'Programming Language :: Python :: 3',
                       'Programming Language :: Python :: 3.6',
                       'Programming Language :: Python :: 3.7',
                       'Programming Language :: Python :: 3.8',
                       'Programming Language :: Python :: 3.9',
                       'Topic :: Scientific/Engineering :: Artificial Intelligence'])
