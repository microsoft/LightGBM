# coding: utf-8
"""Setup lightgbm package."""
import logging
import struct
import subprocess
import sys
from os import chdir
from pathlib import Path
from platform import system
from shutil import copyfile, copytree, rmtree
from typing import List, Optional, Union

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


def find_lib() -> List[str]:
    libpath_py = CURRENT_DIR / 'lightgbm' / 'libpath.py'
    libpath = {'__file__': libpath_py}
    exec(compile(libpath_py.read_bytes(), libpath_py, 'exec'), libpath, libpath)

    LIB_PATH = libpath['find_lib_path']()
    logger.info(f"Installing lib_lightgbm from: {LIB_PATH}")
    return LIB_PATH


def copy_files(integrated_opencl: bool = False, use_gpu: bool = False) -> None:

    def copy_files_helper(folder_name: Union[str, Path]) -> None:
        src = CURRENT_DIR.parent / folder_name
        if src.is_dir():
            dst = CURRENT_DIR / 'compile' / folder_name
            if dst.is_dir():
                rmtree(dst)
            copytree(src, dst)
        else:
            raise Exception(f'Cannot copy {src} folder')

    if not IS_SOURCE_FLAG_PATH.is_file():
        copy_files_helper('include')
        copy_files_helper('src')
        for submodule in (CURRENT_DIR.parent / 'external_libs').iterdir():
            submodule_stem = submodule.stem
            if submodule_stem == 'compute' and not use_gpu:
                continue
            copy_files_helper(Path('external_libs') / submodule_stem)
        (CURRENT_DIR / "compile" / "windows").mkdir(parents=True, exist_ok=True)
        copyfile(CURRENT_DIR.parent / "windows" / "LightGBM.sln",
                 CURRENT_DIR / "compile" / "windows" / "LightGBM.sln")
        copyfile(CURRENT_DIR.parent / "windows" / "LightGBM.vcxproj",
                 CURRENT_DIR / "compile" / "windows" / "LightGBM.vcxproj")
        copyfile(CURRENT_DIR.parent / "LICENSE",
                 CURRENT_DIR / "LICENSE")
        copyfile(CURRENT_DIR.parent / "CMakeLists.txt",
                 CURRENT_DIR / "compile" / "CMakeLists.txt")
        if integrated_opencl:
            (CURRENT_DIR / "compile" / "cmake").mkdir(parents=True, exist_ok=True)
            copyfile(CURRENT_DIR.parent / "cmake" / "IntegratedOpenCL.cmake",
                     CURRENT_DIR / "compile" / "cmake" / "IntegratedOpenCL.cmake")


def clear_path(path: Path) -> None:
    if path.is_dir():
        for file_name in path.iterdir():
            if file_name.is_dir():
                rmtree(file_name)
            else:
                file_name.unlink()


def silent_call(cmd: List[str], raise_error: bool = False, error_msg: str = '') -> int:
    try:
        with open(LOG_PATH, "ab") as log:
            subprocess.check_call(cmd, stderr=log, stdout=log)
        return 0
    except Exception as err:
        if raise_error:
            raise Exception("\n".join((error_msg, LOG_NOTICE)))
        return 1


def compile_cpp(
    use_mingw: bool = False,
    use_gpu: bool = False,
    use_cuda: bool = False,
    use_mpi: bool = False,
    use_hdfs: bool = False,
    boost_root: Optional[str] = None,
    boost_dir: Optional[str] = None,
    boost_include_dir: Optional[str] = None,
    boost_librarydir: Optional[str] = None,
    opencl_include_dir: Optional[str] = None,
    opencl_library: Optional[str] = None,
    nomp: bool = False,
    bit32: bool = False,
    integrated_opencl: bool = False
) -> None:
    build_dir = CURRENT_DIR / "build_cpp"
    rmtree(build_dir, ignore_errors=True)
    build_dir.mkdir(parents=True)
    original_dir = Path.cwd()
    chdir(build_dir)

    logger.info("Starting to compile the library.")

    cmake_cmd = ["cmake", str(CURRENT_DIR / "compile")]
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
            silent_call(["mingw32-make.exe", "_lightgbm", f"-I{build_dir}", "-j4"], raise_error=True,
                        error_msg='Please install MinGW first')
        else:
            status = 1
            lib_path = CURRENT_DIR / "compile" / "windows" / "x64" / "DLL" / "lib_lightgbm.dll"
            if not any((use_gpu, use_cuda, use_mpi, use_hdfs, nomp, bit32, integrated_opencl)):
                logger.info("Starting to compile with MSBuild from existing solution file.")
                platform_toolsets = ("v142", "v141", "v140")
                for pt in platform_toolsets:
                    status = silent_call(["MSBuild",
                                          str(CURRENT_DIR / "compile" / "windows" / "LightGBM.sln"),
                                          "/p:Configuration=DLL",
                                          "/p:Platform=x64",
                                          f"/p:PlatformToolset={pt}"])
                    if status == 0 and lib_path.is_file():
                        break
                    else:
                        clear_path(CURRENT_DIR / "compile" / "windows" / "x64")
                if status != 0 or not lib_path.is_file():
                    logger.warning("Compilation with MSBuild from existing solution file failed.")
            if status != 0 or not lib_path.is_file():
                arch = "Win32" if bit32 else "x64"
                vs_versions = ("Visual Studio 16 2019", "Visual Studio 15 2017", "Visual Studio 14 2015")
                for vs in vs_versions:
                    logger.info(f"Starting to compile with {vs} ({arch}).")
                    status = silent_call(cmake_cmd + ["-G", vs, "-A", arch])
                    if status == 0:
                        break
                    else:
                        clear_path(build_dir)
                if status != 0:
                    raise Exception("\n".join(('Please install Visual Studio or MS Build and all required dependencies first',
                                    LOG_NOTICE)))
                silent_call(["cmake", "--build", str(build_dir), "--target", "_lightgbm", "--config", "Release"], raise_error=True,
                            error_msg='Please install CMake first')
    else:  # Linux, Darwin (macOS), etc.
        logger.info("Starting to compile with CMake.")
        silent_call(cmake_cmd, raise_error=True, error_msg='Please install CMake and all required dependencies first')
        silent_call(["make", "_lightgbm", f"-I{build_dir}", "-j4"], raise_error=True,
                    error_msg='An error has occurred while building lightgbm library file')
    chdir(original_dir)


class CustomInstallLib(install_lib):

    def install(self) -> List[str]:
        outfiles = install_lib.install(self)
        src = find_lib()[0]
        dst = Path(self.install_dir) / 'lightgbm'
        dst, _ = self.copy_file(src, str(dst))
        outfiles.append(dst)
        return outfiles


class CustomInstall(install):

    user_options = install.user_options + LIGHTGBM_OPTIONS

    def initialize_options(self) -> None:
        install.initialize_options(self)
        self.mingw = False
        self.integrated_opencl = False
        self.gpu = False
        self.cuda = False
        self.boost_root = None
        self.boost_dir = None
        self.boost_include_dir = None
        self.boost_librarydir = None
        self.opencl_include_dir = None
        self.opencl_library = None
        self.mpi = False
        self.hdfs = False
        self.precompile = False
        self.nomp = False
        self.bit32 = False

    def run(self) -> None:
        if (8 * struct.calcsize("P")) != 64:
            if self.bit32:
                logger.warning("You're installing 32-bit version. "
                               "This version is slow and untested, so use it on your own risk.")
            else:
                raise Exception("Cannot install LightGBM in 32-bit Python, "
                                "please use 64-bit Python instead.")
        LOG_PATH.touch()
        if not self.precompile:
            copy_files(integrated_opencl=self.integrated_opencl, use_gpu=self.gpu)
            compile_cpp(use_mingw=self.mingw, use_gpu=self.gpu, use_cuda=self.cuda, use_mpi=self.mpi,
                        use_hdfs=self.hdfs, boost_root=self.boost_root, boost_dir=self.boost_dir,
                        boost_include_dir=self.boost_include_dir, boost_librarydir=self.boost_librarydir,
                        opencl_include_dir=self.opencl_include_dir, opencl_library=self.opencl_library,
                        nomp=self.nomp, bit32=self.bit32, integrated_opencl=self.integrated_opencl)
        install.run(self)
        if LOG_PATH.is_file():
            LOG_PATH.unlink()


class CustomBdistWheel(bdist_wheel):

    user_options = bdist_wheel.user_options + LIGHTGBM_OPTIONS

    def initialize_options(self) -> None:
        bdist_wheel.initialize_options(self)
        self.mingw = False
        self.integrated_opencl = False
        self.gpu = False
        self.cuda = False
        self.boost_root = None
        self.boost_dir = None
        self.boost_include_dir = None
        self.boost_librarydir = None
        self.opencl_include_dir = None
        self.opencl_library = None
        self.mpi = False
        self.hdfs = False
        self.precompile = False
        self.nomp = False
        self.bit32 = False

    def finalize_options(self) -> None:
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

    def run(self) -> None:
        copy_files(integrated_opencl=True, use_gpu=True)
        IS_SOURCE_FLAG_PATH.touch()
        rmtree(CURRENT_DIR / 'lightgbm' / 'Release', ignore_errors=True)
        rmtree(CURRENT_DIR / 'lightgbm' / 'windows' / 'x64', ignore_errors=True)
        lib_file = CURRENT_DIR / 'lightgbm' / 'lib_lightgbm.so'
        if lib_file.is_file():
            lib_file.unlink()
        sdist.run(self)
        if IS_SOURCE_FLAG_PATH.is_file():
            IS_SOURCE_FLAG_PATH.unlink()


if __name__ == "__main__":
    CURRENT_DIR = Path(__file__).absolute().parent
    LOG_PATH = Path.home() / 'LightGBM_compilation.log'
    LOG_NOTICE = f"The full version of error log was saved into {LOG_PATH}"
    IS_SOURCE_FLAG_PATH = CURRENT_DIR / '_IS_SOURCE_PACKAGE.txt'
    _version_src = CURRENT_DIR.parent / 'VERSION.txt'
    _version_dst = CURRENT_DIR / 'lightgbm' / 'VERSION.txt'
    if _version_src.is_file():
        copyfile(_version_src, _version_dst)
    version = _version_dst.read_text(encoding='utf-8').strip()
    readme = (CURRENT_DIR / 'README.rst').read_text(encoding='utf-8')

    sys.path.insert(0, str(CURRENT_DIR))

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
          maintainer='Yu Shi',
          maintainer_email='yushi2@microsoft.com',
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
