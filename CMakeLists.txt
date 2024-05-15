option(USE_MPI "Enable MPI-based distributed learning" OFF)
option(USE_OPENMP "Enable OpenMP" ON)
option(USE_GPU "Enable GPU-accelerated training" OFF)
option(USE_SWIG "Enable SWIG to generate Java API" OFF)
option(USE_HDFS "Enable HDFS support (EXPERIMENTAL)" OFF)
option(USE_TIMETAG "Set to ON to output time costs" OFF)
option(USE_CUDA "Enable CUDA-accelerated training " OFF)
option(USE_DEBUG "Set to ON for Debug mode" OFF)
option(USE_SANITIZER "Use santizer flags" OFF)
set(
  ENABLED_SANITIZERS
  "address" "leak" "undefined"
  CACHE
  STRING
  "Semicolon separated list of sanitizer names, e.g., 'address;leak'. \
Supported sanitizers are address, leak, undefined and thread."
)
option(BUILD_CLI "Build the 'lightbgm' command-line interface in addition to lib_lightgbm" ON)
option(BUILD_CPP_TEST "Build C++ tests with Google Test" OFF)
option(BUILD_STATIC_LIB "Build static library" OFF)
option(INSTALL_HEADERS "Install headers to CMAKE_INSTALL_PREFIX (e.g. '/usr/local/include')" ON)
option(__BUILD_FOR_PYTHON "Set to ON if building lib_lightgbm for use with the Python package" OFF)
option(__BUILD_FOR_R "Set to ON if building lib_lightgbm for use with the R package" OFF)
option(__INTEGRATE_OPENCL "Set to ON if building LightGBM with the OpenCL ICD Loader and its dependencies included" OFF)

cmake_minimum_required(VERSION 3.18)

project(lightgbm LANGUAGES C CXX)

if(BUILD_CPP_TEST)
  set(CMAKE_CXX_STANDARD 14)
else()
  set(CMAKE_CXX_STANDARD 11)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")

#-- Sanitizer
if(USE_SANITIZER)
  if(MSVC)
    message(FATAL_ERROR "Sanitizers are not supported with MSVC.")
  endif()
  include(cmake/Sanitizer.cmake)
  enable_sanitizers("${ENABLED_SANITIZERS}")
endif()

if(__INTEGRATE_OPENCL)
  set(__INTEGRATE_OPENCL ON CACHE BOOL "" FORCE)
  set(USE_GPU OFF CACHE BOOL "" FORCE)
  message(STATUS "Building library with integrated OpenCL components")
endif()

if(__BUILD_FOR_PYTHON OR __BUILD_FOR_R)
    # the Python and R package don't require the CLI
    set(BUILD_CLI OFF)
    # installing the R and Python package shouldn't place LightGBM's headers
    # outside of where the package is installed
    set(INSTALL_HEADERS OFF)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "4.8.2")
    message(FATAL_ERROR "Insufficient gcc version")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "3.8")
    message(FATAL_ERROR "Insufficient Clang version")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "8.1.0")
    message(FATAL_ERROR "Insufficient AppleClang version")
  endif()
elseif(MSVC)
  if(MSVC_VERSION LESS 1900)
    message(
      FATAL_ERROR
      "The compiler ${CMAKE_CXX_COMPILER} doesn't support required C++11 features. Please use a newer MSVC."
    )
  endif()
endif()

if(USE_SWIG)
  find_package(SWIG REQUIRED)
  find_package(Java REQUIRED)
  find_package(JNI REQUIRED)
  include(UseJava)
  include(UseSWIG)
  set(SWIG_CXX_EXTENSION "cxx")
  set(SWIG_EXTRA_LIBRARIES "")
  set(SWIG_JAVA_EXTRA_FILE_EXTENSIONS ".java" "JNI.java")
  set(SWIG_MODULE_JAVA_LANGUAGE "JAVA")
  set(SWIG_MODULE_JAVA_SWIG_LANGUAGE_FLAG "java")
  set(CMAKE_SWIG_OUTDIR "${CMAKE_CURRENT_BINARY_DIR}/java")
  include_directories(Java_INCLUDE_DIRS)
  include_directories(JNI_INCLUDE_DIRS)
  include_directories($ENV{JAVA_HOME}/include)
  if(WIN32)
      file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/com/microsoft/ml/lightgbm/windows/x86_64")
      include_directories($ENV{JAVA_HOME}/include/win32)
  elseif(APPLE)
      file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/com/microsoft/ml/lightgbm/osx/x86_64")
      include_directories($ENV{JAVA_HOME}/include/darwin)
  else()
      file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/com/microsoft/ml/lightgbm/linux/x86_64")
      include_directories($ENV{JAVA_HOME}/include/linux)
  endif()
endif()

set(EIGEN_DIR "${PROJECT_SOURCE_DIR}/external_libs/eigen")
include_directories(${EIGEN_DIR})

# See https://gitlab.com/libeigen/eigen/-/blob/master/COPYING.README
add_definitions(-DEIGEN_MPL2_ONLY)
add_definitions(-DEIGEN_DONT_PARALLELIZE)

set(FAST_DOUBLE_PARSER_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/external_libs/fast_double_parser/include")
include_directories(${FAST_DOUBLE_PARSER_INCLUDE_DIR})

set(FMT_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/external_libs/fmt/include")
include_directories(${FMT_INCLUDE_DIR})

if(__BUILD_FOR_R)
    find_package(LibR REQUIRED)
    message(STATUS "LIBR_EXECUTABLE: ${LIBR_EXECUTABLE}")
    message(STATUS "LIBR_INCLUDE_DIRS: ${LIBR_INCLUDE_DIRS}")
    message(STATUS "LIBR_CORE_LIBRARY: ${LIBR_CORE_LIBRARY}")
    include_directories(${LIBR_INCLUDE_DIRS})
    add_definitions(-DLGB_R_BUILD)
endif()

if(USE_TIMETAG)
    add_definitions(-DTIMETAG)
endif()

if(USE_DEBUG)
    add_definitions(-DDEBUG)
endif()

if(USE_MPI)
    find_package(MPI REQUIRED)
    add_definitions(-DUSE_MPI)
else()
    add_definitions(-DUSE_SOCKET)
endif()

if(USE_CUDA)
    set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
    enable_language(CUDA)
    set(USE_OPENMP ON CACHE BOOL "CUDA requires OpenMP" FORCE)
endif()

if(USE_OPENMP)
    if(APPLE)
        find_package(OpenMP)
        if(NOT OpenMP_FOUND)
            # libomp 15.0+ from brew is keg-only, so have to search in other locations.
            # See https://github.com/Homebrew/homebrew-core/issues/112107#issuecomment-1278042927.
            execute_process(COMMAND brew --prefix libomp
                            OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
                            OUTPUT_STRIP_TRAILING_WHITESPACE)
            set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
            set(OpenMP_C_LIB_NAMES omp)
            set(OpenMP_CXX_LIB_NAMES omp)
            set(OpenMP_omp_LIBRARY ${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib)
            find_package(OpenMP REQUIRED)
        endif()
    else()
        find_package(OpenMP REQUIRED)
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if(USE_GPU)
    set(BOOST_COMPUTE_HEADER_DIR ${PROJECT_SOURCE_DIR}/external_libs/compute/include)
    include_directories(${BOOST_COMPUTE_HEADER_DIR})
    find_package(OpenCL REQUIRED)
    include_directories(${OpenCL_INCLUDE_DIRS})
    message(STATUS "OpenCL include directory: " ${OpenCL_INCLUDE_DIRS})
    if(WIN32)
        set(Boost_USE_STATIC_LIBS ON)
    endif()
    find_package(Boost 1.56.0 COMPONENTS filesystem system REQUIRED)
    if(WIN32)
        # disable autolinking in boost
        add_definitions(-DBOOST_ALL_NO_LIB)
    endif()
    include_directories(${Boost_INCLUDE_DIRS})
    add_definitions(-DUSE_GPU)
endif()

if(__INTEGRATE_OPENCL)
    if(APPLE)
        message(FATAL_ERROR "Integrated OpenCL build is not available on macOS")
    else()
        include(cmake/IntegratedOpenCL.cmake)
        add_definitions(-DUSE_GPU)
    endif()
endif()

if(USE_CUDA)
    find_package(CUDA 11.0 REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS} -Xcompiler=-fPIC -Xcompiler=-Wall")

    # reference for mapping of CUDA toolkit component versions to supported architectures ("compute capabilities"):
    # https://en.wikipedia.org/wiki/CUDA#GPUs_supported
    set(CUDA_ARCHS "6.0" "6.1" "6.2" "7.0" "7.5")
    if(CUDA_VERSION VERSION_GREATER_EQUAL "11.0")
        list(APPEND CUDA_ARCHS "8.0")
    endif()
    if(CUDA_VERSION VERSION_GREATER_EQUAL "11.1")
        list(APPEND CUDA_ARCHS "8.6")
    endif()
    if(CUDA_VERSION VERSION_GREATER_EQUAL "11.5")
        list(APPEND CUDA_ARCHS "8.7")
    endif()
    if(CUDA_VERSION VERSION_GREATER_EQUAL "11.8")
        list(APPEND CUDA_ARCHS "8.9")
        list(APPEND CUDA_ARCHS "9.0")
    endif()
    list(POP_BACK CUDA_ARCHS CUDA_LAST_SUPPORTED_ARCH)
    list(APPEND CUDA_ARCHS "${CUDA_LAST_SUPPORTED_ARCH}+PTX")
    cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${CUDA_ARCHS})
    string(REPLACE ";" " " CUDA_ARCH_FLAGS "${CUDA_ARCH_FLAGS}")

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_ARCH_FLAGS}")
    if(USE_DEBUG)
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g")
    else()
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")
    endif()
    message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")

    add_definitions(-DUSE_CUDA)

    if(NOT DEFINED CMAKE_CUDA_STANDARD)
      set(CMAKE_CUDA_STANDARD 11)
      set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()

    set(
      BASE_DEFINES
      -DPOWER_FEATURE_WORKGROUPS=12
      -DUSE_CONSTANT_BUF=0
    )
    set(
      ALLFEATS_DEFINES
      ${BASE_DEFINES}
      -DENABLE_ALL_FEATURES
    )
    set(
      FULLDATA_DEFINES
      ${ALLFEATS_DEFINES}
      -DIGNORE_INDICES
    )

    message(STATUS "ALLFEATS_DEFINES: ${ALLFEATS_DEFINES}")
    message(STATUS "FULLDATA_DEFINES: ${FULLDATA_DEFINES}")

    function(add_histogram hsize hname hadd hconst hdir)
      add_library(histo${hsize}${hname} OBJECT src/treelearner/kernels/histogram${hsize}.cu)
      set_target_properties(histo${hsize}${hname} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
      set_target_properties(histo${hsize}${hname} PROPERTIES CUDA_ARCHITECTURES OFF)
      if(hadd)
        list(APPEND histograms histo${hsize}${hname})
        set(histograms ${histograms} PARENT_SCOPE)
      endif()
      target_compile_definitions(
        histo${hsize}${hname}
        PRIVATE
        -DCONST_HESSIAN=${hconst}
        ${hdir}
      )
    endfunction()

    foreach(hsize _16_64_256)
      add_histogram("${hsize}" "_sp_const" "True" "1" "${BASE_DEFINES}")
      add_histogram("${hsize}" "_sp" "True" "0" "${BASE_DEFINES}")
      add_histogram("${hsize}" "-allfeats_sp_const" "False" "1" "${ALLFEATS_DEFINES}")
      add_histogram("${hsize}" "-allfeats_sp" "False" "0" "${ALLFEATS_DEFINES}")
      add_histogram("${hsize}" "-fulldata_sp_const" "True" "1" "${FULLDATA_DEFINES}")
      add_histogram("${hsize}" "-fulldata_sp" "True" "0" "${FULLDATA_DEFINES}")
    endforeach()
endif()

if(USE_HDFS)
    message(
      DEPRECATION
      "HDFS support in LightGBM is deprecated, and will be removed in a future release.\
      See https://github.com/microsoft/LightGBM/issues/6436.
      "
    )
    find_package(JNI REQUIRED)
    find_path(HDFS_INCLUDE_DIR hdfs.h REQUIRED)
    find_library(HDFS_LIB NAMES hdfs REQUIRED)
    include_directories(${HDFS_INCLUDE_DIR})
    add_definitions(-DUSE_HDFS)
    set(HDFS_CXX_LIBRARIES ${HDFS_LIB} ${JAVA_JVM_LIBRARY})
endif()

include(CheckCXXSourceCompiles)
check_cxx_source_compiles("
#include <xmmintrin.h>
int main() {
  int a = 0;
  _mm_prefetch(&a, _MM_HINT_NTA);
  return 0;
}
" MM_PREFETCH)

if(${MM_PREFETCH})
  message(STATUS "Using _mm_prefetch")
  add_definitions(-DMM_PREFETCH)
endif()

include(CheckCXXSourceCompiles)
check_cxx_source_compiles("
#include <mm_malloc.h>
int main() {
  char *a = (char*)_mm_malloc(8, 16);
  _mm_free(a);
  return 0;
}
" MM_MALLOC)

if(${MM_MALLOC})
  message(STATUS "Using _mm_malloc")
  add_definitions(-DMM_MALLOC)
endif()

if(UNIX OR MINGW OR CYGWIN)
  set(
    CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -pthread -Wextra -Wall -Wno-ignored-attributes -Wno-unknown-pragmas -Wno-return-type"
  )
  if(MINGW)
    # ignore this warning: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=95353
    set(
      CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -Wno-stringop-overflow"
    )
  endif()
  if(USE_DEBUG)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
  else()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
  endif()
  if(USE_SWIG)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-strict-aliasing")
  endif()
  if(NOT USE_OPENMP)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas -Wno-unused-private-field")
  endif()
  if(__BUILD_FOR_R AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-cast-function-type")
  endif()
endif()

if(WIN32 AND MINGW)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++")
endif()

# Check if inet_pton is already available, to avoid conflicts with the implementation in LightGBM.
# As of 2022, MinGW started including a definition of inet_pton.
if(WIN32)
  include(CheckSymbolExists)
  list(APPEND CMAKE_REQUIRED_LIBRARIES "ws2_32")
  check_symbol_exists(inet_pton "ws2tcpip.h" WIN_INET_PTON_FOUND)
  if(WIN_INET_PTON_FOUND)
    add_definitions(-DWIN_HAS_INET_PTON)
  endif()
  list(REMOVE_ITEM CMAKE_REQUIRED_LIBRARIES "ws2_32")
endif()

if(MSVC)
    set(
      variables
        CMAKE_C_FLAGS_DEBUG
        CMAKE_C_FLAGS_MINSIZEREL
        CMAKE_C_FLAGS_RELEASE
        CMAKE_C_FLAGS_RELWITHDEBINFO
        CMAKE_CXX_FLAGS_DEBUG
        CMAKE_CXX_FLAGS_MINSIZEREL
        CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_RELWITHDEBINFO
    )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /MP")
    if(__BUILD_FOR_R)
        # MSVC does not like this commit:
        # https://github.com/wch/r-source/commit/fb52ac1a610571fcb8ac92d886b9fefcffaa7d48
        #
        # and raises "error C3646: 'private_data_c': unknown override specifier"
        add_definitions(-DR_LEGACY_RCOMPLEX)
    endif()
    if(USE_DEBUG)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Od")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /Ob2 /Oi /Ot /Oy")
    endif()
else()
    if(NOT BUILD_STATIC_LIB)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    endif()
    if(NOT USE_DEBUG)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops")
    endif()
endif()

set(LightGBM_HEADER_DIR ${PROJECT_SOURCE_DIR}/include)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

include_directories(${LightGBM_HEADER_DIR})

if(USE_MPI)
  include_directories(${MPI_CXX_INCLUDE_PATH})
endif()

file(
    GLOB
    SOURCES
      src/boosting/*.cpp
      src/io/*.cpp
      src/metric/*.cpp
      src/objective/*.cpp
      src/network/*.cpp
      src/treelearner/*.cpp
      src/utils/*.cpp
)
file(
    GLOB
    LGBM_CUDA_SOURCES
      src/treelearner/*.cu
      src/boosting/cuda/*.cpp
      src/boosting/cuda/*.cu
      src/metric/cuda/*.cpp
      src/metric/cuda/*.cu
      src/objective/cuda/*.cpp
      src/objective/cuda/*.cu
      src/treelearner/cuda/*.cpp
      src/treelearner/cuda/*.cu
      src/io/cuda/*.cu
      src/io/cuda/*.cpp
      src/cuda/*.cpp
      src/cuda/*.cu
)

if(USE_CUDA)
  list(APPEND SOURCES ${LGBM_CUDA_SOURCES})
endif()

add_library(lightgbm_objs OBJECT ${SOURCES})

if(BUILD_CLI)
    add_executable(lightgbm src/main.cpp src/application/application.cpp)
    target_link_libraries(lightgbm PRIVATE lightgbm_objs)
endif()

set(API_SOURCES "src/c_api.cpp")
# Only build the R part of the library if building for
# use with the R package
if(__BUILD_FOR_R)
  list(APPEND API_SOURCES "src/lightgbm_R.cpp")
endif()

add_library(lightgbm_capi_objs OBJECT ${API_SOURCES})

if(BUILD_STATIC_LIB)
  add_library(_lightgbm STATIC)
else()
  add_library(_lightgbm SHARED)
endif()

# R expects libraries of the form <project>.{dll,dylib,so}, not lib_<project>.{dll,dylib,so}
if(__BUILD_FOR_R)
  set_target_properties(
    _lightgbm
    PROPERTIES
      PREFIX ""
      OUTPUT_NAME "lightgbm"
  )
endif()

# LightGBM headers include openmp, cuda, R etc. headers,
# thus PUBLIC is required for building _lightgbm_swig target.
target_link_libraries(_lightgbm PUBLIC lightgbm_capi_objs lightgbm_objs)

if(MSVC AND NOT __BUILD_FOR_R)
  set_target_properties(_lightgbm PROPERTIES OUTPUT_NAME "lib_lightgbm")
endif()

if(USE_SWIG)
  set_property(SOURCE swig/lightgbmlib.i PROPERTY CPLUSPLUS ON)
  list(APPEND swig_options -package com.microsoft.ml.lightgbm)
  set_property(SOURCE swig/lightgbmlib.i PROPERTY SWIG_FLAGS "${swig_options}")
  swig_add_library(_lightgbm_swig LANGUAGE java SOURCES swig/lightgbmlib.i)
  swig_link_libraries(_lightgbm_swig _lightgbm)
  # needed to ensure Linux build does not have lib prefix specified twice, e.g. liblib_lightgbm_swig
  set_target_properties(_lightgbm_swig PROPERTIES PREFIX "")
  # needed in some versions of CMake for VS and MinGW builds to ensure output dll has lib prefix
  set_target_properties(_lightgbm_swig PROPERTIES OUTPUT_NAME "lib_lightgbm_swig")
  if(WIN32)
    if(MINGW OR CYGWIN)
        add_custom_command(
            TARGET _lightgbm_swig
            POST_BUILD
            COMMAND "${Java_JAVAC_EXECUTABLE}" -d . java/*.java
            COMMAND
              "${CMAKE_COMMAND}"
              -E
              copy_if_different
              "${PROJECT_SOURCE_DIR}/lib_lightgbm.dll"
              com/microsoft/ml/lightgbm/windows/x86_64
            COMMAND
              "${CMAKE_COMMAND}"
              -E
              copy_if_different
              "${PROJECT_SOURCE_DIR}/lib_lightgbm_swig.dll"
              com/microsoft/ml/lightgbm/windows/x86_64
            COMMAND "${Java_JAR_EXECUTABLE}" -cf lightgbmlib.jar com
        )
    else()
        add_custom_command(
            TARGET _lightgbm_swig
            POST_BUILD
            COMMAND "${Java_JAVAC_EXECUTABLE}" -d . java/*.java
            COMMAND
              "${CMAKE_COMMAND}"
              -E
              copy_if_different
              "${PROJECT_SOURCE_DIR}/Release/lib_lightgbm.dll"
              com/microsoft/ml/lightgbm/windows/x86_64
            COMMAND
              "${CMAKE_COMMAND}"
              -E
              copy_if_different
              "${PROJECT_SOURCE_DIR}/Release/lib_lightgbm_swig.dll"
              com/microsoft/ml/lightgbm/windows/x86_64
            COMMAND "${Java_JAR_EXECUTABLE}" -cf lightgbmlib.jar com
        )
    endif()
  elseif(APPLE)
    add_custom_command(
        TARGET _lightgbm_swig
        POST_BUILD
        COMMAND "${Java_JAVAC_EXECUTABLE}" -d . java/*.java
        COMMAND
          "${CMAKE_COMMAND}"
          -E
          copy_if_different
          "${PROJECT_SOURCE_DIR}/lib_lightgbm.dylib"
          com/microsoft/ml/lightgbm/osx/x86_64
        COMMAND
          "${CMAKE_COMMAND}"
          -E
          copy_if_different
          "${PROJECT_SOURCE_DIR}/lib_lightgbm_swig.jnilib"
          com/microsoft/ml/lightgbm/osx/x86_64/lib_lightgbm_swig.dylib
        COMMAND "${Java_JAR_EXECUTABLE}" -cf lightgbmlib.jar com
    )
  else()
    add_custom_command(
        TARGET _lightgbm_swig
        POST_BUILD
        COMMAND "${Java_JAVAC_EXECUTABLE}" -d . java/*.java
        COMMAND
          "${CMAKE_COMMAND}"
          -E
          copy_if_different
          "${PROJECT_SOURCE_DIR}/lib_lightgbm.so"
          com/microsoft/ml/lightgbm/linux/x86_64
        COMMAND "${Java_JAR_EXECUTABLE}" -cf lightgbmlib.jar com
    )
  endif()
endif()

if(USE_MPI)
  target_link_libraries(lightgbm_objs PUBLIC ${MPI_CXX_LIBRARIES})
endif()

if(USE_OPENMP)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    target_link_libraries(lightgbm_objs PUBLIC OpenMP::OpenMP_CXX)
    # c_api headers also includes OpenMP headers, thus compiling
    # lightgbm_capi_objs needs include directory for OpenMP.
    # Specifying OpenMP in target_link_libraries will get include directory
    # requirements for compilation.
    # This uses CMake's Transitive Usage Requirements. Refer to CMake doc:
    # https://cmake.org/cmake/help/v3.16/manual/cmake-buildsystem.7.html#transitive-usage-requirements
    target_link_libraries(lightgbm_capi_objs PUBLIC OpenMP::OpenMP_CXX)
  endif()
endif()

if(USE_GPU)
  target_link_libraries(lightgbm_objs PUBLIC ${OpenCL_LIBRARY} ${Boost_LIBRARIES})
endif()

if(__INTEGRATE_OPENCL)
  # targets OpenCL and Boost are added in IntegratedOpenCL.cmake
  add_dependencies(lightgbm_objs OpenCL Boost)
  # variables INTEGRATED_OPENCL_* are set in IntegratedOpenCL.cmake
  target_include_directories(lightgbm_objs PRIVATE ${INTEGRATED_OPENCL_INCLUDES})
  target_compile_definitions(lightgbm_objs PRIVATE ${INTEGRATED_OPENCL_DEFINITIONS})
  target_link_libraries(lightgbm_objs PUBLIC ${INTEGRATED_OPENCL_LIBRARIES} ${CMAKE_DL_LIBS})
endif()

if(USE_CUDA)
  # Disable cmake warning about policy CMP0104. Refer to issue #3754 and PR #4268.
  # Custom target properties does not propagate, thus we need to specify for
  # each target that contains or depends on cuda source.
  set_target_properties(lightgbm_objs PROPERTIES CUDA_ARCHITECTURES OFF)
  set_target_properties(_lightgbm PROPERTIES CUDA_ARCHITECTURES OFF)
  if(BUILD_CLI)
    set_target_properties(lightgbm PROPERTIES CUDA_ARCHITECTURES OFF)
  endif()

  set_target_properties(lightgbm_objs PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  # Device linking is not supported for object libraries.
  # Thus we have to specify them on final targets.
  if(BUILD_CLI)
    set_target_properties(lightgbm PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  endif()
  set_target_properties(_lightgbm PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

  # histograms are list of object libraries. Linking object library to other
  # object libraries only gets usage requirements, the linked objects won't be
  # used. Thus we have to call target_link_libraries on final targets here.
  if(BUILD_CLI)
    target_link_libraries(lightgbm PRIVATE ${histograms})
  endif()
  target_link_libraries(_lightgbm PRIVATE ${histograms})
endif()

if(USE_HDFS)
  target_link_libraries(lightgbm_objs PUBLIC ${HDFS_CXX_LIBRARIES})
endif()

if(WIN32)
    if(MINGW OR CYGWIN)
      target_link_libraries(lightgbm_objs PUBLIC ws2_32 iphlpapi)
    endif()
endif()

if(__BUILD_FOR_R)
  # utils/log.h and capi uses R headers, thus both object libraries need to link
  # with R lib.
  if(MSVC)
    set(R_LIB ${LIBR_MSVC_CORE_LIBRARY})
  else()
    set(R_LIB ${LIBR_CORE_LIBRARY})
  endif()
  target_link_libraries(lightgbm_objs PUBLIC ${R_LIB})
  target_link_libraries(lightgbm_capi_objs PUBLIC ${R_LIB})
endif()

#-- Google C++ tests
if(BUILD_CPP_TEST)
  find_package(GTest CONFIG)
  if(NOT GTEST_FOUND)
    message(STATUS "Did not find Google Test in the system root. Fetching Google Test now...")
    include(FetchContent)
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG        v1.14.0
    )
    FetchContent_MakeAvailable(googletest)
    add_library(GTest::GTest ALIAS gtest)
  endif()

  set(LightGBM_TEST_HEADER_DIR ${PROJECT_SOURCE_DIR}/tests/cpp_tests)
  include_directories(${LightGBM_TEST_HEADER_DIR})

  file(GLOB CPP_TEST_SOURCES tests/cpp_tests/*.cpp)
  if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /permissive-")
    set(
      CompilerFlags
        CMAKE_CXX_FLAGS
        CMAKE_CXX_FLAGS_DEBUG
        CMAKE_CXX_FLAGS_RELEASE
        CMAKE_C_FLAGS
        CMAKE_C_FLAGS_DEBUG
        CMAKE_C_FLAGS_RELEASE
    )
    foreach(CompilerFlag ${CompilerFlags})
      string(REPLACE "/MD" "/MT" ${CompilerFlag} "${${CompilerFlag}}")
    endforeach()
  endif()
  add_executable(testlightgbm ${CPP_TEST_SOURCES})
  target_link_libraries(testlightgbm PRIVATE lightgbm_objs lightgbm_capi_objs GTest::GTest)
endif()

if(BUILD_CLI)
    install(
      TARGETS lightgbm
      RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
    )
endif()

if(__BUILD_FOR_PYTHON)
    set(CMAKE_INSTALL_PREFIX "lightgbm")
endif()

install(
  TARGETS _lightgbm
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)

if(INSTALL_HEADERS)
    install(
      DIRECTORY ${LightGBM_HEADER_DIR}/LightGBM
      DESTINATION ${CMAKE_INSTALL_PREFIX}/include
    )
    install(
      FILES ${FAST_DOUBLE_PARSER_INCLUDE_DIR}/fast_double_parser.h
      DESTINATION ${CMAKE_INSTALL_PREFIX}/include/LightGBM/utils
    )
    install(
      DIRECTORY ${FMT_INCLUDE_DIR}/
      DESTINATION ${CMAKE_INSTALL_PREFIX}/include/LightGBM/utils
      FILES_MATCHING PATTERN "*.h"
    )
endif()
