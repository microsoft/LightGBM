OPTION(USE_MPI "Enable MPI-based distributed learning" OFF)
OPTION(USE_OPENMP "Enable OpenMP" ON)
OPTION(USE_GPU "Enable GPU-accelerated training" OFF)
OPTION(USE_SWIG "Enable SWIG to generate Java API" OFF)
OPTION(USE_HDFS "Enable HDFS support (EXPERIMENTAL)" OFF)
OPTION(USE_TIMETAG "Set to ON to output time costs" OFF)
OPTION(USE_CUDA "Enable CUDA-accelerated training (EXPERIMENTAL)" OFF)
OPTION(USE_DEBUG "Set to ON for Debug mode" OFF)
OPTION(USE_SANITIZER "Use santizer flags" OFF)
SET(SANITIZER_PATH "" CACHE STRING "Path to sanitizer libs")
SET(ENABLED_SANITIZERS "address" "leak" "undefined" CACHE STRING
  "Semicolon separated list of sanitizer names. E.g 'address;leak'. Supported sanitizers are
address, leak, undefined and thread.")
OPTION(BUILD_CPP_TEST "Build C++ tests with Google Test" OFF)
OPTION(BUILD_STATIC_LIB "Build static library" OFF)
OPTION(__BUILD_FOR_R "Set to ON if building lib_lightgbm for use with the R package" OFF)
OPTION(__INTEGRATE_OPENCL "Set to ON if building LightGBM with the OpenCL ICD Loader and its dependencies included" OFF)

if(APPLE)
  OPTION(APPLE_OUTPUT_DYLIB "Output dylib shared library" OFF)
endif(APPLE)

if(__INTEGRATE_OPENCL)
  cmake_minimum_required(VERSION 3.11)
elseif(USE_GPU OR APPLE)
  cmake_minimum_required(VERSION 3.2)
elseif(USE_CUDA)
  cmake_minimum_required(VERSION 3.16)
else()
  cmake_minimum_required(VERSION 3.0)
endif()

PROJECT(lightgbm LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")

#-- Sanitizer
if (USE_SANITIZER)
  include(cmake/Sanitizer.cmake)
  enable_sanitizers("${ENABLED_SANITIZERS}")
endif (USE_SANITIZER)

if(__INTEGRATE_OPENCL)
  set(__INTEGRATE_OPENCL ON CACHE BOOL "" FORCE)
  set(USE_GPU OFF CACHE BOOL "" FORCE)
  message(STATUS "Building library with integrated OpenCL components")
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
  cmake_minimum_required(VERSION 3.16)
elseif(MSVC)
  if(MSVC_VERSION LESS 1900)
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} doesn't support required C++11 features. Please use a newer MSVC.")
  endif()
  cmake_minimum_required(VERSION 3.8)
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
      FILE(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/com/microsoft/ml/lightgbm/windows/x86_64")
      include_directories($ENV{JAVA_HOME}/include/win32)
  elseif(APPLE)
      FILE(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/com/microsoft/ml/lightgbm/osx/x86_64")
      include_directories($ENV{JAVA_HOME}/include/darwin)
  else()
      FILE(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/com/microsoft/ml/lightgbm/linux/x86_64")
      include_directories($ENV{JAVA_HOME}/include/linux)
  endif()
endif(USE_SWIG)

SET(EIGEN_DIR "${PROJECT_SOURCE_DIR}/external_libs/eigen")
include_directories(${EIGEN_DIR})

# See https://gitlab.com/libeigen/eigen/-/blob/master/COPYING.README
ADD_DEFINITIONS(-DEIGEN_MPL2_ONLY)

if(__BUILD_FOR_R)
    find_package(LibR REQUIRED)
    message(STATUS "LIBR_EXECUTABLE: ${LIBR_EXECUTABLE}")
    message(STATUS "LIBR_INCLUDE_DIRS: ${LIBR_INCLUDE_DIRS}")
    message(STATUS "LIBR_CORE_LIBRARY: ${LIBR_CORE_LIBRARY}")
    include_directories(${LIBR_INCLUDE_DIRS})
    ADD_DEFINITIONS(-DLGB_R_BUILD)
endif(__BUILD_FOR_R)

if(USE_TIMETAG)
    ADD_DEFINITIONS(-DTIMETAG)
endif(USE_TIMETAG)

if(USE_DEBUG)
    ADD_DEFINITIONS(-DDEBUG)
endif(USE_DEBUG)

if(USE_MPI)
    find_package(MPI REQUIRED)
    ADD_DEFINITIONS(-DUSE_MPI)
    MESSAGE(STATUS "MPI libraries: " ${MPI_LIBRARIES})
    MESSAGE(STATUS "MPI C++ libraries: " ${MPI_CXX_LIBRARIES})
else()
    ADD_DEFINITIONS(-DUSE_SOCKET)
endif(USE_MPI)

if(USE_CUDA)
    SET(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
    enable_language(CUDA)
    SET(USE_OPENMP ON CACHE BOOL "CUDA requires OpenMP" FORCE)
endif(USE_CUDA)

if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(USE_OPENMP)

if(USE_GPU)
    SET(BOOST_COMPUTE_HEADER_DIR ${PROJECT_SOURCE_DIR}/external_libs/compute/include)
    include_directories(${BOOST_COMPUTE_HEADER_DIR})
    find_package(OpenCL REQUIRED)
    include_directories(${OpenCL_INCLUDE_DIRS})
    MESSAGE(STATUS "OpenCL include directory: " ${OpenCL_INCLUDE_DIRS})
    if (WIN32)
        set(Boost_USE_STATIC_LIBS ON)
    endif()
    find_package(Boost 1.56.0 COMPONENTS filesystem system REQUIRED)
    if (WIN32)
        # disable autolinking in boost
        add_definitions(-DBOOST_ALL_NO_LIB)
    endif()
    include_directories(${Boost_INCLUDE_DIRS})
    ADD_DEFINITIONS(-DUSE_GPU)
endif(USE_GPU)

if(__INTEGRATE_OPENCL)
    if(WIN32)
        include(cmake/IntegratedOpenCL.cmake)
        ADD_DEFINITIONS(-DUSE_GPU)
    else()
        message(FATAL_ERROR "Integrated OpenCL build is available only for Windows")
    endif(WIN32)
endif(__INTEGRATE_OPENCL)

if(USE_CUDA)
    find_package(CUDA 9.0 REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    LIST(APPEND CMAKE_CUDA_FLAGS -Xcompiler=${OpenMP_CXX_FLAGS} -Xcompiler=-fPIC -Xcompiler=-Wall)

    set(CUDA_ARCHS "6.0" "6.1" "6.2" "7.0")
    if(CUDA_VERSION VERSION_GREATER_EQUAL "10.0")
        list(APPEND CUDA_ARCHS "7.5")
    endif()
    list(POP_BACK CUDA_ARCHS CUDA_LAST_SUPPORTED_ARCH)
    list(APPEND CUDA_ARCHS "${CUDA_LAST_SUPPORTED_ARCH}+PTX")
    CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS ${CUDA_ARCHS})

    LIST(APPEND CMAKE_CUDA_FLAGS ${CUDA_ARCH_FLAGS})
    if(USE_DEBUG)
      SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g")
    else()
      SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")
    endif()
    string(REPLACE ";" " " CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
    message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")

    ADD_DEFINITIONS(-DUSE_CUDA)
    if (NOT DEFINED CMAKE_CUDA_STANDARD)
      set(CMAKE_CUDA_STANDARD 11)
      set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()

    set(BASE_DEFINES
     -DPOWER_FEATURE_WORKGROUPS=12
     -DUSE_CONSTANT_BUF=0
    )
    set(ALLFEATS_DEFINES
     ${BASE_DEFINES}
     -DENABLE_ALL_FEATURES
    )
    set(FULLDATA_DEFINES
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
        histo${hsize}${hname} PRIVATE
        -DCONST_HESSIAN=${hconst}
        ${hdir}
      )
    endfunction()

    foreach (hsize _16_64_256)
      add_histogram("${hsize}" "_sp_const" "True" "1" "${BASE_DEFINES}")
      add_histogram("${hsize}" "_sp" "True" "0" "${BASE_DEFINES}")
      add_histogram("${hsize}" "-allfeats_sp_const" "False" "1" "${ALLFEATS_DEFINES}")
      add_histogram("${hsize}" "-allfeats_sp" "False" "0" "${ALLFEATS_DEFINES}")
      add_histogram("${hsize}" "-fulldata_sp_const" "True" "1" "${FULLDATA_DEFINES}")
      add_histogram("${hsize}" "-fulldata_sp" "True" "0" "${FULLDATA_DEFINES}")
    endforeach()
endif(USE_CUDA)

if(USE_HDFS)
    find_package(JNI REQUIRED)
    find_path(HDFS_INCLUDE_DIR hdfs.h REQUIRED)
    find_library(HDFS_LIB NAMES hdfs REQUIRED)
    include_directories(${HDFS_INCLUDE_DIR})
    ADD_DEFINITIONS(-DUSE_HDFS)
    SET(HDFS_CXX_LIBRARIES ${HDFS_LIB} ${JAVA_JVM_LIBRARY})
endif(USE_HDFS)

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
  ADD_DEFINITIONS(-DMM_PREFETCH)
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
  ADD_DEFINITIONS(-DMM_MALLOC)
endif()

if(UNIX OR MINGW OR CYGWIN)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -Wextra -Wall -Wno-ignored-attributes -Wno-unknown-pragmas -Wno-return-type")
    if(USE_DEBUG)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
    else()
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    endif()
    if(USE_SWIG)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-strict-aliasing")
    endif()
    if(NOT USE_OPENMP)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas -Wno-unused-private-field")
    endif()
    if(__BUILD_FOR_R AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-cast-function-type")
    endif()
endif()

if(WIN32 AND MINGW)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++")
endif()

if(MSVC)
    SET(variables
        CMAKE_C_FLAGS_DEBUG
        CMAKE_C_FLAGS_MINSIZEREL
        CMAKE_C_FLAGS_RELEASE
        CMAKE_C_FLAGS_RELWITHDEBINFO
        CMAKE_CXX_FLAGS_DEBUG
        CMAKE_CXX_FLAGS_MINSIZEREL
        CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_RELWITHDEBINFO
    )
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /MP")
    if(USE_DEBUG)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Od")
    else()
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /Ob2 /Oi /Ot /Oy")
    endif()
else()
    if(NOT USE_DEBUG)
      SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -funroll-loops")
    endif()
endif(MSVC)

SET(LightGBM_HEADER_DIR ${PROJECT_SOURCE_DIR}/include)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

include_directories(${LightGBM_HEADER_DIR})

if(APPLE)
  if(APPLE_OUTPUT_DYLIB)
    SET(CMAKE_SHARED_LIBRARY_SUFFIX ".dylib")
  else()
    SET(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
  endif()
endif(APPLE)

if(USE_MPI)
  include_directories(${MPI_CXX_INCLUDE_PATH})
endif(USE_MPI)

file(GLOB SOURCES
    src/boosting/*.cpp
    src/io/*.cpp
    src/metric/*.cpp
    src/objective/*.cpp
    src/network/*.cpp
    src/treelearner/*.cpp
if(USE_CUDA)
    src/treelearner/*.cu
endif(USE_CUDA)
)

add_executable(lightgbm src/main.cpp src/application/application.cpp ${SOURCES})
list(APPEND SOURCES "src/c_api.cpp")

# Only build the R part of the library if building for
# use with the R package
if(__BUILD_FOR_R)
  list(APPEND SOURCES "src/lightgbm_R.cpp")
endif(__BUILD_FOR_R)

if (BUILD_STATIC_LIB)
  add_library(_lightgbm STATIC ${SOURCES})
else()
  add_library(_lightgbm SHARED ${SOURCES})
endif(BUILD_STATIC_LIB)

if(MSVC)
  set_target_properties(_lightgbm PROPERTIES OUTPUT_NAME "lib_lightgbm")
endif(MSVC)

if(USE_SWIG)
  set_property(SOURCE swig/lightgbmlib.i PROPERTY CPLUSPLUS ON)
  LIST(APPEND swig_options -package com.microsoft.ml.lightgbm)
  set_property(SOURCE swig/lightgbmlib.i PROPERTY SWIG_FLAGS "${swig_options}")
  swig_add_module(_lightgbm_swig java swig/lightgbmlib.i)
  swig_link_libraries(_lightgbm_swig _lightgbm)
  # needed to ensure Linux build does not have lib prefix specified twice, e.g. liblib_lightgbm_swig
  set_target_properties(_lightgbm_swig PROPERTIES PREFIX "")
  # needed in some versions of CMake for VS and MinGW builds to ensure output dll has lib prefix
  set_target_properties(_lightgbm_swig PROPERTIES OUTPUT_NAME "lib_lightgbm_swig")
  if(WIN32)
    if(MINGW OR CYGWIN)
        add_custom_command(TARGET _lightgbm_swig POST_BUILD
            COMMAND "${Java_JAVAC_EXECUTABLE}" -d . java/*.java
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${PROJECT_SOURCE_DIR}/lib_lightgbm.dll" com/microsoft/ml/lightgbm/windows/x86_64
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${PROJECT_SOURCE_DIR}/lib_lightgbm_swig.dll" com/microsoft/ml/lightgbm/windows/x86_64
            COMMAND "${Java_JAR_EXECUTABLE}" -cf lightgbmlib.jar com)
    else()
        add_custom_command(TARGET _lightgbm_swig POST_BUILD
            COMMAND "${Java_JAVAC_EXECUTABLE}" -d . java/*.java
            COMMAND cp "${PROJECT_SOURCE_DIR}/Release/*.dll" com/microsoft/ml/lightgbm/windows/x86_64
            COMMAND "${Java_JAR_EXECUTABLE}" -cf lightgbmlib.jar com)
    endif()
  elseif(APPLE)
    add_custom_command(TARGET _lightgbm_swig POST_BUILD
            COMMAND "${Java_JAVAC_EXECUTABLE}" -d . java/*.java
            COMMAND cp "${PROJECT_SOURCE_DIR}/*.dylib" com/microsoft/ml/lightgbm/osx/x86_64
            COMMAND cp "${PROJECT_SOURCE_DIR}/lib_lightgbm_swig.jnilib" com/microsoft/ml/lightgbm/osx/x86_64/lib_lightgbm_swig.dylib
            COMMAND "${Java_JAR_EXECUTABLE}" -cf lightgbmlib.jar com)
  else()
    add_custom_command(TARGET _lightgbm_swig POST_BUILD
	    COMMAND "${Java_JAVAC_EXECUTABLE}" -d . java/*.java
	    COMMAND cp "${PROJECT_SOURCE_DIR}/*.so" com/microsoft/ml/lightgbm/linux/x86_64
	    COMMAND "${Java_JAR_EXECUTABLE}" -cf lightgbmlib.jar com)
  endif()
endif(USE_SWIG)

if(USE_MPI)
  TARGET_LINK_LIBRARIES(lightgbm ${MPI_CXX_LIBRARIES})
  TARGET_LINK_LIBRARIES(_lightgbm ${MPI_CXX_LIBRARIES})
endif(USE_MPI)

if(USE_OPENMP)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        TARGET_LINK_LIBRARIES(lightgbm OpenMP::OpenMP_CXX)
        TARGET_LINK_LIBRARIES(_lightgbm OpenMP::OpenMP_CXX)
    endif()
endif(USE_OPENMP)

if(USE_GPU)
  TARGET_LINK_LIBRARIES(lightgbm ${OpenCL_LIBRARY} ${Boost_LIBRARIES})
  TARGET_LINK_LIBRARIES(_lightgbm ${OpenCL_LIBRARY} ${Boost_LIBRARIES})
endif(USE_GPU)

if(__INTEGRATE_OPENCL)
  # targets OpenCL and Boost are added in IntegratedOpenCL.cmake
  add_dependencies(lightgbm OpenCL Boost)
  add_dependencies(_lightgbm OpenCL Boost)
  # variables INTEGRATED_OPENCL_* are set in IntegratedOpenCL.cmake
  target_include_directories(lightgbm PRIVATE ${INTEGRATED_OPENCL_INCLUDES})
  target_include_directories(_lightgbm PRIVATE ${INTEGRATED_OPENCL_INCLUDES})
  target_compile_definitions(lightgbm PRIVATE ${INTEGRATED_OPENCL_DEFINITIONS})
  target_compile_definitions(_lightgbm PRIVATE ${INTEGRATED_OPENCL_DEFINITIONS})
  target_link_libraries(lightgbm PRIVATE ${INTEGRATED_OPENCL_LIBRARIES})
  target_link_libraries(_lightgbm PRIVATE ${INTEGRATED_OPENCL_LIBRARIES})
endif()

if(USE_CUDA)
  set_target_properties(lightgbm PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  set_target_properties(lightgbm PROPERTIES CUDA_ARCHITECTURES OFF)
  TARGET_LINK_LIBRARIES(
    lightgbm
    ${histograms}
  )
  set_target_properties(_lightgbm PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  set_target_properties(_lightgbm PROPERTIES CUDA_ARCHITECTURES OFF)
  TARGET_LINK_LIBRARIES(
    _lightgbm
    ${histograms}
  )
endif(USE_CUDA)

if(USE_HDFS)
  TARGET_LINK_LIBRARIES(lightgbm ${HDFS_CXX_LIBRARIES})
  TARGET_LINK_LIBRARIES(_lightgbm ${HDFS_CXX_LIBRARIES})
endif(USE_HDFS)

if(WIN32)
    if(MINGW OR CYGWIN)
      TARGET_LINK_LIBRARIES(lightgbm Ws2_32)
      TARGET_LINK_LIBRARIES(_lightgbm Ws2_32)
      TARGET_LINK_LIBRARIES(lightgbm IPHLPAPI)
      TARGET_LINK_LIBRARIES(_lightgbm IPHLPAPI)
    endif(MINGW OR CYGWIN)
endif(WIN32)

if(__BUILD_FOR_R)
  if(MSVC)
    TARGET_LINK_LIBRARIES(_lightgbm ${LIBR_MSVC_CORE_LIBRARY})
  else()
    TARGET_LINK_LIBRARIES(_lightgbm ${LIBR_CORE_LIBRARY})
  endif(MSVC)
endif(__BUILD_FOR_R)

#-- Google C++ tests
if(BUILD_CPP_TEST)
  find_package(GTest CONFIG)
  if(NOT GTEST_FOUND)
    message(STATUS "Did not find Google Test in the system root. Fetching Google Test now...")
    include(FetchContent)
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG        release-1.10.0
    )
    FetchContent_MakeAvailable(googletest)
    add_library(GTest::GTest ALIAS gtest)
  endif()
  file(GLOB CPP_TEST_SOURCES tests/cpp_tests/*.cpp)
  if(MSVC)
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
  endif(MSVC)
  add_executable(testlightgbm ${CPP_TEST_SOURCES} ${SOURCES})
  target_link_libraries(testlightgbm PRIVATE GTest::GTest)
endif()

install(TARGETS lightgbm _lightgbm
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

install(DIRECTORY ${LightGBM_HEADER_DIR}/LightGBM DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
