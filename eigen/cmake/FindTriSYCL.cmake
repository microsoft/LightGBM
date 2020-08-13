#.rst:
# FindTriSYCL
#---------------
#
# TODO : insert Copyright and licence

#########################
#  FindTriSYCL.cmake
#########################
#
#  Tools for finding and building with TriSYCL.
#
#  User must define TRISYCL_INCLUDE_DIR pointing to the triSYCL
#  include directory.
#
#  Latest version of this file can be found at:
#    https://github.com/triSYCL/triSYCL

# Requite CMake version 3.5 or higher
cmake_minimum_required (VERSION 3.5)

# Check that a supported host compiler can be found
if(CMAKE_COMPILER_IS_GNUCXX)
  # Require at least gcc 5.4
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.4)
    message(FATAL_ERROR
      "host compiler - Not found! (gcc version must be at least 5.4)")
  else()
    message(STATUS "host compiler - gcc ${CMAKE_CXX_COMPILER_VERSION}")
  endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  # Require at least clang 3.9
  if (${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 3.9)
    message(FATAL_ERROR
      "host compiler - Not found! (clang version must be at least 3.9)")
  else()
    message(STATUS "host compiler - clang ${CMAKE_CXX_COMPILER_VERSION}")
  endif()
else()
  message(WARNING
    "host compiler - Not found! (triSYCL supports GCC and Clang)")
endif()

#triSYCL options
option(TRISYCL_OPENMP "triSYCL multi-threading with OpenMP" ON)
option(TRISYCL_OPENCL "triSYCL OpenCL interoperability mode" OFF)
option(TRISYCL_NO_ASYNC "triSYCL use synchronous kernel execution" OFF)
option(TRISYCL_DEBUG "triSCYL use debug mode" OFF)
option(TRISYCL_DEBUG_STRUCTORS "triSYCL trace of object lifetimes" OFF)
option(TRISYCL_TRACE_KERNEL "triSYCL trace of kernel execution" OFF)

mark_as_advanced(TRISYCL_OPENMP)
mark_as_advanced(TRISYCL_OPENCL)
mark_as_advanced(TRISYCL_NO_ASYNC)
mark_as_advanced(TRISYCL_DEBUG)
mark_as_advanced(TRISYCL_DEBUG_STRUCTORS)
mark_as_advanced(TRISYCL_TRACE_KERNEL)

#triSYCL definitions
set(CL_SYCL_LANGUAGE_VERSION 220 CACHE VERSION
  "Host language version to be used by trisYCL (default is: 220)")
set(TRISYCL_CL_LANGUAGE_VERSION 220 CACHE VERSION
  "Device language version to be used by trisYCL (default is: 220)")
#set(TRISYCL_COMPILE_OPTIONS "-std=c++1z -Wall -Wextra")
set(CMAKE_CXX_STANDARD 14)
set(CXX_STANDARD_REQUIRED ON)


# Find OpenCL package
if(TRISYCL_OPENCL)
  find_package(OpenCL REQUIRED)
  if(UNIX)
    set(BOOST_COMPUTE_INCPATH /usr/include/compute CACHE PATH
      "Path to Boost.Compute headers (default is: /usr/include/compute)")
  endif()
endif()

# Find OpenMP package
if(TRISYCL_OPENMP)
  find_package(OpenMP REQUIRED)
endif()

# Find Boost
find_package(Boost 1.58 REQUIRED COMPONENTS chrono log)

# If debug or trace we need boost log
if(TRISYCL_DEBUG OR TRISYCL_DEBUG_STRUCTORS OR TRISYCL_TRACE_KERNEL)
  set(LOG_NEEDED ON)
else()
  set(LOG_NEEDED OFF)
endif()

find_package(Threads REQUIRED)

# Find triSYCL directory
if(NOT TRISYCL_INCLUDE_DIR)
  message(FATAL_ERROR
    "triSYCL include directory - Not found! (please set TRISYCL_INCLUDE_DIR")
else()
  message(STATUS "triSYCL include directory - Found ${TRISYCL_INCLUDE_DIR}")
endif()

#######################
#  add_sycl_to_target
#######################
#
#  Sets the proper flags and includes for the target compilation.
#
#  targetName : Name of the target to add a SYCL to.
#  sourceFile : Source file to be compiled for SYCL.
#  binaryDir : Intermediate directory to output the integration header.
#
function(add_sycl_to_target targetName sourceFile binaryDir)

  # Add include directories to the "#include <>" paths
  target_include_directories (${targetName} PUBLIC
    ${TRISYCL_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
    $<$<BOOL:${TRISYCL_OPENCL}>:${OpenCL_INCLUDE_DIRS}>
    $<$<BOOL:${TRISYCL_OPENCL}>:${BOOST_COMPUTE_INCPATH}>)


  # Link dependencies
  target_link_libraries(${targetName} PUBLIC
    $<$<BOOL:${TRISYCL_OPENCL}>:${OpenCL_LIBRARIES}>
    Threads::Threads
    $<$<BOOL:${LOG_NEEDED}>:Boost::log>
    Boost::chrono)


  # Compile definitions
  target_compile_definitions(${targetName} PUBLIC
    $<$<BOOL:${TRISYCL_NO_ASYNC}>:TRISYCL_NO_ASYNC>
    $<$<BOOL:${TRISYCL_OPENCL}>:TRISYCL_OPENCL>
    $<$<BOOL:${TRISYCL_DEBUG}>:TRISYCL_DEBUG>
    $<$<BOOL:${TRISYCL_DEBUG_STRUCTORS}>:TRISYCL_DEBUG_STRUCTORS>
    $<$<BOOL:${TRISYCL_TRACE_KERNEL}>:TRISYCL_TRACE_KERNEL>
    $<$<BOOL:${LOG_NEEDED}>:BOOST_LOG_DYN_LINK>)

  # C++ and OpenMP requirements
  target_compile_options(${targetName} PUBLIC
    ${TRISYCL_COMPILE_OPTIONS}
    $<$<BOOL:${TRISYCL_OPENMP}>:${OpenMP_CXX_FLAGS}>)

  if(${TRISYCL_OPENMP} AND (NOT WIN32))
    # Does not support generator expressions
    set_target_properties(${targetName}
      PROPERTIES
      LINK_FLAGS ${OpenMP_CXX_FLAGS})
  endif()

endfunction()
