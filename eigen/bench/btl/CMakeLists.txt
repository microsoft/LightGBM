project(BTL)

cmake_minimum_required(VERSION 2.6.2)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${Eigen_SOURCE_DIR}/cmake)
include(MacroOptionalAddSubdirectory)

option(BTL_NOVEC "Disable SSE/Altivec optimizations when possible" OFF)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

string(REGEX MATCH icpc IS_ICPC ${CMAKE_CXX_COMPILER})
if(CMAKE_COMPILER_IS_GNUCXX OR IS_ICPC)
  set(CMAKE_CXX_FLAGS "-g0 -O3 -DNDEBUG ${CMAKE_CXX_FLAGS}")
  set(CMAKE_Fortran_FLAGS "-g0 -O3 -DNDEBUG ${CMAKE_Fortran_FLAGS}")
  if(BTL_NOVEC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_DONT_VECTORIZE")
  endif(BTL_NOVEC)
endif(CMAKE_COMPILER_IS_GNUCXX OR IS_ICPC)

if(MSVC)
  set(CMAKE_CXX_FLAGS " /O2 /Ot /GL /fp:fast -DNDEBUG")
#   set(CMAKE_Fortran_FLAGS "-g0 -O3 -DNDEBUG")
  if(BTL_NOVEC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_DONT_VECTORIZE")
  endif(BTL_NOVEC)
endif(MSVC)

if(IS_ICPC)
  set(CMAKE_CXX_FLAGS "-fast ${CMAKE_CXX_FLAGS}")
  set(CMAKE_Fortran_FLAGS "-fast ${CMAKE_Fortran_FLAGS}")
endif()

include_directories(
  ${PROJECT_SOURCE_DIR}/actions
  ${PROJECT_SOURCE_DIR}/generic_bench
  ${PROJECT_SOURCE_DIR}/generic_bench/utils
  ${PROJECT_SOURCE_DIR}/libs/STL)

# find_package(MKL)
# if (MKL_FOUND)
#   add_definitions(-DHAVE_MKL)
#   set(DEFAULT_LIBRARIES ${MKL_LIBRARIES})
# endif ()

find_library(EIGEN_BTL_RT_LIBRARY rt)
# if we cannot find it easily, then we don't need it!
if(NOT EIGEN_BTL_RT_LIBRARY)
  set(EIGEN_BTL_RT_LIBRARY "")
endif()

macro(BTL_ADD_BENCH targetname)

  foreach(_current_var ${ARGN})
    set(_last_var ${_current_var})
  endforeach()

  set(_sources ${ARGN})
  list(LENGTH _sources _argn_length)

  list(REMOVE_ITEM _sources ON OFF TRUE FALSE)

  list(LENGTH _sources _src_length)

  if (${_argn_length} EQUAL ${_src_length})
    set(_last_var ON)
  endif ()

  option(BUILD_${targetname} "Build benchmark ${targetname}" ${_last_var})

  if(BUILD_${targetname})
    add_executable(${targetname} ${_sources})
    add_test(${targetname} "${targetname}")
    target_link_libraries(${targetname} ${DEFAULT_LIBRARIES} ${EIGEN_BTL_RT_LIBRARY})
  endif(BUILD_${targetname})

endmacro(BTL_ADD_BENCH)

macro(btl_add_target_property target prop value)

  if(BUILD_${target})
    get_target_property(previous ${target} ${prop})
    if(NOT previous)
      set(previous "")
    endif()
    set_target_properties(${target} PROPERTIES ${prop} "${previous} ${value}")
  endif()

endmacro()

enable_testing()

add_subdirectory(libs/eigen3)
add_subdirectory(libs/eigen2)
add_subdirectory(libs/tensors)
add_subdirectory(libs/BLAS)
add_subdirectory(libs/ublas)
add_subdirectory(libs/gmm)
add_subdirectory(libs/mtl4)
add_subdirectory(libs/blitz)
add_subdirectory(libs/tvmet)
add_subdirectory(libs/STL)
add_subdirectory(libs/blaze)

add_subdirectory(data)


