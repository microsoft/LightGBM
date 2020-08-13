FILE(GLOB examples_SRCS "*.cpp")

set(EIGEN_SYCL ON)
list(APPEND CMAKE_EXE_LINKER_FLAGS -pthread)
if(EIGEN_SYCL_TRISYCL)
  set(CMAKE_CXX_STANDARD 14)
  set(STD_CXX_FLAG "-std=c++1z")
else(EIGEN_SYCL_TRISYCL)
  if(MSVC)
    # Set the host and device compilers C++ standard to C++14. On Windows setting this to C++11
    # can cause issues with the ComputeCpp device compiler parsing Visual Studio Headers.
    set(CMAKE_CXX_STANDARD 14)
    list(APPEND COMPUTECPP_USER_FLAGS -DWIN32)
  else()
    set(CMAKE_CXX_STANDARD 11)
    list(APPEND COMPUTECPP_USER_FLAGS -Wall)
  endif()
  # The following flags are not supported by Clang and can cause warnings
  # if used with -Werror so they are removed here.
  if(COMPUTECPP_USE_COMPILER_DRIVER)
    set(CMAKE_CXX_COMPILER ${ComputeCpp_DEVICE_COMPILER_EXECUTABLE})
    string(REPLACE "-Wlogical-op" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    string(REPLACE "-Wno-psabi" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    string(REPLACE "-ansi" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
  endif()
  list(APPEND COMPUTECPP_USER_FLAGS
      -DEIGEN_NO_ASSERTION_CHECKING=1
      -no-serial-memop
      -Xclang
      -cl-mad-enable)
endif(EIGEN_SYCL_TRISYCL)

FOREACH(example_src ${examples_SRCS})
  GET_FILENAME_COMPONENT(example ${example_src} NAME_WE)
  ei_add_test_internal(${example} example_${example})
  ADD_DEPENDENCIES(unsupported_examples example_${example})
ENDFOREACH(example_src)
set(EIGEN_SYCL OFF)
