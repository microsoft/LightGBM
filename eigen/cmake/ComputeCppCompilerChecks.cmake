cmake_minimum_required(VERSION 3.4.3)

if(CMAKE_COMPILER_IS_GNUCXX)
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.8)
    message(FATAL_ERROR "host compiler - gcc version must be > 4.8")
  endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  if (${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 3.6)
    message(FATAL_ERROR "host compiler - clang version must be > 3.6")
  endif()
endif()

if(MSVC)
  set(ComputeCpp_STL_CHECK_SRC __STL_check)
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${ComputeCpp_STL_CHECK_SRC}.cpp
    "#include <ios>\n"
    "int main() { return 0; }\n")
  execute_process(
    COMMAND ${ComputeCpp_DEVICE_COMPILER_EXECUTABLE}
            ${COMPUTECPP_DEVICE_COMPILER_FLAGS}
            -isystem ${ComputeCpp_INCLUDE_DIRS}
            -o ${ComputeCpp_STL_CHECK_SRC}.sycl
            -c ${ComputeCpp_STL_CHECK_SRC}.cpp
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    RESULT_VARIABLE ComputeCpp_STL_CHECK_RESULT
    ERROR_QUIET
    OUTPUT_QUIET)
  if(NOT ${ComputeCpp_STL_CHECK_RESULT} EQUAL 0)
    # Try disabling compiler version checks
    execute_process(
      COMMAND ${ComputeCpp_DEVICE_COMPILER_EXECUTABLE}
              ${COMPUTECPP_DEVICE_COMPILER_FLAGS}
              -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
              -isystem ${ComputeCpp_INCLUDE_DIRS}
              -o ${ComputeCpp_STL_CHECK_SRC}.cpp.sycl
              -c ${ComputeCpp_STL_CHECK_SRC}.cpp
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      RESULT_VARIABLE ComputeCpp_STL_CHECK_RESULT
      ERROR_QUIET
      OUTPUT_QUIET)
    if(NOT ${ComputeCpp_STL_CHECK_RESULT} EQUAL 0)
      message(STATUS "Device compiler cannot consume hosted STL headers. Using any parts of the STL will likely result in device compiler errors.")
    else()
    message(STATUS "Device compiler does not meet certain STL version requirements. Disabling version checks and hoping for the best.")
      list(APPEND COMPUTECPP_DEVICE_COMPILER_FLAGS -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH)
    endif()
  endif()
  file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/${ComputeCpp_STL_CHECK_SRC}.cpp
              ${CMAKE_CURRENT_BINARY_DIR}/${ComputeCpp_STL_CHECK_SRC}.cpp.sycl)
endif(MSVC)
