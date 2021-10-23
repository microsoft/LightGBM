cmake_minimum_required(VERSION 3.15.0 FATAL_ERROR)

project ("TransformProcessor")

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} -fpermissive")

add_subdirectory (${CMAKE_CURRENT_SOURCE_DIR}/TransformProcessor)
add_subdirectory (${CMAKE_CURRENT_SOURCE_DIR}/test)