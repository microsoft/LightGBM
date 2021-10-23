cmake_minimum_required(VERSION 3.15.0 FATAL_ERROR)

project(NeuralTree.Library CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -std=c++11")

add_subdirectory(src)
