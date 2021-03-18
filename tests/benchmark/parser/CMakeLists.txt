cmake_minimum_required(VERSION 3.0)

project(benchmark)

OPTION(USE_PRECISE_TEXT_PARSER "Use precise (and faster) double parser for text input file" OFF)

if(USE_PRECISE_TEXT_PARSER)
    ADD_DEFINITIONS(-DUSE_PRECISE_TEXT_PARSER)
endif(USE_PRECISE_TEXT_PARSER)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

include_directories(${CMAKE_SOURCE_DIR}/../../../include)
include_directories(${CMAKE_SOURCE_DIR}/../../../src)
#link_directories(${CMAKE_SOURCE_DIR}/lib)

add_executable(parser parser.cpp)
