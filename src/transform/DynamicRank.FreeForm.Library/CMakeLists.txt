cmake_minimum_required(VERSION 3.15.0 FATAL_ERROR)

project(DynamicRank.FreeForm.Library)

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} -fpermissive")

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libs/Shared)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libs/Expression)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libs/Parse/SExpression/libs)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libs/Transform)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libs/Backend/llvm)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libs/External)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/inc/
        DESTINATION include
        FILES_MATCHING PATTERN "*.h"
        )
