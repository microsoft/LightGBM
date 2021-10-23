cmake_minimum_required(VERSION 3.15)

set(PROJECT_NAME DRFreeFormLibrary)

project(${PROJECT_NAME})

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} -fpermissive")

add_library(${PROJECT_NAME} STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/Compiler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Executable.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/FreeForm2ExternalData.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/FreeForm2Result.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/FreeForm2Type.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NeuralInputFreeForm2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Program.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ValueResult.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR} 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../NeuralTree.Library/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../Shared
    ${CMAKE_CURRENT_SOURCE_DIR}/../Expression
    ${CMAKE_CURRENT_SOURCE_DIR}/../Backend/llvm
    ${CMAKE_CURRENT_SOURCE_DIR}/../Transform
    ${CMAKE_CURRENT_SOURCE_DIR}/../Parse/SExpression/inc
    )

install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    )