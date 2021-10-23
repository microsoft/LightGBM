cmake_minimum_required(VERSION 3.15)

set(PROJECT_NAME DRFreeFormLlvmBackendLibrary)

project(${PROJECT_NAME})

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} -fpermissive")

add_library(${PROJECT_NAME} STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/Extend/FreeForm2Support.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Extend/JITEmitter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Extend/JITExtend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CompilationState.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ArrayCodeGen.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/LlvmCodeGenerator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/LlvmCodeGenUtils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/LlvmCompiler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/LlvmRuntimeLibrary.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR} 
    ${CMAKE_CURRENT_SOURCE_DIR}/Extend
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../NeuralTree.Library/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Shared
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Expression
    ${CMAKE_CURRENT_SOURCE_DIR}/../../External
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Transform
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Parse/SExpression/inc
    )

install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    )