cmake_minimum_required(VERSION 3.15)

set(PROJECT_NAME DRFreeFormSExpressionLibrary)


Project(${PROJECT_NAME})

SET(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} -fpermissive")



add_library(${PROJECT_NAME} STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/Arithmetic.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Bitwise.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ExpressionFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/FreeForm2Tokenizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Logic.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/MiscFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ProgramParseState.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SExpressionParse.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR} 
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../NeuralTree.Library/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../Transform
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../Shared
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../Expression
    )
