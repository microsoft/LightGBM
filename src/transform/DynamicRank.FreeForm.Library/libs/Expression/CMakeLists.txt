cmake_minimum_required(VERSION 3.15)

set(PROJECT_NAME DRFreeFormExpressionLibrary)

project(${PROJECT_NAME})

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} -fpermissive")

add_library(${PROJECT_NAME} STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/Allocation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ArrayDereferenceExpression.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ArrayLength.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ArrayLiteralExpression.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/BinaryOperator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/BlockExpression.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Conditional.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ConvertExpression.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DebugExpression.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Declaration.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Expression.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Extern.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/FeatureSpec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Function.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/LetExpression.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/LiteralExpression.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Match.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/MemberAccessExpression.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Mutation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/OperatorExpression.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/PhiNode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Publish.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RandExpression.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RangeReduceExpression.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RefExpression.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SelectNth.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/StateMachine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/StreamData.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SymbolTable.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TypeUtil.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/UnaryOperator.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR} 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../NeuralTree.Library/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../Shared
    )

install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    )