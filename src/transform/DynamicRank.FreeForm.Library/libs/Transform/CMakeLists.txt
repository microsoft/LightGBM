cmake_minimum_required(VERSION 3.15)

set(PROJECT_NAME DRFreeFormTransformLibrary)


Project(${PROJECT_NAME})

SET(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} -fpermissive")



add_library(${PROJECT_NAME} STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/AllocationVisitor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CopyingVisitor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/FunctionInlineVisitor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ObjectResolutionVisitor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/OperandPromotionVisitor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ProcessFeaturesUsed.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TypeCheckingVisitor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/UniformExpressionVisitor.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR} 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../NeuralTree.Library/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../Shared
    ${CMAKE_CURRENT_SOURCE_DIR}/../Expression
    )

install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    )