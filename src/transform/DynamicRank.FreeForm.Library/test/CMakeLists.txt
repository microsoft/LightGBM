cmake_minimum_required(VERSION 3.15)

set(PROJECT_NAME FreeForm2Test)

project(${PROJECT_NAME})

link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../libs/Shared
    ${CMAKE_CURRENT_SOURCE_DIR}/../libs/Expression
    ${CMAKE_CURRENT_SOURCE_DIR}/../libs/Backend/llvm
    ${CMAKE_CURRENT_SOURCE_DIR}/../libs/Transform
    ${CMAKE_CURRENT_SOURCE_DIR}/../libs/Parse/SExpression/lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../libs/External
    ${CMAKE_CURRENT_SOURCE_DIR}/../../NeuralTree.Library/src
    )

add_executable(${PROJECT_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/FreeFormLibTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SimpleFeatureMap.cpp
    )

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR} 
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../NeuralTree.Library/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../libs/Shared
    ${CMAKE_CURRENT_SOURCE_DIR}/../libs/Expression
    ${CMAKE_CURRENT_SOURCE_DIR}/../libs/Backend/llvm
    ${CMAKE_CURRENT_SOURCE_DIR}/../libs/Transform
    ${CMAKE_CURRENT_SOURCE_DIR}/../libs/Parse/SExpression/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../libs/External
    )

target_link_libraries(${PROJECT_NAME}
    -Wl,--no-as-needed
    -Wl,--start-group
    -lpthread
    -lz
    -ldl
    -ltinfo
    DRNeuralTreeLibrary
    DRFreeFormLlvmBackendLibrary
    DRFreeFormSharedLibrary
    DRFreeFormExpressionLibrary
    DRFreeFormTransformLibrary
    DRFreeFormSExpressionLibrary
    DRFreeFormLibrary
    ${BOOST_LIB}
    ${LLVM_LIB}
    -Wl,--end-group
    )
