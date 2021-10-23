cmake_minimum_required (VERSION 3.15)

set(PROJECT_NAME NeuralTreeEvaluatorTest)

Project(${PROJECT_NAME})

link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../DynamicRank.FreeForm.Library/lib/Shared
    ${CMAKE_CURRENT_SOURCE_DIR}/../../DynamicRank.FreeForm.Library/lib/Expression
    ${CMAKE_CURRENT_SOURCE_DIR}/../../DynamicRank.FreeForm.Library/lib/Backend/llvm
    ${CMAKE_CURRENT_SOURCE_DIR}/../../DynamicRank.FreeForm.Library/lib/Transform
    ${CMAKE_CURRENT_SOURCE_DIR}/../../DynamicRank.FreeForm.Library/lib/Parse/SExpression/lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../../DynamicRank.FreeForm.Library/lib/External
    ${CMAKE_CURRENT_SOURCE_DIR}/../../NeuralTree.Library/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    )

add_executable(${PROJECT_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/NeuralTreeEvaluatorLibTest.cpp
    )

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR} 
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../NeuralTree.Library/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../DynamicRank.FreeForm.Library/lib/Shared
    ${CMAKE_CURRENT_SOURCE_DIR}/../../DynamicRank.FreeForm.Library/lib/Expression
    ${CMAKE_CURRENT_SOURCE_DIR}/../../DynamicRank.FreeForm.Library/lib/Backend/llvm
    ${CMAKE_CURRENT_SOURCE_DIR}/../../DynamicRank.FreeForm.Library/lib/Transform
    ${CMAKE_CURRENT_SOURCE_DIR}/../../DynamicRank.FreeForm.Library/lib/Parse/SExpression/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../DynamicRank.FreeForm.Library/lib/External
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
    NeuralTreeEvaluatorLibrary
    ${BOOST_LIB}
    ${LLVM_LIB}
    -Wl,--end-group
    )
