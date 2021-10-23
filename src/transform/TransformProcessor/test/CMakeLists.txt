cmake_minimum_required(VERSION 3.15)

set(PROJECT_NAME Tests)

project(${PROJECT_NAME})

link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../NeuralTree.Library/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../NeuralTreeEvaluator.Library/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../TransformProcessor/TransformProcessor
    )

add_executable(${PROJECT_NAME} 
    Tests.h
    Tests.cpp
    )

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR} 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../NeuralTreeEvaluator.Library/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../NeuralTree.Library/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../TransformProcessor/TransformProcessor
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
    _transform
    ${BOOST_LIB}
    ${LLVM_LIB}
    -Wl,--end-group
    )

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD 
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                   ${CMAKE_CURRENT_SOURCE_DIR}/data/ $<TARGET_FILE_DIR:${PROJECT_NAME}>/TestData)
