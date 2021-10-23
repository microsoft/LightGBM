set(PROJECT_NAME NeuralTreeEvaluatorLibrary)

set(Headers
    "InputExtractor.h"
    "LocalFactoryHolder.h"
    "MinimalFeatureMap.h"
)
source_group("Headers" FILES ${Headers})

set(Sources
    "FeatureMap.cpp"
    "InputComputation.cpp"
    "InputExtraction.cpp"
    "InputExtractor.cpp"
    "LocalFactoryHolder.cpp"
    "MinimalFeatureMap.cpp"
)
source_group("Sources" FILES ${Sources})

set(inc
    "../inc/FeatureMap.h"
    "../inc/InputComputation.h"
    "../inc/InputExtraction.h"
)
source_group("inc" FILES ${inc})

set(ALL_FILES
    ${Headers}
    ${Sources}
    ${inc}
)

add_library(${PROJECT_NAME} STATIC ${ALL_FILES})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../NeuralTree.Library/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../DynamicRank.FreeForm.Library/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../DynamicRank.FreeForm.Library/lib/Shared
    ${CMAKE_CURRENT_SOURCE_DIR}/../../DynamicRank.FreeForm.Library/lib/Expression
    ${CMAKE_CURRENT_SOURCE_DIR}/../../DynamicRank.FreeForm.Library/lib/Backend/llvm
    ${CMAKE_CURRENT_SOURCE_DIR}/../../DynamicRank.FreeForm.Library/lib/Transform
    ${CMAKE_CURRENT_SOURCE_DIR}/../../DynamicRank.FreeForm.Library/lib/Parse/SExpression/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../DynamicRank.FreeForm.Library/lib/External
)
