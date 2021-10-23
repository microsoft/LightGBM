set(PROJECT_NAME DRNeuralTreeLibrary)

set(Headers
    ../inc/basic_types.h
    ../inc/CsHash.h
    ../inc/FeaSpecConfig.h
    ../inc/IFeatureMap.h
    ../inc/INeuralNetFeatures.h
    ../inc/MigratedApi.h
    ../inc/NeuralInput_types.h
    ../inc/NeuralInput.h
    ../inc/NeuralInputFactory.h
    ../inc/NeuralInputFreeForm2_types.h
    ../inc/UnionBondInput_types.h
)
source_group("Headers" FILES ${Headers})

set(Sources
    "CsHash.cpp"
    "FeaSpecConfig.cpp"
    "NeuralInput.cpp"
    "NeuralInputFactory.cpp"
)
source_group("Sources" FILES ${Sources})

set(ALL_FILES
    ${Headers}
    ${Sources}
)

add_library(${PROJECT_NAME} STATIC ${ALL_FILES})

target_include_directories(${PROJECT_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../inc"
)
