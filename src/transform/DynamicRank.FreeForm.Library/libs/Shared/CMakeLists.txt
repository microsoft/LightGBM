cmake_minimum_required(VERSION 3.15)

set(PROJECT_NAME DRFreeFormSharedLibrary)


Project(${PROJECT_NAME})

SET(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} -fpermissive")



add_library(${PROJECT_NAME} STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/ArrayType.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CompoundType.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/FreeForm2Assert.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/FreeForm2Utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/FunctionType.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ObjectType.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/StateMachineType.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/StructType.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TypeImpl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TypeManager.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR} 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../NeuralTree.Library/inc
    )

install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    )