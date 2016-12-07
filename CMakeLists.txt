cmake_minimum_required(VERSION 2.8)

PROJECT(lightgbm)

OPTION(USE_MPI "MPI based parallel learning" OFF)

if(USE_MPI)
    find_package(MPI REQUIRED)
    ADD_DEFINITIONS(-DUSE_MPI)
    MESSAGE(${MPI_LIBRARIES})
    MESSAGE(${MPI_CXX_LIBRARIES})
else()
    ADD_DEFINITIONS(-DUSE_SOCKET)
endif(USE_MPI)

find_package(OpenMP REQUIRED)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

if(UNIX OR MINGW OR CYGWIN)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -O3 -Wall -std=c++11")
endif()

if(MSVC)
    if(MSVC_VERSION LESS 1800)
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a newer msvc.")
    endif()

    SET(variables
        CMAKE_C_FLAGS_DEBUG
        CMAKE_C_FLAGS_MINSIZEREL
        CMAKE_C_FLAGS_RELEASE
        CMAKE_C_FLAGS_RELWITHDEBINFO
        CMAKE_CXX_FLAGS_DEBUG
        CMAKE_CXX_FLAGS_MINSIZEREL
        CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_RELWITHDEBINFO
    )
    foreach(variable ${variables})
        if(${variable} MATCHES "/MD")
            string(REGEX REPLACE "/MD" "/MT" ${variable} "${${variable}}")
        endif()
    endforeach()
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /O2 /Ob2 /Oi /Ot /Oy /GL")
else()
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()


SET(LightGBM_HEADER_DIR ${PROJECT_SOURCE_DIR}/include)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

include_directories (${LightGBM_HEADER_DIR})

if(APPLE)
  SET(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
endif(APPLE)

if(USE_MPI)
  include_directories(${MPI_CXX_INCLUDE_PATH})
endif(USE_MPI)

file(GLOB SOURCES 
    src/application/*.cpp
    src/boosting/*.cpp
    src/io/*.cpp
    src/metric/*.cpp
    src/objective/*.cpp
    src/network/*.cpp
    src/treelearner/*.cpp
)

add_executable(lightgbm src/main.cpp ${SOURCES})
add_library(_lightgbm SHARED src/c_api.cpp ${SOURCES})

if(MSVC)
    set_target_properties(_lightgbm PROPERTIES OUTPUT_NAME "lib_lightgbm")
endif(MSVC)

if(USE_MPI)
  TARGET_LINK_LIBRARIES(lightgbm ${MPI_CXX_LIBRARIES})
  TARGET_LINK_LIBRARIES(_lightgbm ${MPI_CXX_LIBRARIES})
endif(USE_MPI)

if(WIN32 AND (MINGW OR CYGWIN))
    TARGET_LINK_LIBRARIES(lightgbm Ws2_32)
    TARGET_LINK_LIBRARIES(_lightgbm Ws2_32)
    TARGET_LINK_LIBRARIES(lightgbm IPHLPAPI)
    TARGET_LINK_LIBRARIES(_lightgbm IPHLPAPI)
endif()

install(TARGETS lightgbm _lightgbm
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(DIRECTORY ${LightGBM_HEADER_DIR}/LightGBM DESTINATION ${CMAKE_INSTALL_PREFIX}/include)