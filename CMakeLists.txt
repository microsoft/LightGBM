if(MSVC)
  cmake_minimum_required(VERSION 3.8)
elseif(USE_GPU OR APPLE)
  cmake_minimum_required(VERSION 3.2)
else()
  cmake_minimum_required(VERSION 2.8)
endif()

PROJECT(lightgbm)

OPTION(USE_MPI "MPI based parallel learning" OFF)
OPTION(USE_OPENMP "Enable OpenMP" ON)
OPTION(USE_GPU "Enable GPU-acclerated training (EXPERIMENTAL)" OFF)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "4.8.2")
    message(FATAL_ERROR "Insufficient gcc version")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "3.8")
    message(FATAL_ERROR "Insufficient Clang version")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    message(FATAL_ERROR "AppleClang wasn't supported. Please see https://github.com/Microsoft/LightGBM/blob/master/docs/Installation-Guide.rst#macos")
endif()

if(APPLE)
    OPTION(APPLE_OUTPUT_DYLIB "Output dylib shared library" OFF)
endif()

if(USE_MPI)
    find_package(MPI REQUIRED)
    ADD_DEFINITIONS(-DUSE_MPI)
    MESSAGE(${MPI_LIBRARIES})
    MESSAGE(${MPI_CXX_LIBRARIES})
else()
    ADD_DEFINITIONS(-DUSE_SOCKET)
endif(USE_MPI)

if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    # Ignore unknown #pragma warning
    if( (CMAKE_CXX_COMPILER_ID MATCHES "[cC][lL][aA][nN][gG]")
      OR (CMAKE_CXX_COMPILER_ID MATCHES "[gG][nN][uU]"))
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
    endif()
endif(USE_OPENMP)

if(USE_GPU)
    SET(BOOST_COMPUTE_HEADER_DIR ${PROJECT_SOURCE_DIR}/compute/include)
    include_directories (${BOOST_COMPUTE_HEADER_DIR})
    find_package(OpenCL REQUIRED)
    include_directories(${OpenCL_INCLUDE_DIRS})
    MESSAGE(STATUS "OpenCL include directory:" ${OpenCL_INCLUDE_DIRS})
    if (WIN32)
        set(Boost_USE_STATIC_LIBS ON)
    endif()
    find_package(Boost 1.56.0 COMPONENTS filesystem system REQUIRED)
    if (WIN32)
        # disable autolinking in boost
        add_definitions( -DBOOST_ALL_NO_LIB )
    endif()
    include_directories(${Boost_INCLUDE_DIRS})
    ADD_DEFINITIONS(-DUSE_GPU)
endif(USE_GPU)

if(UNIX OR MINGW OR CYGWIN)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -O3 -Wextra -Wall -Wno-ignored-attributes -Wno-unknown-pragmas")
endif()

if(WIN32 AND MINGW)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++")
endif()

if(MSVC)
    if(MSVC_VERSION LESS 1900)
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a newer MSVC.")
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
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /O2 /Ob2 /Oi /Ot /Oy /GL")
else()
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()


SET(LightGBM_HEADER_DIR ${PROJECT_SOURCE_DIR}/include)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

include_directories (${LightGBM_HEADER_DIR})

if(APPLE)
  if (APPLE_OUTPUT_DYLIB)
    SET(CMAKE_SHARED_LIBRARY_SUFFIX ".dylib")
  else()
    SET(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
  endif()
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
add_library(_lightgbm SHARED src/c_api.cpp src/lightgbm_R.cpp ${SOURCES})

if(MSVC)
    set_target_properties(_lightgbm PROPERTIES OUTPUT_NAME "lib_lightgbm")
endif(MSVC)

if(USE_MPI)
  TARGET_LINK_LIBRARIES(lightgbm ${MPI_CXX_LIBRARIES})
  TARGET_LINK_LIBRARIES(_lightgbm ${MPI_CXX_LIBRARIES})
endif(USE_MPI)

if(USE_GPU)
  TARGET_LINK_LIBRARIES(lightgbm ${OpenCL_LIBRARY} ${Boost_LIBRARIES})
  TARGET_LINK_LIBRARIES(_lightgbm ${OpenCL_LIBRARY} ${Boost_LIBRARIES})
endif(USE_GPU)

if(WIN32 AND (MINGW OR CYGWIN))
    TARGET_LINK_LIBRARIES(lightgbm Ws2_32)
    TARGET_LINK_LIBRARIES(_lightgbm Ws2_32)
    TARGET_LINK_LIBRARIES(lightgbm IPHLPAPI)
    TARGET_LINK_LIBRARIES(_lightgbm IPHLPAPI)
endif()

install(TARGETS lightgbm _lightgbm
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

install(DIRECTORY ${LightGBM_HEADER_DIR}/LightGBM DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
