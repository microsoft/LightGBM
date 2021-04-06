set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BOOST_VERSION_DOT "1.74")
string(REPLACE "." "_" BOOST_VERSION_UNDERSCORE ${BOOST_VERSION_DOT})

set(OPENCL_HEADER_REPOSITORY "https://github.com/KhronosGroup/OpenCL-Headers.git")
set(OPENCL_HEADER_TAG "1b2a1850f410aaaaeaa56cead5a179b5aea4918e")

set(OPENCL_LOADER_REPOSITORY "https://github.com/KhronosGroup/OpenCL-ICD-Loader.git")
set(OPENCL_LOADER_TAG "98ca71fb9f8484f1cd1999f55224bf9e8d18693b")

set(BOOST_REPOSITORY "https://github.com/boostorg/boost.git")
set(BOOST_TAG "boost-${BOOST_VERSION_DOT}.0")

# Build Independent OpenCL library
include(FetchContent)
FetchContent_Declare(OpenCL-Headers GIT_REPOSITORY ${OPENCL_HEADER_REPOSITORY} GIT_TAG ${OPENCL_HEADER_TAG})
FetchContent_GetProperties(OpenCL-Headers)
if(NOT OpenCL-Headers_POPULATED)
  FetchContent_Populate(OpenCL-Headers)
  message(STATUS "Populated OpenCL Headers")
endif()
set(OPENCL_ICD_LOADER_HEADERS_DIR ${opencl-headers_SOURCE_DIR} CACHE PATH "") # for OpenCL ICD Loader
set(OpenCL_INCLUDE_DIR ${opencl-headers_SOURCE_DIR} CACHE PATH "") # for Boost::Compute

FetchContent_Declare(OpenCL-ICD-Loader GIT_REPOSITORY ${OPENCL_LOADER_REPOSITORY} GIT_TAG ${OPENCL_LOADER_TAG})
FetchContent_GetProperties(OpenCL-ICD-Loader)
if(NOT OpenCL-ICD-Loader_POPULATED)
  FetchContent_Populate(OpenCL-ICD-Loader)
  set(USE_DYNAMIC_VCXX_RUNTIME ON)
  add_subdirectory(${opencl-icd-loader_SOURCE_DIR} ${opencl-icd-loader_BINARY_DIR} EXCLUDE_FROM_ALL)
  message(STATUS "Populated OpenCL ICD Loader")
endif()
list(APPEND INTEGRATED_OPENCL_INCLUDES ${OPENCL_ICD_LOADER_HEADERS_DIR})
list(APPEND INTEGRATED_OPENCL_LIBRARIES ${opencl-icd-loader_BINARY_DIR}/Release/OpenCL.lib cfgmgr32.lib runtimeobject.lib)
list(APPEND INTEGRATED_OPENCL_DEFINITIONS CL_TARGET_OPENCL_VERSION=120)

# Build Independent Boost libraries
include(ExternalProject)
include(ProcessorCount)
ProcessorCount(J)
set(BOOST_BASE "${PROJECT_BINARY_DIR}/Boost")
set(BOOST_BOOTSTRAP "${BOOST_BASE}/source/bootstrap.bat")
set(BOOST_BUILD "${BOOST_BASE}/source/b2.exe")
set(BOOST_FLAGS "")
list(APPEND BOOST_SUBMODULES "libs/algorithm" "libs/align" "libs/any" "libs/array" "libs/assert" "libs/bind" "libs/chrono" "libs/compute" "libs/concept_check" "libs/config" "libs/container" "libs/container_hash" "libs/core" "libs/detail" "libs/filesystem" "libs/foreach" "libs/format" "libs/function" "libs/function_types" "libs/fusion" "libs/headers" "libs/integer" "libs/io" "libs/iterator" "libs/lexical_cast" "libs/math" "libs/move" "libs/mpl" "libs/multi_index" "libs/numeric/conversion" "libs/optional" "libs/predef" "libs/preprocessor" "libs/property_tree" "libs/range" "libs/ratio" "libs/serialization" "libs/smart_ptr" "libs/static_assert" "libs/system" "libs/throw_exception" "libs/tuple" "libs/typeof" "libs/type_index" "libs/type_traits" "libs/utility" "libs/uuid" "libs/winapi" "tools/boost_install" "tools/build")
ExternalProject_Add(Boost
  TMP_DIR "${BOOST_BASE}/tmp"
  STAMP_DIR "${BOOST_BASE}/stamp"
  DOWNLOAD_DIR "${BOOST_BASE}/download"
  SOURCE_DIR "${BOOST_BASE}/source"
  BINARY_DIR "${BOOST_BASE}/source"
  INSTALL_DIR "${BOOST_BASE}/install"
  GIT_REPOSITORY ${BOOST_REPOSITORY}
  GIT_TAG ${BOOST_TAG}
  GIT_SUBMODULES ${BOOST_SUBMODULES}
  GIT_SHALLOW ON
  UPDATE_COMMAND ""
  PATCH_COMMAND ""
  CONFIGURE_COMMAND ${BOOST_BOOTSTRAP}
  BUILD_COMMAND ${BOOST_BUILD} -sBOOST_ROOT=${BOOST_BASE}/source -a -q -j ${J} --with-headers --with-chrono --with-filesystem --with-system link=static runtime-link=shared variant=release threading=multi cxxflags="${BOOST_FLAGS}"
  INSTALL_COMMAND ""
)
set(BOOST_INCLUDE "${BOOST_BASE}/source" CACHE PATH "")
set(BOOST_LIBRARY "${BOOST_BASE}/source/stage/lib" CACHE PATH "")
list(APPEND INTEGRATED_OPENCL_INCLUDES ${BOOST_INCLUDE})
if(MSVC)
  if(${MSVC_VERSION} GREATER 1929)
    message(FATAL_ERROR "Unrecognized MSVC version number: ${MSVC_VERSION}")
  elseif(${MSVC_VERSION} GREATER 1919)
    set(MSVC_TOOLCHAIN_ID "142")
  elseif(${MSVC_VERSION} GREATER 1909)
    set(MSVC_TOOLCHAIN_ID "141")
  elseif(${MSVC_VERSION} GREATER 1899)
    set(MSVC_TOOLCHAIN_ID "140")
  else()
    message(FATAL_ERROR "Unrecognized MSVC version number: ${MSVC_VERSION}")
  endif()
  list(APPEND INTEGRATED_OPENCL_LIBRARIES ${BOOST_LIBRARY}/libboost_filesystem-vc${MSVC_TOOLCHAIN_ID}-mt-x64-${BOOST_VERSION_UNDERSCORE}.lib)
  list(APPEND INTEGRATED_OPENCL_LIBRARIES ${BOOST_LIBRARY}/libboost_system-vc${MSVC_TOOLCHAIN_ID}-mt-x64-${BOOST_VERSION_UNDERSCORE}.lib)
  list(APPEND INTEGRATED_OPENCL_LIBRARIES ${BOOST_LIBRARY}/libboost_chrono-vc${MSVC_TOOLCHAIN_ID}-mt-x64-${BOOST_VERSION_UNDERSCORE}.lib)
else()
  message(FATAL_ERROR "Integrated OpenCL build is not yet available for MinGW")
endif()

set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
