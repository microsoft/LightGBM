cmake_minimum_required(VERSION 3.15.0 FATAL_ERROR)

project(transform)

# following lib paths are required by linking directories not compiling.
set(LLVM_LIB
    LLVMAnalysis
    LLVMAsmParser
    LLVMAsmPrinter
    LLVMBitReader
    LLVMBitWriter
    LLVMCodeGen
    LLVMCore
    LLVMDebugInfo
    LLVMExecutionEngine
    LLVMInstCombine
    LLVMInstrumentation
    LLVMInterpreter
    LLVMipa
    LLVMipo
    LLVMIRReader
    LLVMJIT
    LLVMLineEditor
    LLVMLinker
    LLVMLTO
    LLVMMC
    LLVMMCAnalysis
    LLVMMCDisassembler
    LLVMMCJIT
    LLVMMCParser
    LLVMObjCARCOpts
    LLVMObject
    LLVMOption
    LLVMProfileData
    LLVMRuntimeDyld
    LLVMScalarOpts
    LLVMSelectionDAG
    LLVMSupport
    LLVMTarget
    LLVMTransformUtils
    LLVMVectorize
    LLVMX86AsmParser
    LLVMX86AsmPrinter
    LLVMX86CodeGen
    LLVMX86Desc
    LLVMX86Disassembler
    LLVMX86Info
    LLVMX86Utils
    )

set(BOOST_LIB
    boost_atomic
    boost_chrono
    boost_context
    boost_coroutine
    boost_date_time
    boost_exception
    boost_filesystem
    boost_graph
    boost_graph_parallel
    boost_iostreams
    boost_locale
    boost_log
    boost_log_setup
    boost_math_c99
    boost_math_c99f
    boost_math_c99l
    boost_math_tr1
    boost_math_tr1f
    boost_math_tr1l
    boost_mpi
    boost_prg_exec_monitor
    boost_program_options
    boost_random
    boost_regex
    boost_serialization
    boost_system
    boost_test_exec_monitor
    boost_thread
    boost_timer
    boost_unit_test_framework
    boost_wave
    boost_wserialization
    )

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(USE_DEBUG)
    SET(CMAKE_BUILD_TYPE "Debug")
    SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
    SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/NeuralTree.Library)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/DynamicRank.FreeForm.Library)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/NeuralTreeEvaluator.Library)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/TransformProcessor)
