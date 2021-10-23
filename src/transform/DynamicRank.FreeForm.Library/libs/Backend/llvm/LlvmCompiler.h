#pragma once

#include "ArrayCodeGen.h"
#include <boost/shared_array.hpp>
#include "Compiler.h"
#include "Executable.h"
#include "FreeForm2.h"
#include <memory>
#include "TypeImpl.h"

namespace llvm
{
    namespace legacy {
        class FunctionPassManager;
    }

    using legacy::FunctionPassManager;
    class JITMemoryManager;
}

namespace FreeForm2
{
    // class CodeHeap;
    class CompilationState;
    class LlvmExecutableImpl;
    class PersistentJITMemoryManager;

    // This is to control whether we're usign llvm code gen single-threaded or not.
    // Default value is true, which restricts llvm code gen to be single-threaded globally,
    // due to LLVM's restrictions that may or may not exists in current version.
    // TODO: we're setting this flag to false experimentally in restricted scenario (e.g. URPService TC2)
    // to allow multi-threaded llvm code gen, so that we can alleviate some long-standing
    // pain caused by extremely time-consuming compilation process of very large freeform files.
    // After double confirming that the version of LLVM integrated in current code base is fully
    // multi-thread safe, we shall remove this critical section all together.
    extern bool s_use_llvmCriticalSection;

    class LlvmCompilerImpl : public CompilerImpl
    {
    public:
        LlvmCompilerImpl(unsigned int p_optimizationLevel,
                         CompilerFactory::DestinationFunctionType p_destinationFunctionType);
        virtual ~LlvmCompilerImpl();

        // Compile the given program into an executable.
        virtual 
        std::unique_ptr<CompilerResults> 
        Compile(const ProgramImpl& p_program, 
                bool p_debugOutput) override;

        CompilationState& GetState();
        boost::shared_ptr<llvm::JITMemoryManager> GetMemoryManager();
        llvm::FunctionPassManager& GetFunctionPassManager();
        unsigned int GetOptimizationLevel() const;
        const PersistentJITMemoryManager& GetPersistentMemoryManager() const;

    private:
        std::unique_ptr<CompilationState> m_state;

        boost::shared_ptr<llvm::JITMemoryManager> m_memoryManager;

        std::unique_ptr<llvm::FunctionPassManager> m_functionPassManager;

        PersistentJITMemoryManager* m_persistentMemoryManager;

        unsigned int m_optimizationLevel;

        // Destination function type.
        CompilerFactory::DestinationFunctionType m_destinationFunctionType;
    };

    // An ExectuableImpl is the implementation class for executables, which
    // currently compiles and runs via LLVM. Supports serialization and 
    // deserialization of executable code.
    //
    // There are two ways for LlvmExecutableImpl to be created:
    // * As a result of FreeForm compilation.  In this case binary code is allocated 
    //   by m_memoryManager and m_relocations contain relocation table which should be
    //   serialized with the code.
    // * As a result of deserialization.  In this case binary code is allocaed by m_codeHeap
    //   and relocation table precedes it in the memory.  Re-serilaization just writes
    //   an extent of memory specified by m_funcInfo.
    class LlvmExecutableImpl : public ExecutableImpl
    {
    public:
        static const int c_serializedVersion = 3;

        // Descriptor for LLVM-generated code chunk.
        struct FunctionInfo
        {
            uint8_t* m_start;
            ptrdiff_t m_length;
        };

        // Code relocation descriptor.
        struct RelocationInfo
        {
            // Relocation type.
            enum Type
            {
                // m_delta is an offset inside LLVM-generated code.
                Internal, 
                // m_dela is an index in the table of external functions.
                External
            };

            // Relocation type.
            uint32_t m_type;

            // Offset of relocated value from the m_start of FunctionInfo.
            uint32_t m_offset;

            // See comment for enum Type.
            uint32_t m_delta;
        };

        // Compile a program to executable code.
        LlvmExecutableImpl(LlvmCompilerImpl& p_compiler,
                           const ProgramImpl& p_program, 
                           bool p_dumpExecutable,
                           CompilerFactory::DestinationFunctionType p_destinationFunctionType);

        // Take and wrap deserialized executable code.
        // p_binary and p_binarySize defines a serialized representation of function preceded by relocation table.
        // LlvmExecutableImpl(unsigned char* p_binary,
        //                    size_t p_binarySize,
        //                    const TypeImpl& p_type,
        //                    boost::shared_ptr<CodeHeap> p_codeHeap,
        //                    CompilerFactory::DestinationFunctionType p_destinationFunctionType);

        virtual ~LlvmExecutableImpl();

        virtual boost::shared_ptr<Result> 
        Evaluate(StreamFeatureInput* p_input,
                 const Executable::FeatureType p_features[]) const override;

        virtual boost::shared_ptr<Result>
        Evaluate(const Executable::FeatureType* const* p_features,
                 UInt32 p_currentDocument,
                 UInt32 p_documentCount,
                 Int64* p_cache) const override;

        virtual Executable::DirectEvalFun 
        EvaluationFunction() const override;

        virtual Executable::AggregatedEvalFun
        AggregatedEvaluationFunction() const override;

        virtual const Type& GetType() const override;
       
        // Get the size of external memory.
        virtual size_t GetExternalSize() const override;

        const FunctionInfo& GetFuncInfo() const;

        // unsigned char* SerializeBinary(unsigned char* p_buffer) const;

        // size_t GetSerializedSize() const;

        // Compares this ExecutableImpl against another one.
        bool operator==(const LlvmExecutableImpl& p_other) const;

        // static LlvmExecutableImpl* DeserializeBinary(unsigned char* p_binary, const boost::shared_ptr<CodeHeap>& p_codeHeap, size_t p_codeSize);

    private:

        // Function to support aggregated freefom.
        template<typename ReturnType>
        ReturnType EvaluateInternal(ReturnType (*p_fun)(const Executable::FeatureType* const* p_features,
                                                        UInt32 p_currentDocument,
                                                        UInt32 p_documentCount,
                                                        Int64* p_cache), 
                                    const Executable::FeatureType* const* p_features,
                                    UInt32 p_currentDocument,
                                    UInt32 p_documentCount,
                                    Int64* p_cache, 
                                    const char* p_sourceFile, 
                                    unsigned int p_sourceLine) const;

        // Function to take flattened array results of templated type, and reify
        // them into a multi-dimensional array result structure.
        template<typename T>
        std::pair<ArrayCodeGen::ArrayBoundsType, boost::shared_array<T>>
        EvaluateArray(StreamFeatureInput* p_input,
                      const Executable::FeatureType p_features[],
                      const char* p_sourceFile, 
                      unsigned int p_sourceLine) const;

        template<typename ReturnType, typename ArrayArgType>
        ReturnType EvaluateInternal(ReturnType (*p_fun)(StreamFeatureInput* p_input,
                                                        const Executable::FeatureType p_features[], 
                                                        ArrayArgType* p_arraySpace), 
                                    StreamFeatureInput* p_input,
                                    const Executable::FeatureType p_features[],
                                    ArrayArgType* p_arraySpace, 
                                    const char* p_sourceFile, 
                                    unsigned int p_sourceLine) const;

        // Top-level type of the program.
        Type m_type;

        // Generic pointer to generated function.
        void* m_fun;

        // JIT memory manager that owns the memory pointed to by m_fun.
        boost::shared_ptr<llvm::JITMemoryManager> m_memoryManager;

        // Feature map used to compile program.
        DynamicRank::IFeatureMap& m_map;

        // Destination function type.
        CompilerFactory::DestinationFunctionType m_destinationFunctionType;

        // Program compilation result descriptor. Used by 1st ctor only.
        FunctionInfo m_funcInfo;

        // Executable relocation data. Used by 1st ctor only.
        std::vector<RelocationInfo> m_relocations;
    };
}
