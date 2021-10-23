#include "LlvmCompiler.h"

#include "ArrayCodeGen.h"
#include "ArrayType.h"
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include <cmath>
#include "CompilationState.h"
#include "Executable.h"
#include "FreeForm2Assert.h"
#include "FreeForm2Type.h"
#include "FreeForm2Utils.h"
#include "LlvmCodeGenUtils.h"
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/PassManager.h>
#include <llvm/ADT/APInt.h>
#include <llvm/IR/Verifier.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/GenericValue.h>
#include <llvm/ExecutionEngine/JIT.h>
#include <llvm/ExecutionEngine/JITMemoryManager.h>
#include <llvm/PassManager.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include "Program.h"
#include <sstream>
#include "ValueResult.h"
#include <mutex>
#include "FreeForm2Support.h"

using namespace FreeForm2;

#define __stdcall

// Chkstk method signature needed to identify __chkstk call.
// This appears within low-level IR if the Freeform expression incurs
// a large amount of stack usage.
// extern "C" void __stdcall __chkstk(size_t);


// Copied from LLVM code.
namespace X86
{
    /// RelocationType - An enum for the x86 relocation codes. Note that
    /// the terminology here doesn't follow x86 convention - word means
    /// 32-bit and dword means 64-bit. The relocations will be treated
    /// by JIT or ObjectCode emitters, this is transparent to the x86 code
    /// emitter but JIT and ObjectCode will treat them differently
    enum RelocationType {
        /// reloc_pcrel_word - PC relative relocation, add the relocated value to
        /// the value already in memory, after we adjust it for where the PC is.
        reloc_pcrel_word = 0,

        /// reloc_picrel_word - PIC base relative relocation, add the relocated
        /// value to the value already in memory, after we adjust it for where the
        /// PIC base is.
        reloc_picrel_word = 1,

        /// reloc_absolute_word - absolute relocation, just add the relocated
        /// value to the value already in memory.
        reloc_absolute_word = 2,

        /// reloc_absolute_word_sext - absolute relocation, just add the relocated
        /// value to the value already in memory. In object files, it represents a
        /// value which must be sign-extended when resolving the relocation.
        reloc_absolute_word_sext = 3,

        /// reloc_absolute_dword - absolute relocation, just add the relocated
        /// value to the value already in memory.
        reloc_absolute_dword = 4
    };
}

namespace
{
    using namespace FreeForm2;

    // Table of external functions to be called from LLVM-generated code.
    static const void* s_externalFunctions[] =
    {
        pow, //std::powf,
        log, //std::logf,
        floor, //std::floorf,
        ceil, //std::ceilf,
        fmod, //std::fmodf,
        tanh, //std::tanhf,
        std::rand,
        // __chkstk,
        FreeForm2GetRandomValue
    };

    // Table of names for external functions (must match size and order of previous table)
    // The callers use these s_externalNames to match s_externalFunctions. Any rename may result in failure in external function matching.
    static const char* s_externalNames[] =
    {
        "powf",
        "logf",
        "floorf",
        "ceilf",
        "fmodf",
        "tanhf",
        "rand",
        // "__chkstk",
        "FreeForm2GetRandomValue"
    };

    void ConvertRelocations(std::vector<LlvmExecutableImpl::RelocationInfo>& p_dest,
                            const std::vector<llvm::MachineRelocation>& p_src,
                            const LlvmExecutableImpl::FunctionInfo& p_func,
                            const std::vector<void*>& p_externals)
    {
        for (const auto& mr : p_src)
        {
            LlvmExecutableImpl::RelocationInfo relocation = { 0, 0, 0 };
            relocation.m_offset = static_cast<uint32_t>(mr.getMachineCodeOffset());
            FF2_ASSERT(relocation.m_offset < p_func.m_length);
            if (mr.getRelocationType() == X86::reloc_pcrel_word)
            {
                // Jump relative to PC, need no relocation
                continue;
            }
            FF2_ASSERT(mr.getRelocationType() == X86::reloc_absolute_dword);
            uint64_t relocated = *(reinterpret_cast<const uint64_t*>(p_func.m_start + relocation.m_offset));
            const uint8_t* result = reinterpret_cast<uint8_t*>(mr.getResultPointer());

            if (p_func.m_start <= result && result < p_func.m_start + p_func.m_length)
            {
                relocation.m_type = LlvmExecutableImpl::RelocationInfo::Internal;
                relocation.m_delta = static_cast<uint32_t>(reinterpret_cast<uint8_t*>(relocated) - p_func.m_start);
            }
            else
            {
                const void* pFreeForm2GetRandomValue = reinterpret_cast<const void*>(&FreeForm2GetRandomValue);
                relocation.m_type = LlvmExecutableImpl::RelocationInfo::External;
                const void* relocatedFun = reinterpret_cast<const void*>(relocated);
                const auto found = std::find(p_externals.cbegin(), p_externals.cend(), relocatedFun);
                if (found != p_externals.cend())
                {
                    relocation.m_delta = static_cast<uint32_t>(std::distance(p_externals.cbegin(), found));
                }
                else
                {
                    FF2_ASSERT(relocatedFun == pFreeForm2GetRandomValue);
                    relocation.m_delta = static_cast<uint32_t>(p_externals.size() - 1);
                }
            }

            p_dest.push_back(relocation);
        }
    }
}

namespace FreeForm2
{
    // Wrapper class around LLVM default memory, that allows us to persist the
    // memory manager beyond the lifetime of the owning module.
    class PersistentJITMemoryManager : public llvm::JITMemoryManager
    {
    public:
        PersistentJITMemoryManager()
            : m_manager(llvm::JITMemoryManager::CreateDefaultMemManager())
        {
            m_fun.m_start = nullptr;
            m_fun.m_length = 0;
        }


        // Return the delegated manager.
        boost::shared_ptr<llvm::JITMemoryManager> 
        GetDelegate()
        {
            return m_manager;
        }

        // Implement all virtual functions declared by JITMemoryManager by
        // passing them to the delegate.


        virtual void 
        setMemoryWritable() override
        {
            return m_manager->setMemoryWritable();
        }


        virtual void 
        setMemoryExecutable() override
        {
            return m_manager->setMemoryExecutable();
        }


        virtual void 
        setPoisonMemory(bool p_poison) override
        {
            return m_manager->setPoisonMemory(p_poison);
        }


        virtual void 
        AllocateGOT() override
        {
            return m_manager->AllocateGOT();
        }


        virtual uint8_t*
        getGOTBase() const override
        {
            return m_manager->getGOTBase();
        }


        virtual uint8_t*
        startFunctionBody(const llvm::Function* p_f, uintptr_t& p_actualSize) override
        {
            return m_manager->startFunctionBody(p_f, p_actualSize);
        }


        virtual uint8_t*
        allocateStub(const llvm::GlobalValue* p_f, unsigned p_stubSize, unsigned p_alignment) override
        {
            return m_manager->allocateStub(p_f, p_stubSize, p_alignment);
        }


        virtual void 
        endFunctionBody(const llvm::Function* p_f, uint8_t* p_functionStart, uint8_t* p_functionEnd) override
        {
            m_fun.m_start = p_functionStart;
            m_fun.m_length = p_functionEnd - p_functionStart;
            return m_manager->endFunctionBody(p_f, p_functionStart, p_functionEnd);
        }


        virtual uint8_t*
        allocateSpace(intptr_t p_size, unsigned p_alignment) override 
        {
            return m_manager->allocateSpace(p_size, p_alignment);
        }


        virtual uint8_t*
        allocateGlobal(uintptr_t p_size, unsigned p_alignment) override
        {
            return m_manager->allocateGlobal(p_size, p_alignment);
        }


        virtual void 
        deallocateFunctionBody(void* p_body) override
        {
            return m_manager->deallocateFunctionBody(p_body);
        }

        virtual uint8_t*
        allocateCodeSection(uintptr_t p_size, unsigned p_alignment, unsigned p_sectionID, llvm::StringRef p_sectionName) override
        {
            return m_manager->allocateCodeSection(p_size, p_alignment, p_sectionID, p_sectionName);
        }

        virtual uint8_t*
        allocateDataSection(uintptr_t p_size, unsigned p_alignment, unsigned p_sectionID, llvm::StringRef p_sectionName, bool p_isReadOnly) override
        {
            return m_manager->allocateDataSection(p_size, p_alignment, p_sectionID, p_sectionName, p_isReadOnly);
        }

        virtual bool
        finalizeMemory(std::string* p_errMsg) override
        {
            return m_manager->finalizeMemory(p_errMsg);
        }

        // Function length and size.
        LlvmExecutableImpl::FunctionInfo m_fun;

    private:

        // Delegate manager.
        boost::shared_ptr<llvm::JITMemoryManager> m_manager;
    };

    // Critical section used to single-thread entry to LLVM code generation.
    // CRITSEC s_llvmCriticalSection;
    std::mutex s_llvmCriticalSection;
    bool s_use_llvmCriticalSection = true;
}

class ConditionalAutoCriticalSection //: public INoHeapInstance
{
private:
    // CRITSEC* m_pCritSec;
    std::mutex* m_pCritSec;
    bool m_yes = true;

    ConditionalAutoCriticalSection(const ConditionalAutoCriticalSection&) = delete;
    ConditionalAutoCriticalSection& operator=(const ConditionalAutoCriticalSection&) = delete;

public:
    // ConditionalAutoCriticalSection(CRITSEC* pCritSec, bool pYes = true) 
    ConditionalAutoCriticalSection(std::mutex* pCritSec, bool pYes = true) 
        : m_pCritSec(pCritSec)
        , m_yes(pYes)
    {
        if (m_yes)
        {
            this->m_pCritSec->lock();
        }
    }

    ~ConditionalAutoCriticalSection()
    {
        if (m_yes)
        {
            this->m_pCritSec->unlock();
        }
    }
};

FreeForm2::LlvmCompilerImpl::LlvmCompilerImpl(unsigned int p_optimizationLevel,
                                              CompilerFactory::DestinationFunctionType p_destinationFunctionType)
    : m_optimizationLevel(p_optimizationLevel),
      m_destinationFunctionType(p_destinationFunctionType),
      m_persistentMemoryManager(nullptr)
{
    ConditionalAutoCriticalSection cs(&s_llvmCriticalSection, s_use_llvmCriticalSection);
    llvm::InitializeNativeTarget();

    std::unique_ptr<PersistentJITMemoryManager> persistent(new PersistentJITMemoryManager());
    m_persistentMemoryManager = persistent.get();

    m_state.reset(new CompilationState(*persistent));

    // Arrange our memory managers, so that the persistent memory
    // manager is owned by the module, and we have a shared pointer to
    // the delegated manager.
    m_memoryManager = persistent->GetDelegate();
    persistent.release();

    // Initialize and run a function pass manager, for optimization.
    m_functionPassManager.reset(new llvm::FunctionPassManager(&m_state->GetModule()));

    llvm::PassManagerBuilder builder;
    builder.OptLevel = p_optimizationLevel;
    builder.populateFunctionPassManager(*m_functionPassManager);

    m_functionPassManager->doInitialization();
}


FreeForm2::LlvmCompilerImpl::~LlvmCompilerImpl()
{
    ConditionalAutoCriticalSection cs(&s_llvmCriticalSection, s_use_llvmCriticalSection);

    // Reset the state variables inside the critical section.
    m_memoryManager.reset();
    //m_persistentMemoryManager.release();
    m_functionPassManager.reset();
    m_state.reset();
}


CompilationState&
LlvmCompilerImpl::GetState()
{
    return *m_state;
}


boost::shared_ptr<llvm::JITMemoryManager>
LlvmCompilerImpl::GetMemoryManager()
{
    return m_memoryManager;
}


llvm::FunctionPassManager&
LlvmCompilerImpl::GetFunctionPassManager()
{
    return *m_functionPassManager;
}


unsigned int
LlvmCompilerImpl::GetOptimizationLevel() const
{
    return m_optimizationLevel;
}


const PersistentJITMemoryManager&
LlvmCompilerImpl::GetPersistentMemoryManager() const
{
    FF2_ASSERT(m_persistentMemoryManager != nullptr);
    return *m_persistentMemoryManager;
}


std::unique_ptr<FreeForm2::CompilerResults>
FreeForm2::LlvmCompilerImpl::Compile(const ProgramImpl& p_program, bool p_debugOutput)
{
    std::auto_ptr<ExecutableImpl> execImpl(new LlvmExecutableImpl(*this,
        p_program,
        p_debugOutput,
        m_destinationFunctionType));
    auto exec = boost::make_shared<Executable>(execImpl);

    return std::unique_ptr<CompilerResults>(new ExecutableCompilerResults(exec));
}


FreeForm2::LlvmExecutableImpl::LlvmExecutableImpl(FreeForm2::LlvmCompilerImpl& p_compiler,
                                                  const FreeForm2::ProgramImpl& p_program, 
                                                  bool p_dumpExecutable,
                                                  CompilerFactory::DestinationFunctionType p_destinationFunctionType)
    : m_type(p_program.GetType().GetImplementation()),
      m_map(p_program.GetFeatureMap()),
      m_destinationFunctionType(p_destinationFunctionType)
{
    // Single-thread access to LLVM, pending work to allow
    // safe-multithreaded access.
    ConditionalAutoCriticalSection cs(&s_llvmCriticalSection, s_use_llvmCriticalSection);

    CompilationState& state = p_compiler.GetState();

    llvm::Function& fun = LlvmCodeGenerator::Compile(p_program.GetExpression(),
                                                     state,
                                                     p_program.GetAllocations(),
                                                     m_destinationFunctionType);

    if (p_dumpExecutable)
    {
        state.GetModule().dump();
    }

    // Verify the function.  Note that, despite the name, verifyFunction
    // returns true if the function is corrupt.
    std::string verifierMessage;
    llvm::raw_string_ostream messageStream(verifierMessage);
    if (llvm::verifyFunction(fun, &messageStream))
    {
        // Print the LLVM IR for the module.
        std::string module;
        llvm::raw_string_ostream moduleStream(module);
        state.GetModule().print(moduleStream, NULL);
        std::ostringstream err;
        err << "Error verifying LLVM function ('" << verifierMessage 
            << "'): " << std::endl << moduleStream.str() << std::endl;
        std::cout << err.str() << std::endl;
        throw std::runtime_error(err.str());
    }

    p_compiler.GetFunctionPassManager().run(fun);

    if (p_compiler.GetOptimizationLevel() > 0 && p_dumpExecutable)
    {
        state.GetModule().dump();
    }

    // Get pointer to function for later execution.
    m_fun = state.GetExecutionEngine().getPointerToFunction(&fun);

    m_memoryManager = p_compiler.GetMemoryManager();
    m_funcInfo = p_compiler.GetPersistentMemoryManager().m_fun;
    
    std::vector<void*> externals;
    for (const auto& name : s_externalNames)
    {
        externals.push_back(state.GetExecutionEngine().getPointerToNamedFunction(name, false));
    }

    ConvertRelocations(m_relocations,
                       llvm::GetMachineRelocations(&state.GetExecutionEngine()),
                       GetFuncInfo(),
                       externals);
}


FreeForm2::LlvmExecutableImpl::~LlvmExecutableImpl()
{
    ConditionalAutoCriticalSection cs(&s_llvmCriticalSection, s_use_llvmCriticalSection);

    // Reset the memory manager inside the critical section.
    m_memoryManager.reset();
}


boost::shared_ptr<Result> 
FreeForm2::LlvmExecutableImpl::Evaluate(StreamFeatureInput* p_input,
                                        const Executable::FeatureType p_features[]) const
{
    FF2_ASSERT(m_destinationFunctionType == FreeForm2::CompilerFactory::SingleDocumentEvaluation);
    switch (m_type.Primitive())
    {
        // Return Result::IntType for any integer result for Phoenix compatibility.
        case Type::UInt32: __attribute__((__fallthrough__));
        case Type::Int32: __attribute__((__fallthrough__));
        case Type::Int:
        {
            typedef Result::IntType (*TypedFun)(StreamFeatureInput* p_input,
                                                const Executable::FeatureType*, 
                                                Result::IntType*);
            TypedFun fun = reinterpret_cast<TypedFun>(m_fun);
            return boost::shared_ptr<Result>(
                new ValueResult(EvaluateInternal<Result::IntType, Result::IntType>(
                    fun, p_input, p_features, NULL, __FILE__, __LINE__)));
        }

        case Type::Float:
        {
            typedef Result::FloatType (*TypedFun)(StreamFeatureInput* p_input,
                                                  const Executable::FeatureType*, 
                                                  Result::IntType*);
            TypedFun fun = reinterpret_cast<TypedFun>(m_fun);
            return boost::shared_ptr<Result>(
                new ValueResult(EvaluateInternal<Result::FloatType, Result::IntType>(
                    fun, p_input, p_features, NULL, __FILE__, __LINE__)));
        }

        case Type::Bool:
        {
            typedef bool (*TypedFun)(StreamFeatureInput* p_input,
                                     const Executable::FeatureType*, 
                                     Result::IntType*);
            TypedFun fun = reinterpret_cast<TypedFun>(m_fun);
            return boost::shared_ptr<Result>(
                new ValueResult(EvaluateInternal<bool, Result::IntType>(
                    fun, p_input, p_features, NULL, __FILE__, __LINE__)));
        }

        case Type::Array:
        {
            const ArrayType& arrayType = static_cast<const ArrayType&>(m_type.GetImplementation());
            switch (arrayType.GetChildType().Primitive())
            {
                case Type::Int:
                {
                    std::pair<ArrayCodeGen::ArrayBoundsType, 
                              boost::shared_array<Result::IntType>> 
                        ret = EvaluateArray<Result::IntType>(p_input,
                                                             p_features, 
                                                             __FILE__, 
                                                             __LINE__);
                    return ArrayCodeGen::CreateArrayResult<Result::IntType>(arrayType, 
                                                                            ret.first, 
                                                                            ret.second);
                }

                case Type::Float:
                {
                    std::pair<ArrayCodeGen::ArrayBoundsType, 
                              boost::shared_array<Result::FloatType>> 
                        ret = EvaluateArray<Result::FloatType>(p_input,
                                                               p_features, 
                                                               __FILE__, 
                                                               __LINE__);
                    return ArrayCodeGen::CreateArrayResult<Result::FloatType>(arrayType,
                                                                              ret.first, 
                                                                              ret.second);
                }

                case Type::Bool:
                {
                    std::pair<ArrayCodeGen::ArrayBoundsType, 
                              boost::shared_array<bool>> 
                        ret = EvaluateArray<bool>(p_input, p_features, __FILE__, __LINE__);
                    return ArrayCodeGen::CreateArrayResult<bool>(arrayType,
                                                                 ret.first, 
                                                                 ret.second);
                }

                default:
                {
                    Unreachable(__FILE__, __LINE__);
                }
            }
            break;
        }

        default:
        {
            Unreachable(__FILE__, __LINE__);
        }
    }
}


boost::shared_ptr<Result>
FreeForm2::LlvmExecutableImpl::Evaluate(const Executable::FeatureType* const* p_features,
                                        UInt32 p_currentDocument,
                                        UInt32 p_documentCount,
                                        Int64* p_cache) const 
{
    FF2_ASSERT(m_destinationFunctionType == FreeForm2::CompilerFactory::DocumentSetEvaluation);
    switch (m_type.Primitive())
    {
        case Type::Int:
        case Type::Int32:
        case Type::UInt32:
        {
            typedef Result::IntType (*TypedFun) (const Executable::FeatureType* const*,
                                                 UInt32,
                                                 UInt32,
                                                 Int64*);
            TypedFun fun = reinterpret_cast<TypedFun>(m_fun);
            return boost::shared_ptr<Result>(
                new ValueResult(EvaluateInternal<Result::IntType>(
                    fun, p_features, p_currentDocument, p_documentCount, p_cache, __FILE__, __LINE__)));
        }

        case Type::Float:
        {
            typedef Result::FloatType (*TypedFun) (const Executable::FeatureType* const*,
                                                   UInt32,
                                                   UInt32,
                                                   Int64*);
            TypedFun fun = reinterpret_cast<TypedFun>(m_fun);
            return boost::shared_ptr<Result>(
                new ValueResult(EvaluateInternal<Result::FloatType>(
                    fun, p_features, p_currentDocument, p_documentCount, p_cache, __FILE__, __LINE__)));
        }

        case Type::Bool:
        {
            typedef bool (*TypedFun) (const Executable::FeatureType* const*,
                                      UInt32,
                                      UInt32,
                                      Int64*);
            TypedFun fun = reinterpret_cast<TypedFun>(m_fun);
            return boost::shared_ptr<Result>(
                new ValueResult(EvaluateInternal<bool>(
                    fun, p_features, p_currentDocument, p_documentCount, p_cache, __FILE__, __LINE__)));
        }

        default:
        {
            Unreachable(__FILE__, __LINE__);
        }
    }
}


Executable::DirectEvalFun 
FreeForm2::LlvmExecutableImpl::EvaluationFunction() const
{
    // DirectEvalFun assumes float return for freeform2 float type.
    BOOST_STATIC_ASSERT(sizeof(Result::FloatType) == sizeof(float));
    if (m_type.Primitive() == Type::Float)
    {
        return reinterpret_cast<Executable::DirectEvalFun>(m_fun);
    }
    else
    {
        return NULL;
    }
}


Executable::AggregatedEvalFun
FreeForm2::LlvmExecutableImpl::AggregatedEvaluationFunction() const
{
    FF2_ASSERT(m_destinationFunctionType == FreeForm2::CompilerFactory::DocumentSetEvaluation);
    // Aggregated EvalFun assumes float return for freeform2 float type.
    BOOST_STATIC_ASSERT(sizeof(Result::FloatType) == sizeof(float));
    if (m_type.Primitive() == Type::Float)
    {
        return reinterpret_cast<FreeForm2::Executable::AggregatedEvalFun>(m_fun);
    }
    else
    {
        return nullptr;
    }
}


const Type&
FreeForm2::LlvmExecutableImpl::GetType() const
{
    return m_type;
}


// Get the size of external memory.
size_t
FreeForm2::LlvmExecutableImpl::GetExternalSize() const
{
    if (m_memoryManager.get())
    {
        // This will give us an approximation of the memory allocated by LLVM, albeit one that is almost certainly correct for our purposes. 
        const size_t sizeOfMemoryAllocatedByLlvm = (m_memoryManager->GetNumCodeSlabs() * m_memoryManager->GetDefaultCodeSlabSize())
                                                   + (m_memoryManager->GetNumDataSlabs() * m_memoryManager->GetDefaultDataSlabSize());

        // JITMemoryManager is a shared resource by all executables (an executable belongs to one neural input).
        // Each executable will report an equal part of the shared memory.
        return (sizeOfMemoryAllocatedByLlvm + sizeof(llvm::JITMemoryManager)) / m_memoryManager.use_count();
    }
    else
    {
        return sizeof(llvm::JITMemoryManager);
    }
}


template<typename ReturnType>
ReturnType
FreeForm2::LlvmExecutableImpl::EvaluateInternal(ReturnType(*p_fun)(const Executable::FeatureType* const* p_features,
                                                                  UInt32 p_currentDocument,
                                                                  UInt32 p_documentCount,
                                                                  Int64* p_cache),
                                                const Executable::FeatureType* const* p_features,
                                                UInt32 p_currentDocument,
                                                UInt32 p_documentCount,
                                                Int64* p_cache,
                                                const char* p_sourceFile,
                                                unsigned int p_sourceLine) const
{
    return p_fun(p_features, p_currentDocument, p_documentCount, p_cache);
}


// Function to take flattened array results of templated type, and reify
// them into a multi-dimensional array result structure.
template<typename T>
std::pair<ArrayCodeGen::ArrayBoundsType, boost::shared_array<T>>
FreeForm2::LlvmExecutableImpl::EvaluateArray(StreamFeatureInput* p_input,
                                             const Executable::FeatureType p_features[],
                                             const char* p_sourceFile, 
                                             unsigned int p_sourceLine) const
{
    typedef ArrayCodeGen::ArrayBoundsType (*TypedFun)(StreamFeatureInput*, 
                                                      const Executable::FeatureType*, 
                                                      T*);
    TypedFun fun = reinterpret_cast<TypedFun>(m_fun);
    const ArrayType& arrayType = static_cast<const ArrayType&>(m_type.GetImplementation());
    boost::shared_array<T> space(new T[arrayType.GetMaxElements()]);
    return std::make_pair(EvaluateInternal(fun, 
                                           p_input,
                                           p_features, 
                                           space.get(), 
                                           p_sourceFile, 
                                           p_sourceLine),
                          space);
}


template<typename ReturnType, typename ArrayArgType>
ReturnType
FreeForm2::LlvmExecutableImpl::EvaluateInternal(ReturnType (*p_fun)(StreamFeatureInput* p_input,
                                                            const Executable::FeatureType p_features[], 
                                                            ArrayArgType* p_arraySpace), 
                                                StreamFeatureInput* p_input,
                                                const Executable::FeatureType p_features[],
                                                ArrayArgType* p_arraySpace, 
                                                const char* p_sourceFile, 
                                                unsigned int p_sourceLine) const
{
    try
    {
        return p_fun(p_input, p_features, p_arraySpace);
    }
    catch(std::exception e){

    }
    Unreachable(__FILE__, __LINE__);
}


const LlvmExecutableImpl::FunctionInfo&
LlvmExecutableImpl::GetFuncInfo() const
{
    return m_funcInfo;
}


bool
LlvmExecutableImpl::operator==(const FreeForm2::LlvmExecutableImpl& p_other) const
{
    // TODO: implement correct equality check for executable code with respect to relocations.
    return true;
}
