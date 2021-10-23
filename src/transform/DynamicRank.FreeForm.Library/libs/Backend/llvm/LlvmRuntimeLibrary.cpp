#include "LlvmRuntimeLibrary.h"

#include <basic_types.h>
#include <cstdlib>
#include <FreeForm2Assert.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <forward_list>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <map>
#include <random>
#include <string>
// #include <Windows.h>
#include <mutex>
#include <time.h>

using namespace FreeForm2;

namespace FreeForm2
{
    // static CRITSEC s_randLock;
    static std::mutex s_randLock; 
    static std::minstd_rand0 s_randGenerator;
    static bool s_isRandSeeded = false;
    static std::uniform_real_distribution<> s_randDistribution;
}

    // Get a random number in the range [0, 1.0). This function uses a simple 
    // LCPRNG.
extern "C" double FreeForm2GetRandomValue()
{
    // ::AutoCriticalSection lock(&FreeForm2::s_randLock);
    s_randLock.lock();
    if (!FreeForm2::s_isRandSeeded)
    {
        FreeForm2::s_randGenerator.seed(GetTickCount());
        FreeForm2::s_isRandSeeded = true;
    }
    double value = FreeForm2::s_randDistribution(FreeForm2::s_randGenerator);
    s_randLock.unlock();
    return value;  
    // return FreeForm2::s_randDistribution(FreeForm2::s_randGenerator);
}

namespace
{

    struct GlobalEntry
    {
        // Create a GlobalEntry for a name, GlobalValue, and mapping tuple.
        GlobalEntry(const char* p_name, llvm::GlobalValue* p_value, void* p_mapping)
            : m_name(p_name),
              m_value(p_value),
              m_mapping(p_mapping)
        {
        }

        // The name of the global.
        std::string m_name;

        // The actual llvm value type.
        llvm::GlobalValue* m_value;

        // The pointer to which the value will be mapped.
        void* m_mapping;
    };

    // Create a ValueAndMapping function for std::rand.
    GlobalEntry CreateRand(llvm::LLVMContext& p_context)
    {
        llvm::Type* randRet = llvm::Type::getDoubleTy(p_context); 
        FF2_ASSERT(randRet->getPrimitiveSizeInBits() == sizeof(decltype(FreeForm2GetRandomValue())) * 8);
        llvm::FunctionType* randSig = llvm::FunctionType::get(randRet, false);
        llvm::Function* randFunc = llvm::Function::Create(randSig,
                                                          llvm::GlobalValue::ExternalLinkage,
                                                          llvm::Twine("rand"));
        return GlobalEntry("rand", randFunc, &FreeForm2GetRandomValue);
    }

    // Compare SIZED_STRING values for a less-than ordering.
    struct SizedStrLess
    {
        bool operator()(SIZED_STRING p_left, SIZED_STRING p_right) const
        {
            int val = std::char_traits<char>::compare(p_left.pcData, 
                                                      p_right.pcData,
                                                      std::min(p_left.cbData, p_right.cbData));
            if (!val)
            {
                return p_left.cbData < p_right.cbData;
            }
            else
            {
                return val < 0;
            }
        }
    };
}

class LlvmRuntimeLibrary::Impl final
{
public:
    Impl(llvm::LLVMContext& p_context)
        : m_context(p_context)
    {
        Initialize();
    }

    // Delete copy constructor and assignment operator.
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    // Add all runtime symbols to the specified module. This method implements
    // LlvmRuntimeLibrary::AddLibraryToModule.
    void AddLibraryToModule(llvm::Module& p_module) const
    {
        for (const auto& entry : m_globals)
        {
            llvm::GlobalValue* value = entry.second.first;
            if (llvm::Function* func = llvm::dyn_cast<llvm::Function>(value))
            {
                p_module.getFunctionList().push_back(func);
            }
            else if (llvm::GlobalVariable* var = llvm::dyn_cast<llvm::GlobalVariable>(value))
            {
                p_module.getGlobalList().push_back(var);
            }
            else
            {
                FF2_UNREACHABLE();
            }
        }
    }

    // Add global value mappings to an exeuction engine. This method implements
    // LlvmRuntimeLibrary::AddExecutionMappings.
    void AddExecutionMappings(llvm::ExecutionEngine& p_engine) const
    {
        for (const auto& entry : m_globals)
        {
            const llvm::GlobalValue* const value = entry.second.first;
            void* const mapping = entry.second.second;
            p_engine.updateGlobalMapping(value, mapping);
        }
    }

    // Look up a GlobalValue by name. This implements 
    // LlvmRuntimeLibrary::FundValue.
    llvm::GlobalValue* FindValue(SIZED_STRING p_name) const
    {
        const auto find = m_globals.find(p_name);
        if (find == m_globals.end())
        {
            return nullptr;
        }
        else
        {
            const GlobalAndMapping& value = find->second;
            return value.first;
        }
    }

    // Find a runtime function with the specified name. This implements
    // LlvmRuntimeLibrary::FindFunction.
    llvm::Function* FindFunction(SIZED_STRING p_name) const
    {
        return llvm::dyn_cast_or_null<llvm::Function>(FindValue(p_name));
    }

private:
    // Initialize the+ runtime library.
    void Initialize();

    // Add a GlobalEntry to the mapping structures.
    void AddEntry(const GlobalEntry& p_entry);

    // A pair containing the LLVM GlobalValue and the mapping pointer.
    typedef std::pair<llvm::GlobalValue*, void*> GlobalAndMapping;

    // A mapping of global name to GlobalEntry struct.
    std::map<SIZED_STRING, GlobalAndMapping, SizedStrLess> m_globals;

    // The storage for names of the globals.
    std::forward_list<std::string> m_globalNames;

    // The LLVMContext passed to the constructor.
    llvm::LLVMContext& m_context;
};


void LlvmRuntimeLibrary::Impl::Initialize()
{
    AddEntry(CreateRand(m_context));
}


void LlvmRuntimeLibrary::Impl::AddEntry(const GlobalEntry& p_entry)
{
    m_globalNames.push_front(p_entry.m_name);
    const std::string& name = m_globalNames.front();
    const SIZED_STRING sizedName = CStackSizedString(name.c_str(), name.size());
    
    auto pairIterBool
        = m_globals.emplace(sizedName, GlobalAndMapping(p_entry.m_value, p_entry.m_mapping));
    FF2_ASSERT(pairIterBool.second && "Global already exists");
}


LlvmRuntimeLibrary::LlvmRuntimeLibrary(llvm::LLVMContext& p_context)
    : m_impl(new Impl(p_context))
{
}


// This empty destructor is required so that the compiled can find Impl::~Impl.
// If it is not explicitly defined in this file, the compiler will not call
// the Impl destructor and will issue a warning.
LlvmRuntimeLibrary::~LlvmRuntimeLibrary()
{
}


void
LlvmRuntimeLibrary::AddLibraryToModule(llvm::Module& p_module) const
{
    m_impl->AddLibraryToModule(p_module);
}


void
LlvmRuntimeLibrary::AddExecutionMappings(llvm::ExecutionEngine& p_engine) const
{
    m_impl->AddExecutionMappings(p_engine);
}


llvm::GlobalValue*
LlvmRuntimeLibrary::FindValue(SIZED_STRING p_name) const
{
    return m_impl->FindValue(p_name);
}


llvm::Function*
LlvmRuntimeLibrary::FindFunction(SIZED_STRING p_name) const
{
    return m_impl->FindFunction(p_name);
}
