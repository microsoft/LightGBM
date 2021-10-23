#include "CompilationState.h"

#include "ArrayCodeGen.h"
#include "ArrayType.h"
#include <boost/static_assert.hpp>
#include <cstdlib>
#include "Expression.h"
#include "FreeForm2Assert.h"
#include "FreeForm2Result.h"
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/GenericValue.h>
#include <llvm/ExecutionEngine/JIT.h>
#include <llvm/ExecutionEngine/JITMemoryManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/DataLayout.h>
#include "LlvmCodeGenUtils.h"
#include <sstream>

namespace
{
    std::auto_ptr<llvm::ExecutionEngine> 
    CreateEngine(llvm::Module& p_module, llvm::JITMemoryManager& p_memoryManager)
    {
        std::string builderErr;
        llvm::EngineBuilder engineBuilder(&p_module);
        engineBuilder.setJITMemoryManager(&p_memoryManager);
        engineBuilder.setEngineKind(llvm::EngineKind::JIT);
        engineBuilder.setErrorStr(&builderErr);
        std::auto_ptr<llvm::ExecutionEngine> engine(engineBuilder.create());
        if (engine.get() == NULL)
        {
            std::ostringstream err;
            err << "JIT builder error: " << builderErr;
            throw std::runtime_error(err.str());
        }
        return engine;
    }
}


FreeForm2::CompilationState::CompilationState(llvm::JITMemoryManager& p_memoryManager)
    : m_context(new llvm::LLVMContext()),
      m_runtimeLibrary(new LlvmRuntimeLibrary(*m_context)),
      m_builder(new llvm::IRBuilder<>(*m_context)),
      m_intBits(sizeof(Result::IntType) * 8),
      m_arrayBoundsType(llvm::IntegerType::get(*m_context, 
                                               sizeof(ArrayCodeGen::ArrayBoundsType) * 8)), 
      m_arrayCountType(llvm::IntegerType::get(*m_context, 
                                              sizeof(ArrayCodeGen::ArrayCountType) * 8)), 
      m_boolType(llvm::IntegerType::get(*m_context, 1)), 
      m_arrayBoolType(&CreateArrayType(m_boolType)), 
      m_intType(llvm::IntegerType::get(*m_context, m_intBits)),
      m_arrayIntType(&CreateArrayType(m_intType)),
      m_int32Type(llvm::IntegerType::get(*m_context, sizeof(Result::Int32Type) * 8)),
      m_arrayInt32Type(&CreateArrayType(m_int32Type)),
      m_floatType(llvm::Type::getFloatTy(*m_context)),
      m_floatPtrType(llvm::Type::getFloatPtrTy(*m_context)),
      m_arrayFloatType(&CreateArrayType(m_floatType)), 

      // Use a two-bit integer to represent the freeform void type in LLVM.  We
      // can't use the LLVM void type because LLVM doesn't have a value that
      // corresponds to that (is natively imperative), whereas we're taking the
      // approach that void exists to remove the distinction between expressions
      // and statements (convert-to-imperative approach).  As a result, we rely
      // on being able to instantiate void values, though they should be
      // immediately discarded by the underlying backend (and we check this).
      m_voidType(llvm::IntegerType::get(*m_context, 2)),

      m_featureType(llvm::IntegerType::get(*m_context, sizeof(Expression::FeatureType) * 8)), 
      m_featureArgument(nullptr),
      m_arraySpace(nullptr),
      m_previousOffset(nullptr),
      m_aggregatedDocumentCount(nullptr),
      m_aggregatedDocumentIndex(nullptr),
      m_aggregatedCache(nullptr)
{
    BOOST_STATIC_ASSERT(sizeof(Result::FloatType) == sizeof(float));

    CHECK_LLVM_RET(m_arrayBoundsType);
    CHECK_LLVM_RET(m_boolType);
    CHECK_LLVM_RET(m_intType);
    CHECK_LLVM_RET(m_floatType);
    CHECK_LLVM_RET(m_featureType);

    std::auto_ptr<llvm::Module> ownedModule(new llvm::Module("FreeForm2", *m_context));
    m_engine = CreateEngine(*ownedModule, p_memoryManager);
    m_module = ownedModule.release();
    m_targetData = m_engine->getDataLayout();
    InitializeRuntimeLibrary();
}


void
FreeForm2::CompilationState::InitializeRuntimeLibrary()
{
    m_runtimeLibrary->AddLibraryToModule(*m_module);
    m_runtimeLibrary->AddExecutionMappings(*m_engine);
}


llvm::IRBuilder<>& 
FreeForm2::CompilationState::GetBuilder()
{
    return *m_builder;
}


llvm::LLVMContext& 
FreeForm2::CompilationState::GetContext()
{
    return *m_context;
}


llvm::Module&
FreeForm2::CompilationState::GetModule()
{
    return *m_module;
}


llvm::ExecutionEngine&
FreeForm2::CompilationState::GetExecutionEngine()
{
    return *m_engine;
}


const FreeForm2::LlvmRuntimeLibrary&
FreeForm2::CompilationState::GetRuntimeLibrary() const
{
    return *m_runtimeLibrary;
}


void
FreeForm2::CompilationState::SetVariableValue(VariableID p_id, 
                                              llvm::Value& p_value)
{
    m_variables[p_id] = &p_value;
}


llvm::Value*
FreeForm2::CompilationState::GetVariableValue(VariableID p_id) const
{
    auto find = m_variables.find(p_id);
    FF2_ASSERT(find != m_variables.end());
    return find->second;
}


llvm::Value* 
FreeForm2::CompilationState::GetFeatureArgument() const
{
    FF2_ASSERT(m_featureArgument != NULL);
    return m_featureArgument;
}


void 
FreeForm2::CompilationState::SetFeatureArgument(llvm::Value& p_val)
{
    m_featureArgument = &p_val;
}


llvm::Value& 
FreeForm2::CompilationState::GetArrayReturnSpace() const
{
    FF2_ASSERT(m_arraySpace != NULL);
    return *m_arraySpace;
}


void 
FreeForm2::CompilationState::SetArrayReturnSpace(llvm::Value& p_value)
{
    m_arraySpace = &p_value;
}


llvm::Value&
FreeForm2::CompilationState::GetAggregatedDocumentCount() const
{
    return *m_aggregatedDocumentCount;
}


void
FreeForm2::CompilationState::SetAggregatedDocumentCount(llvm::Value& p_value)
{
    m_aggregatedDocumentCount = &p_value;
}


llvm::Value&
FreeForm2::CompilationState::GetAggregatedDocumentIndex() const
{
    return *m_aggregatedDocumentIndex;
}


void
FreeForm2::CompilationState::SetAggregatedDocumentIndex(llvm::Value& p_value)
{
    m_aggregatedDocumentIndex = &p_value;
}


llvm::Value&
FreeForm2::CompilationState::GetAggregatedCache() const
{
    return *m_aggregatedCache;
}


void
FreeForm2::CompilationState::SetAggregatedCache(llvm::Value& p_value)
{
    m_aggregatedCache = &p_value;
}


llvm::Value&
FreeForm2::CompilationState::GetFeatureArrayPointer() const
{
    return *m_featureArrayPointer;
}


void 
FreeForm2::CompilationState::SetFeatureArrayPointer(llvm::Value& p_value)
{
    m_featureArrayPointer = &p_value;
}


llvm::IntegerType& 
FreeForm2::CompilationState::GetArrayBoundsType() const
{
    return *m_arrayBoundsType;
}


llvm::IntegerType& 
FreeForm2::CompilationState::GetArrayCountType() const
{
    return *m_arrayCountType;
}


llvm::Type& 
FreeForm2::CompilationState::GetBoolType() const
{
    return *m_boolType;
}


llvm::IntegerType& 
FreeForm2::CompilationState::GetIntType() const
{
    return *m_intType;
}


unsigned int
FreeForm2::CompilationState::GetIntBits() const
{
    return m_intBits;
}


llvm::IntegerType&
FreeForm2::CompilationState::GetInt32Type() const
{
    return *m_int32Type;
}


llvm::Type& 
FreeForm2::CompilationState::GetFloatType() const
{
    return *m_floatType;
}


llvm::Type&
FreeForm2::CompilationState::GetFloatPtrType() const
{
    return *m_floatPtrType;
}


llvm::Type& 
FreeForm2::CompilationState::GetVoidType() const
{
    return *m_voidType;
}


llvm::Type& 
FreeForm2::CompilationState::GetFeatureType() const
{
    return *m_featureType;
}


unsigned int
FreeForm2::CompilationState::GetSizeInBytes(llvm::Type* p_type) const
{
    return static_cast<unsigned int>(m_targetData->getTypeAllocSize(p_type));
}


llvm::Type& 
FreeForm2::CompilationState::GetType(const TypeImpl& p_type) const
{
    switch (p_type.Primitive())
    {
        case Type::Int:
        {
            return GetIntType();
        }

        case Type::UInt32: __attribute__((__fallthrough__));
        case Type::Int32:
        {
            return GetInt32Type();
        }

        case Type::Float:
        {
            return GetFloatType();
        }

        case Type::Bool:
        {
            return GetBoolType();
        }

        case Type::Array:
        {
            const ArrayType& arrayType = static_cast<const ArrayType&>(p_type);
            switch (arrayType.GetChildType().Primitive())
            {
                case Type::Int:
                {
                    return *m_arrayIntType;
                }

                case Type::UInt32: __attribute__((__fallthrough__));
                case Type::Int32:
                {
                    return *m_arrayInt32Type;
                }

                case Type::Float:
                {
                    return *m_arrayFloatType;
                }

                case Type::Bool:
                {
                    return *m_arrayBoolType;
                }

                default:
                {
                    Unreachable(__FILE__, __LINE__);
                }
            }
        }

        case Type::Void:
        {
            return GetVoidType();
        }

        default:
        {
            Unreachable(__FILE__, __LINE__);
        }
    }
}


llvm::Value& 
FreeForm2::CompilationState::CreateZeroValue(const TypeImpl& p_type)
{
    switch (p_type.Primitive())
    {
        case Type::Int:
        {
            llvm::Value* val = llvm::ConstantInt::get(m_intType, 0);
            CHECK_LLVM_RET(val);
            return *val;
        }

        case Type::UInt32: __attribute__((__fallthrough__));
        case Type::Int32:
        {
            llvm::Value* val = llvm::ConstantInt::get(m_int32Type, 0);
            CHECK_LLVM_RET(val);
            return *val;
        }

        case Type::Float:
        {
            llvm::Value* val = llvm::ConstantFP::get(m_floatType, 0);
            CHECK_LLVM_RET(val);
            return *val;
        }

        case Type::Bool:
        {
            llvm::Value* val = llvm::ConstantInt::get(m_boolType, 0);
            CHECK_LLVM_RET(val);
            return *val;
        }

        case Type::Array:
        {
            return ArrayCodeGen::CreateEmptyArray(*this, static_cast<const ArrayType&>(p_type));
        }

        default:
        {
            Unreachable(__FILE__, __LINE__);
        }
    }
}


llvm::Value&
FreeForm2::CompilationState::CreateVoidValue() const
{
    LlvmCodeGenerator::CompiledValue* ret 
        = llvm::UndefValue::get(m_voidType);
    CHECK_LLVM_RET(ret);
    return *ret;
}


llvm::Type& 
FreeForm2::CompilationState::CreateArrayType(llvm::Type* p_base)
{
    CHECK_LLVM_RET(m_arrayBoundsType);
    CHECK_LLVM_RET(p_base);
    llvm::Type* pointerType = llvm::PointerType::get(p_base, 0);
    CHECK_LLVM_RET(pointerType);

    std::vector<llvm::Type*> structure(1, m_arrayBoundsType);
    FF2_ASSERT(structure.size() - 1 == ArrayCodeGen::boundsPosition);

    structure.push_back(m_arrayCountType);
    FF2_ASSERT(structure.size() - 1 == ArrayCodeGen::countPosition);

    structure.push_back(pointerType);
    FF2_ASSERT(structure.size() - 1 == ArrayCodeGen::pointerPosition);

    llvm::Type* result = llvm::StructType::get(*m_context, structure);
    CHECK_LLVM_RET(result);
    return *result;
}
