#include "LlvmCodeGenUtils.h"

#include "CompilationState.h"
#include "FreeForm2.h"
#include "FreeForm2Assert.h"
#include "FreeForm2Type.h"
#include <sstream>
#include <stdexcept>
#include <llvm/IR/Function.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/BasicBlock.h>

namespace
{
    using namespace FreeForm2;

    void 
    LLVMRetFailed(const char* p_file, 
                  unsigned int p_line, 
                  const char* p_description)
    {
        std::ostringstream err;
        err << "Received NULL " << p_description << " from LLVM at " 
            << p_file << ":" << p_line;
        throw std::runtime_error(err.str());
    }


    std::string 
    NameBlock(const char* p_description, const char* p_block)
    {
        std::string ret(p_description);
        ret += ": ";
        ret += p_block;
        return ret;
    }


    // Convert to bool by comparing the source value to zero.
    llvm::Value* ConvertToBool(llvm::Value* p_value,
                               const TypeImpl& p_sourceType,
                               CompilationState& p_state)
    {
        llvm::Value& zero = p_state.CreateZeroValue(p_sourceType);
        if (p_sourceType.IsIntegerType())
        {
            return p_state.GetBuilder().CreateICmpNE(p_value, &zero);
        }
        else if (p_sourceType.IsFloatingPointType())
        {
            return p_state.GetBuilder().CreateFCmpONE(p_value, &zero);
        }
        else
        {
            FF2_UNREACHABLE();
        }
    }


    // Convert any FreeForm2 primitive to any int type.
    llvm::Value* ConvertToInt(llvm::Value* p_value,
                              const TypeImpl& p_sourceType,
                              const TypeImpl& p_destType,
                              CompilationState& p_state)
    {
        FF2_ASSERT(p_destType.IsIntegerType());
        llvm::Type& destType = p_state.GetType(p_destType);
        const llvm::Type& sourceType = p_state.GetType(p_sourceType);

        if (p_sourceType.IsIntegerType())
        {
            FF2_ASSERT(destType.getPrimitiveSizeInBits() > 0
                       && sourceType.getPrimitiveSizeInBits() > 0);

            // For integer types of the same size, LLVM does not associate 
            // signed-ness with the type, so this case is an identity 
            // conversion.
            if (destType.getPrimitiveSizeInBits() == sourceType.getPrimitiveSizeInBits())
            {
                return p_value;
            }
            else if (p_destType.IsSigned() && p_sourceType.IsSigned())
            {
                return p_state.GetBuilder().CreateSExt(p_value, &destType);
            }
            else
            {
                return p_state.GetBuilder().CreateZExt(p_value, &destType);
            }
        }
        else if (p_sourceType.Primitive() == Type::Bool)
        {
            return p_state.GetBuilder().CreateZExt(p_value, &destType);
        }
        else if (p_sourceType.IsFloatingPointType())
        {
            if (p_destType.IsSigned())
            {
                return p_state.GetBuilder().CreateFPToSI(p_value, &destType);
            }
            else
            {
                return p_state.GetBuilder().CreateFPToUI(p_value, &destType);
            }
        }
        else
        {
            FF2_UNREACHABLE();
        }
    }


    // Convert to a value to a floating-point type. Right now, there is only
    // Type::Float, but this method is generic enough to support more floating-
    // point types in the future. This method does not support identity 
    // conversion.
    llvm::Value* ConvertToFloat(llvm::Value* p_value,
                                const TypeImpl& p_sourceType,
                                const TypeImpl& p_destType,
                                CompilationState& p_state)
    {
        llvm::Type& destType = p_state.GetType(p_destType);
        const llvm::Type& sourceType = p_state.GetType(p_sourceType);

        if (p_sourceType.IsIntegerType() || p_sourceType.Primitive() == Type::Bool)
        {
            if (p_sourceType.IsSigned())
            {
                return p_state.GetBuilder().CreateSIToFP(p_value, &destType);
            }
            else
            {
                return p_state.GetBuilder().CreateUIToFP(p_value, &destType);
            }
        }
        else if (p_sourceType.IsFloatingPointType())
        {
            // LLVM differentiates floating-point types based on their 
            // TypeID attributes, not on mantissa/exponent size, bit size, etc.
            // For now, assert that they are the same LLVM type ID to prevent
            // any issues. In the future, use fptrunc and fpext commands to
            // convert between floating-point types.
            FF2_ASSERT(destType.getTypeID() == sourceType.getTypeID());
            return p_value;
        }
        else
        {
            FF2_UNREACHABLE();
        }
    }
}


FreeForm2::GenerateConditional::GenerateConditional(CompilationState& p_state, 
                                                    llvm::Value& p_cond,
                                                    const char* p_description)
    : m_state(&p_state),
      m_description(p_description),
      m_then(llvm::BasicBlock::Create(p_state.GetContext(), 
                                      llvm::Twine(NameBlock(p_description, "then")),
                                      p_state.GetBuilder().GetInsertBlock()->getParent())),
      m_else(llvm::BasicBlock::Create(p_state.GetContext(), 
                                      llvm::Twine(NameBlock(p_description, "else")))),
      m_finalBlock(llvm::BasicBlock::Create(p_state.GetContext(), 
                                            llvm::Twine(NameBlock(p_description, "final"))))
{
    // Other blocks are checked by Case constructor.
    CHECK_LLVM_RET(m_finalBlock);

    p_state.GetBuilder().CreateCondBr(&p_cond, m_then.m_block, m_else.m_block);

    m_then.m_block->moveAfter(p_state.GetBuilder().GetInsertBlock());

    // Set up the builder to generate code into the 'then' block.
    p_state.GetBuilder().SetInsertPoint(m_then.m_block);
}


void 
FreeForm2::GenerateConditional::FinishThen(llvm::Value* p_value)
{
    CHECK_LLVM_RET(p_value);

    if (m_state->GetBuilder().GetInsertBlock()->getTerminator() == NULL)
    {
        m_then.m_value = p_value;

        // Finish up the 'then' block.
        m_state->GetBuilder().CreateBr(m_finalBlock);
    }
    else
    {
        m_then.m_value = NULL;
    }

    // Update 'then' block in case other codegen has altered it.
    m_then.m_block = m_state->GetBuilder().GetInsertBlock();

    // Set up the builder to generate code into the 'else' block.
    llvm::Function* fun = m_state->GetBuilder().GetInsertBlock()->getParent();
    fun->getBasicBlockList().push_back(m_else.m_block);
    m_else.m_block->moveAfter(m_then.m_block);
    m_state->GetBuilder().SetInsertPoint(m_else.m_block);
}


void 
FreeForm2::GenerateConditional::FinishElse(llvm::Value* p_value)
{
    CHECK_LLVM_RET(p_value);

    if (m_state->GetBuilder().GetInsertBlock()->getTerminator() == NULL)
    {
        m_else.m_value = p_value;

        // Finish up the 'else' block.
        m_state->GetBuilder().CreateBr(m_finalBlock);
    }
    else
    {
        m_else.m_value = NULL;
    }

    // Update 'else' block in case other codegen has altered it.
    m_else.m_block = m_state->GetBuilder().GetInsertBlock();
}


llvm::Value& 
FreeForm2::GenerateConditional::Finish(llvm::Type& p_type)
{
    // Set up the builder to generate code into the final block.
    llvm::Function* fun = m_state->GetBuilder().GetInsertBlock()->getParent();
    fun->getBasicBlockList().push_back(m_finalBlock);
    m_finalBlock->moveAfter(m_else.m_block);
    m_state->GetBuilder().SetInsertPoint(m_finalBlock);

    if (m_then.m_value != NULL && m_else.m_value != NULL)
    {
        llvm::PHINode* phi = m_state->GetBuilder().CreatePHI(&p_type, 2);
        CHECK_LLVM_RET(phi);
        phi->addIncoming(m_then.m_value, m_then.m_block);
        phi->addIncoming(m_else.m_value, m_else.m_block);
        return *phi;
    }
    else if (m_then.m_value != NULL)
    {
        return *m_then.m_value;
    }
    else if (m_else.m_value != NULL)
    {
        return *m_else.m_value;
    }
    else
    {
        return m_state->CreateVoidValue();
    }
}


llvm::Value&
FreeForm2::ValueConversion::Do(llvm::Value& p_value,
                               const TypeImpl& p_sourceType,
                               const TypeImpl& p_destType,
                               CompilationState& p_state)
{
    llvm::Value* ret = nullptr;
    if (p_sourceType.IsSameAs(p_destType, true))
    {
        ret = &p_value;
    }
    else if (p_destType.IsIntegerType())
    {
        ret = ConvertToInt(&p_value, p_sourceType, p_destType, p_state);
    }
    else if (p_destType.IsFloatingPointType())
    {
        ret = ConvertToFloat(&p_value, p_sourceType, p_destType, p_state);
    }
    else if (p_destType.Primitive() == Type::Bool)
    {
        ret = ConvertToBool(&p_value, p_sourceType, p_state);
    }
    CHECK_LLVM_RET(ret);
    return *ret;
}


FreeForm2::GenerateConditional::Case::Case(llvm::BasicBlock* p_block)
    : m_block(p_block),
      m_value(NULL)
{
    CHECK_LLVM_RET(p_block);
}


void 
FreeForm2::CheckLLVMRet(const llvm::Value* p_value, 
                        const char* p_file, 
                        unsigned int p_line)
{
    if (p_value == NULL)
    {
        LLVMRetFailed(p_file, p_line, "value");
    }
}


void 
FreeForm2::CheckLLVMRet(const llvm::Type* p_type, 
                        const char* p_file, 
                        unsigned int p_line)
{
    if (p_type == NULL)
    {
        LLVMRetFailed(p_file, p_line, "type");
    }
}


void 
FreeForm2::CheckLLVMRet(const llvm::BasicBlock* p_block, 
                        const char* p_file, 
                        unsigned int p_line)
{
    if (p_block == NULL)
    {
        LLVMRetFailed(p_file, p_line, "basic block");
    }
}


void 
FreeForm2::CheckLLVMRet(const llvm::Instruction* p_ins, 
                        const char* p_file, 
                        unsigned int p_line)
{
    if (p_ins == NULL)
    {
        LLVMRetFailed(p_file, p_line, "instruction");
    }
}
