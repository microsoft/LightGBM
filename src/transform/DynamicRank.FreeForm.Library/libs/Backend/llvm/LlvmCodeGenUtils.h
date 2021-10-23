#pragma once

#ifndef FREEFORM2_CODEGENUTILS_H
#define FREEFORM2_CODEGENUTILS_H

#include "FreeForm2Type.h"

namespace llvm
{
    class Type;
    class Value;
    class BasicBlock;
    class LLVMContext;
    class Instruction;
}

namespace FreeForm2
{
    class CompilationState;

    class GenerateConditional
    {
    public:
        // Constructor, taking the compilation state, conditional
        // value and (optional) description of the conditional.
        GenerateConditional(CompilationState& p_state, 
                            llvm::Value& p_cond,
                            const char* p_description);

        // Note: call functions in the order presented.  Pointers are
        // checked for NULL, with exceptions thrown in that case.

        // Finish a 'then' block (you'll want to have generated some code
        // between construction and FinishThen).
        void FinishThen(llvm::Value* p_value);

        // Finish an 'else' block (you'll want to have generated some code
        // between FinishThen and FinishElse).
        void FinishElse(llvm::Value* p_value);

        // Finish generating the conditional, returning the value of the
        // conditional.  Call this last, and don't call anything else afterward.
        llvm::Value& Finish(llvm::Type& p_type);

    private:
        // Compilation state used for generation.
        CompilationState* m_state;

        // (Optional) description of the conditional.
        const char* m_description;

        // Structure holding information about conditional cases.
        struct Case 
        {
            Case(llvm::BasicBlock* p_block);

            llvm::BasicBlock* m_block;
            llvm::Value* m_value;
        };

        // 'then' case.
        Case m_then;

        // 'else' case.
        Case m_else;

        // Basic block that merges all the cases together.
        llvm::BasicBlock* m_finalBlock;
    };

    // This class offers a mechanism to convert LLVM value types among the 
    // various TypeImpl types supported in FF2.
    class ValueConversion
    {
    public:
        // Convert a value from a source type to a destination type. Depending
        // on the type, zero or more expressions may be added to the 
        // compilation state.
        static llvm::Value& Do(llvm::Value& p_value,
                               const TypeImpl& p_sourceType,
                               const TypeImpl& p_destType,
                               CompilationState& p_state);
    };

    // Macro to help with the constant checking of return values
    // required to use LLVM.
#define CHECK_LLVM_RET(val)                                                   \
    if (val == NULL)                                                          \
    {                                                                         \
        FreeForm2::CheckLLVMRet(val, __FILE__, __LINE__);                     \
    }

    // Functions to support CHECK_LLVM_RET above.
    void CheckLLVMRet(const llvm::Value* p_value, const char* p_file, unsigned int p_line);
    void CheckLLVMRet(const llvm::Type* p_type, const char* p_file, unsigned int p_line);
    void CheckLLVMRet(const llvm::BasicBlock* p_block, const char* p_file, unsigned int p_line);
    void CheckLLVMRet(const llvm::Instruction* p_ins, const char* p_file, unsigned int p_line);
};

#endif



