#pragma once

#ifndef FREEFORM2_COMPILATION_STATE_H
#define FREEFORM2_COMPILATION_STATE_H

#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/noncopyable.hpp>
#include "LlvmRuntimeLibrary.h"
#include <llvm/IR/IRBuilder.h>
#include <map>
#include "Expression.h"
#include "FreeForm2Type.h"

namespace llvm
{
    class ExecutionEngine;
    class JITMemoryManager;
    class LLVMContext;
    class Module;
    class DataLayout;
}

namespace FreeForm2
{
    // CompilationState tracks state used during compilation, notably
    // the underlying LLVM state objects, and the symbol table.
    class CompilationState : boost::noncopyable
    {
    public:
        // Create a compilation state using a memory manager to create the
        // ExecutionEngine.
        CompilationState(llvm::JITMemoryManager& p_memoryManager);

        // Get the intermediate representation builder that we use to 
        // issue instructions.
        llvm::IRBuilder<>& GetBuilder();

        // Get the LLVMContext object, which basically acts as a big container
        // object during code generation.
        llvm::LLVMContext& GetContext();

        // Get the LLVM module we're issuing code into (Module in the
        // programming sense, of a related group of code).
        llvm::Module& GetModule();

        // Get the LLVM ExecutionEngine used to compile execute a program.
        llvm::ExecutionEngine& GetExecutionEngine();

        // Get the LlvmRuntimeLibrary for this state.
        const LlvmRuntimeLibrary& GetRuntimeLibrary() const;

        // Get and set feature array value, which is passed as an argument to
        // the top-level execution and we use in feature reference expressions.
        // This function returns NULL if the feature argument has not been set.
        llvm::Value* GetFeatureArgument() const;
        void SetFeatureArgument(llvm::Value& p_val);

        // Get and set the value containing the array return space for the 
        // function. This is pre-allocated space into which a program copies
        // an array return value.
        llvm::Value& GetArrayReturnSpace() const;
        void SetArrayReturnSpace(llvm::Value& p_value);

        // Get and set document count for aggregated freeforms.
        llvm::Value& GetAggregatedDocumentCount() const;
        void SetAggregatedDocumentCount(llvm::Value& p_value);

        // Get and set document index for aggregated freeforms.
        llvm::Value& GetAggregatedDocumentIndex() const;
        void SetAggregatedDocumentIndex(llvm::Value& p_value);

        // Get and set cache pointer for aggregated freeforms.
        llvm::Value& GetAggregatedCache() const;
        void SetAggregatedCache(llvm::Value& p_value);

        // Get and set array of pointers to feature arrays for aggregated freeforms.
        llvm::Value& GetFeatureArrayPointer() const;
        void SetFeatureArrayPointer(llvm::Value& p_value);

        // Get the integer type we're using for the array bounds type.
        llvm::IntegerType& GetArrayBoundsType() const;

        // Get the integer type we're using for the array element count.
        llvm::IntegerType& GetArrayCountType() const;

        // Get the boolean type we're using for the freeform bool type.
        llvm::Type& GetBoolType() const;

        // Get the integer type we're using for the freeform int type.
        llvm::IntegerType& GetIntType() const;

        // Get number of bits in Int type.
        unsigned int GetIntBits() const;

        // Get the type used to represent signed and unsigned int32 types.
        llvm::IntegerType& GetInt32Type() const;

        // Get the type we're using for the freeform float type.
        llvm::Type& GetFloatType() const;

        // Get the type for a pointer to the freeform float type.
        llvm::Type& GetFloatPtrType() const;

        // Get the integer type we're using for the freeform void type.
        llvm::Type& GetVoidType() const;

        // Get number of bytes that a certain type needs to allocate.
        unsigned int GetSizeInBytes(llvm::Type* p_type) const;

        // Get the LLVM type corresponding to a given freeform type.
        llvm::Type& GetType(const TypeImpl& p_type) const;

        // LLVM type corresponding to the feature input type (doesn't have an
        // exactly corresponding type in the freeform type system).
        llvm::Type& GetFeatureType() const;

        // Get the LLVM type for the StreamFeatureInput object.
        llvm::Type& GetStreamFeatureInputType() const;

        // Get the zero value (0, 0.0, false, []) for given type.
        llvm::Value& CreateZeroValue(const TypeImpl& p_type);

        // Create a void value.
        llvm::Value& CreateVoidValue() const;

        // Push a value onto the stack, returning the slot that this value was
        // stored in.  Note that the stack involved here is the compile-time
        // dual of the stack of values assigned during parsing.
        void SetVariableValue(VariableID p_id, llvm::Value& p_value);

        // Get a value from the stack.  Note that the stack involved here is 
        // the compile-time dual of the stack of values assigned during parsing.
        llvm::Value* GetVariableValue(VariableID p_id) const;

        // Number of bits in each field in a word.
        static const unsigned int c_wordFieldBits = 32;

    private:
        llvm::Type& CreateArrayType(llvm::Type* p_base);

        // Initialize the execution engine to contain references to all 
        // necessary runtime functions.
        void InitializeRuntimeLibrary();
            
        // LLVMContext, which tracks global variables and other
        // 'program'-level constructs.
        boost::shared_ptr<llvm::LLVMContext> m_context;

        // LLVM module, which roughly corresponds to a module in the
        // programming sense, being a collection of functions.
        llvm::Module* m_module;

        // This object is used to add and lookup runtime functions.
        std::unique_ptr<LlvmRuntimeLibrary> m_runtimeLibrary;

        // Object to help with building intermediate representation.
        boost::shared_ptr<llvm::IRBuilder<>> m_builder;

        // Execution engine.
        std::auto_ptr<llvm::ExecutionEngine> m_engine;

        // Bit-counts of LLVM types.
        unsigned int m_intBits;

        // Type that represents array bounds.
        llvm::IntegerType* m_arrayBoundsType;

        // Type that represents array count.
        llvm::IntegerType* m_arrayCountType;

        // LLVM types corresponding to freeform types.
        llvm::Type* m_boolType;
        llvm::Type* m_arrayBoolType;
        llvm::IntegerType* m_intType;
        llvm::Type* m_arrayIntType;
        llvm::IntegerType* m_int32Type;
        llvm::Type* m_arrayInt32Type;
        llvm::Type* m_floatType;
        llvm::Type* m_floatPtrType;
        llvm::Type* m_arrayFloatType;
        llvm::Type* m_voidType;

        // LLVM type corresponding to the feature input type (doesn't have an
        // exactly corresponding type in the freeform type system).
        llvm::IntegerType* m_featureType;

        // Feature array value, passed as a top-level arg to execution.
        llvm::Value* m_featureArgument;

        // The LLVM value for a pre-allocated array return value.
        llvm::Value* m_arraySpace;

        // The value of the current word of the match.
        //llvm::Value* m_currentWord;

        // The value of the offset of the previous query word.
        llvm::Value* m_previousOffset;

        // Document count for aggregated freeforms.
        llvm::Value* m_aggregatedDocumentCount;

        // Document index for aggregated freeforms.
        llvm::Value* m_aggregatedDocumentIndex;

        // Cache pointer for aggregated freeforms.
        llvm::Value* m_aggregatedCache;

        // Array of pointers to feature arrays.
        llvm::Value* m_featureArrayPointer;

        // Map of precalculated quantities, referred to by variable ID, as well
        // as a boolean indication of whether the quantity is a reference (a
        // pointer) or not.
        std::map<VariableID, llvm::Value*> m_variables;

        // Holds information about the LLVM target, like sizes of structures in memory.
        const llvm::DataLayout* m_targetData;
    };
};

#endif

