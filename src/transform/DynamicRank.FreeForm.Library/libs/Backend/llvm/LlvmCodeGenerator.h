#pragma once

#ifndef FREEFORM2_LLVMCODEGENERATOR_H
#define FREEFORM2_LLVMCODEGENERATOR_H

#include <basic_types.h>
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include "CompilationState.h"
#include "Expression.h"
#include <map>
#include <stack>
#include <vector>
#include "Visitor.h"

namespace llvm
{
    class Value;
    class BasicBlock;
    class PHINode;
}

namespace FreeForm2
{
    class Allocation;

    // LlvmCodeGenerator, an interface to implement the visitor pattern.
    class LlvmCodeGenerator : public Visitor
    {
    public:
        typedef llvm::Value CompiledValue;
        typedef std::vector<boost::shared_ptr<Allocation>> AllocationVector;

        // Compile an expression tree down to LLVM-IR.
        static llvm::Function& Compile(const Expression& p_expr,
                                       CompilationState& p_state,
                                       const AllocationVector& p_allocations,
                                       CompilerFactory::DestinationFunctionType p_destinationFunctionType);

        LlvmCodeGenerator(CompilationState& p_state,
                          const AllocationVector& p_allocations,
                          CompilerFactory::DestinationFunctionType p_destinationFunctionType);

        // Provide the implementation of the Visitor pattern to generate
        // LLVM code for each expression and operator.
        void Visit(const LiteralFloatExpression& p_expr) override;
        void Visit(const SelectNthExpression& p_expr) override;
        void Visit(const SelectRangeExpression& p_expr) override;
        void Visit(const ArrayLengthExpression& p_expr) override;
        void Visit(const ArrayDereferenceExpression& p_expr) override;
        void Visit(const ArrayLiteralExpression& p_expr) override;
        bool AlternativeVisit(const ConditionalExpression& p_expr) override;
        void Visit(const ConditionalExpression& p_expr) override;
        void Visit(const ConvertToFloatExpression& p_expr) override;
        void Visit(const ConvertToIntExpression& p_expr) override;
        void Visit(const ConvertToUInt64Expression& p_expr) override;
        void Visit(const ConvertToInt32Expression& p_expr) override;
        void Visit(const ConvertToUInt32Expression& p_expr) override;
        void Visit(const ConvertToBoolExpression& p_expr) override;
        void Visit(const ConvertToImperativeExpression& p_expr) override;
        void Visit(const DeclarationExpression& p_expr) override;
        void Visit(const DirectPublishExpression& p_expr) override;
        void Visit(const ExternExpression& p_expr) override;
        void Visit(const LiteralIntExpression& p_expr) override;
        void Visit(const LiteralUInt64Expression& p_expr) override;
        void Visit(const LiteralInt32Expression& p_expr) override;
        void Visit(const LiteralUInt32Expression& p_expr) override;
        void Visit(const LiteralBoolExpression& p_expr) override;
        void Visit(const LiteralVoidExpression& p_expr) override;
        void Visit(const LiteralStreamExpression& p_expr) override;
        void Visit(const LiteralWordExpression& p_expr) override;
        void Visit(const LiteralInstanceHeaderExpression& p_expr) override;
        bool AlternativeVisit(const LetExpression& p_expr) override;
        void Visit(const LetExpression& p_expr) override;
        void Visit(const MutationExpression& p_expr) override;
        void Visit(const MatchExpression& p_expr) override;
        void Visit(const MatchOperatorExpression&) override;
        void Visit(const MatchGuardExpression&) override;
        void Visit(const MatchBindExpression&) override;
        void Visit(const MemberAccessExpression&) override;
        void Visit(const PhiNodeExpression&) override;
        void Visit(const PublishExpression&) override;
        bool AlternativeVisit(const BlockExpression& p_expr) override;
        void Visit(const BlockExpression& p_expr) override;
        void Visit(const FeatureRefExpression& p_expr) override;
        void Visit(const FunctionExpression& p_expr) override;
        void Visit(const FunctionCallExpression& p_expr) override;
        bool AlternativeVisit(const RangeReduceExpression& p_expr) override;
        void Visit(const RangeReduceExpression& p_expr) override;
        void Visit(const ForEachLoopExpression& p_expr) override;
        void Visit(const ComplexRangeLoopExpression& p_expr) override;
        void Visit(const UnaryOperatorExpression& p_expr) override;
        void Visit(const BinaryOperatorExpression& p_expr) override;
        bool AlternativeVisit(const FeatureSpecExpression& p_expr) override;
        void Visit(const FeatureSpecExpression& p_expr) override;
        void Visit(const FeatureGroupSpecExpression& p_expr) override;
        void Visit(const ReturnExpression& p_expr) override;
        void Visit(const StreamDataExpression& p_expr) override;
        void Visit(const UpdateStreamDataExpression& p_expr) override;
        void Visit(const VariableRefExpression& p_expr) override;
        void Visit(const ImportFeatureExpression& p_expr) override;
        void Visit(const StateExpression& p_expr) override;
        void Visit(const StateMachineExpression& p_expr) override;
        void Visit(const ExecuteMachineExpression& p_expr) override;
        void Visit(const ExecuteStreamRewritingStateMachineGroupExpression& p_expr) override;
        void Visit(const ExecuteMachineGroupExpression& p_expr) override;
        void Visit(const YieldExpression& p_expr) override;
        void Visit(const RandFloatExpression& p_expr) override;
        void Visit(const RandIntExpression& p_expr) override;
        void Visit(const ThisExpression& p_expr) override;
        void Visit(const UnresolvedAccessExpression& p_expr) override;
        void Visit(const TypeInitializerExpression& p_expr) override;
        void Visit(const AggregateContextExpression& p_expr) override;
        void Visit(const DebugExpression& p_expr) override;

        void VisitReference(const ArrayDereferenceExpression& p_expr) override;
        void VisitReference(const VariableRefExpression& p_expr) override;
        void VisitReference(const MemberAccessExpression& p_expr) override;
        void VisitReference(const ThisExpression& p_expr) override;
        void VisitReference(const UnresolvedAccessExpression& p_expr) override;

        // Gets the compiled version of the whole object tree.
        LlvmCodeGenerator::CompiledValue* GetResult();

        llvm::Function* GetFuction() const;

    private:
        // Helper functions for unary operator expressions.
        void VisitUnaryMinus(const UnaryOperatorExpression& p_expr);
        void VisitUnaryLog(const UnaryOperatorExpression& p_expr, bool p_addOne);
        void VisitUnaryNot(const UnaryOperatorExpression& p_expr);
        void VisitUnaryAbs(const UnaryOperatorExpression& p_expr);
        void VisitUnaryRound(const UnaryOperatorExpression& p_expr);
        void VisitUnaryTrunc(const UnaryOperatorExpression& p_expr);

        // Helper functions for binary operator expressions.
        void VisitPlus(const BinaryOperatorExpression& p_expr);
        void VisitMinus(const BinaryOperatorExpression& p_expr);
        void VisitMultiply(const BinaryOperatorExpression& p_expr);
        void VisitDivides(const BinaryOperatorExpression& p_expr);
        void VisitMod(const BinaryOperatorExpression& p_expr);
        void VisitAnd(const BinaryOperatorExpression& p_expr);
        void VisitOr(const BinaryOperatorExpression& p_expr);
        void VisitLog(const BinaryOperatorExpression& p_expr);
        void VisitPow(const BinaryOperatorExpression& p_expr);
        void VisitMaxMin(const BinaryOperatorExpression& p_expr, bool p_minimum);
        void VisitEqual(const BinaryOperatorExpression& p_expr, bool p_inequality);
        void VisitCompare(const BinaryOperatorExpression& p_expr, bool p_less, bool p_equal);

        // Helper function for array dereference expressions.
        void VisitArrayDereference(const ArrayDereferenceExpression& p_expr, 
                                   bool p_isRef);

        // Allocate memory for the given allocation.
        void Allocate(const Allocation& p_allocation);

        // Create an LLVM function to wrap a Derived/FeatureSpecSpec.
        llvm::Function* CreateFeatureFunction(const TypeImpl& p_returnType);

        // Create a return value of the specified type. Array space is provided
        // for returning an array.
        CompiledValue& CreateReturn(const TypeImpl& p_type);

        // Generate all allocation code stored in the program's allocation vector.
        void CreateAllocations();

        // A stack of intermediate expressions that have already been compiled.
        std::stack<LlvmCodeGenerator::CompiledValue*> m_stack;

        // A stack that keeps track potision of the document context.
        // This is for aggregated freeform, do not use it in ffv2.
        std::stack<LlvmCodeGenerator::CompiledValue*> m_documentContextStack;

        // Holds the underlying LLVM state objects, and the symbol table.
        CompilationState& m_state;

        // A reference to the program being compiled.
        const AllocationVector& m_allocations;

        // Holds the state needed for decoupling array allocation and initialization. 
        struct PreAllocatedArray
        {
            LlvmCodeGenerator::CompiledValue* m_bounds;
            LlvmCodeGenerator::CompiledValue* m_count;
            LlvmCodeGenerator::CompiledValue* m_array;
        };

        // A mapping to get the PreallocatedArray structure for each ArrayLiteralExpression.
        std::map<VariableID, PreAllocatedArray> m_allocatedArrays;

        // A mapping to get the pre-allocated CompiledValue for each expression.
        std::map<VariableID, LlvmCodeGenerator::CompiledValue*> m_allocatedValues;

        // Destination Function Type.
        CompilerFactory::DestinationFunctionType m_destinationFunctionType;

        const TypeImpl* m_returnType;
        CompiledValue* m_returnValue;
        llvm::Function* m_function;
    };
}

#endif
