#pragma once

#ifndef FREEFORM2_TYPE_CHECKING_VISITOR_H
#define FREEFORM2_TYPE_CHECKING_VISITOR_H

#include "UniformExpressionVisitor.h"
#include "TypeImpl.h"
#include "FeatureSpec.h"
#include <stack>

namespace FreeForm2
{
    // Visitor that visits all expressions to ensure that
    // type-checking occurs.
    class TypeCheckingVisitor : public UniformExpressionVisitor
    {
    public:
        // Construct a type checking visitor.
        TypeCheckingVisitor();

        // These expressions require special handling to ensure that their
        // children have side effects. Blocks, conditionals, matches, and
        // conversions-to-imperative have side effects iff all of their
        // children have side effects.
        virtual bool AlternativeVisit(const BlockExpression& p_expr) override;
        virtual bool AlternativeVisit(const ConditionalExpression& p_expr) override;
        virtual bool AlternativeVisit(const MatchExpression& p_expr) override;
        virtual bool AlternativeVisit(const ConvertToImperativeExpression& p_expr) override;
        virtual bool AlternativeVisit(const RangeReduceExpression& p_expr) override;
        virtual bool AlternativeVisit(const ForEachLoopExpression& p_expr) override;
        virtual bool AlternativeVisit(const ComplexRangeLoopExpression& p_expr) override;
        virtual bool AlternativeVisit(const AggregateContextExpression& p_expr) override;
        virtual bool AlternativeVisit(const FunctionExpression& p_expr) override;
        virtual bool AlternativeVisit(const FunctionCallExpression& p_expr) override;
        virtual bool AlternativeVisit(const FeatureSpecExpression& p_expr) override;
        virtual bool AlternativeVisit(const StateMachineExpression& p_expr) override;
        virtual bool AlternativeVisit(const LetExpression& p_expr) override;
        virtual bool AlternativeVisit(const ExecuteStreamRewritingStateMachineGroupExpression& p_expr) override;
        virtual bool AlternativeVisit(const ExecuteMachineGroupExpression& p_expr) override;

        // Mutations and declarations always have side effects.
        virtual void Visit(const MutationExpression& p_expr) override;
        virtual void Visit(const DeclarationExpression& p_expr) override;
        virtual void Visit(const DirectPublishExpression& p_expr) override;
        virtual void Visit(const PublishExpression& p_expr) override;
        virtual void Visit(const ReturnExpression& p_expr) override;
        virtual void Visit(const ImportFeatureExpression& p_expr) override;
        // virtual void Visit(const ObjectMethodExpression& p_expr) override;
        virtual void Visit(const VariableRefExpression& p_expr) override;
        virtual void Visit(const DebugExpression& p_expr) override;
        virtual void VisitReference(const VariableRefExpression& p_expr) override;
        
        // Check types for all statements.
        virtual void Visit(const Expression& p_expr) override;

    private:
        // This flag indicates whether the last child expression of the 
        // current block acts as the return value of the block.
        bool m_lastExpressionReturns;

        // This flag indicates whether the last visited expression tree has
        // side effects.
        bool m_hasSideEffects;

        // Store the map of published features to their return types so that
        // publish expressions can be type checked against it.
        const FeatureSpecExpression::PublishFeatureMap* m_publishFeatureMap;

        // Assert that the last visited expression tree has side effects.
        void AssertSideEffects(const SourceLocation& p_sourceLocation) const;

        // This map is used for verifying variable declaration/reference type
        // matching.
        std::map<VariableID, const TypeImpl*> m_variableTypes;

        // This structure holds information about the function that is currently
        // being analyzed.
        struct FunctionState
        {
            const TypeImpl* m_returnType;
            bool m_allPathsReturn;
        };

        // A stack that holds the checking state for functions.
        // Since FunctionCallExpression has a pointer to the FunctionExpression,
        // function declarations can be nested in the expression tree, although
        // this is forbidden by the grammar.
        std::stack<FunctionState> m_functions;
    };
}

#endif
