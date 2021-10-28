/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#pragma once

#ifndef FREEFORM2_COPYING_VISITOR_H
#define FREEFORM2_COPYING_VISITOR_H

#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include "Expression.h"
#include <map>
#include "Visitor.h"
// #include "WordConstraint.h"
#include <vector>

namespace FreeForm2
{
    class AllocationExpression;
    class SimpleExpressionOwner;
    class ExpressionOwner;

    class CopyingVisitor : public Visitor
    {
    public:
        // Create a new instance.
        CopyingVisitor();
        CopyingVisitor(const boost::shared_ptr<SimpleExpressionOwner> &p_owner,
                       const boost::shared_ptr<TypeManager> &p_typeManager);

        // Get the ExpressionOwner used by the class to own the
        // new Expression nodes.
        boost::shared_ptr<ExpressionOwner> GetExpressionOwner() const;

        // Get the TypeManager used by the class to own new Types.
        boost::shared_ptr<TypeManager> GetTypeManager() const;

        // Add an expression to the expression owner.
        void AddExpressionToOwner(const boost::shared_ptr<Expression> &p_expr);

        // Add an expression to the owner and the back of the stack.
        void AddExpression(const boost::shared_ptr<Expression> &p_expr);

        // Return the new syntax tree created by this class.
        const Expression *GetSyntaxTree() const;

        // Get the stack used by this visitor to create copies of the expression tree.
        std::vector<const Expression *> &GetStack();

        // Method inherited from Visitor.
        virtual void Visit(const SelectNthExpression &p_expr) override;
        virtual void Visit(const SelectRangeExpression &p_expr) override;
        virtual void Visit(const ConditionalExpression &p_expr) override;
        virtual void Visit(const ArrayLiteralExpression &p_expr) override;
        virtual void Visit(const LetExpression &p_expr) override;
        virtual void Visit(const BlockExpression &p_expr) override;
        virtual void Visit(const BinaryOperatorExpression &p_expr) override;
        virtual void Visit(const RangeReduceExpression &p_expr) override;
        virtual void Visit(const ForEachLoopExpression &p_expr) override;
        virtual void Visit(const ComplexRangeLoopExpression &p_expr) override;
        virtual void Visit(const MutationExpression &p_expr) override;
        virtual void Visit(const MatchExpression &p_expr) override;
        // virtual bool AlternativeVisit(const MatchWordExpression& p_expr) override;
        // virtual void Visit(const MatchWordExpression& p_expr) override;
        // virtual void Visit(const MatchLiteralExpression& p_expr) override;
        // virtual void Visit(const MatchCurrentWordExpression& p_expr) override;
        virtual void Visit(const MatchOperatorExpression &p_expr) override;
        virtual void Visit(const MatchGuardExpression &p_expr) override;
        virtual void Visit(const MatchBindExpression &p_expr) override;
        virtual void Visit(const MemberAccessExpression &p_expr) override;
        // virtual void Visit(const NeuralInputResultExpression& p_expr) override;
        // virtual void Visit(const ObjectMethodExpression& p_expr) override;
        virtual void Visit(const ArrayLengthExpression &p_expr) override;
        virtual void Visit(const ArrayDereferenceExpression &p_expr) override;
        virtual void Visit(const ConvertToFloatExpression &p_expr) override;
        virtual void Visit(const ConvertToIntExpression &p_expr) override;
        virtual void Visit(const ConvertToUInt64Expression &p_expr) override;
        virtual void Visit(const ConvertToInt32Expression &p_expr) override;
        virtual void Visit(const ConvertToUInt32Expression &p_expr) override;
        virtual void Visit(const ConvertToBoolExpression &p_expr) override;
        virtual void Visit(const ConvertToImperativeExpression &p_expr) override;
        virtual void Visit(const DeclarationExpression &p_expr) override;
        virtual void Visit(const DirectPublishExpression &p_expr) override;
        virtual void Visit(const ExternExpression &p_expr) override;
        virtual void Visit(const FunctionExpression &p_expr) override;
        virtual bool AlternativeVisit(const FunctionCallExpression &p_expr) override;
        virtual void Visit(const FunctionCallExpression &p_expr) override;
        virtual void Visit(const LiteralIntExpression &p_expr) override;
        virtual void Visit(const LiteralUInt64Expression &p_expr) override;
        virtual void Visit(const LiteralInt32Expression &p_expr) override;
        virtual void Visit(const LiteralUInt32Expression &p_expr) override;
        virtual void Visit(const LiteralFloatExpression &p_expr) override;
        virtual void Visit(const LiteralBoolExpression &p_expr) override;
        virtual void Visit(const LiteralVoidExpression &p_expr) override;
        virtual void Visit(const LiteralStreamExpression &p_expr) override;
        virtual void Visit(const LiteralWordExpression &p_expr) override;
        virtual void Visit(const LiteralInstanceHeaderExpression &p_expr) override;
        virtual void Visit(const FeatureRefExpression &p_expr) override;
        // virtual bool AlternativeVisit(const FSMExpression& p_expr) override;
        // virtual void Visit(const FSMExpression& p_expr) override;
        virtual void Visit(const UnaryOperatorExpression &p_expr) override;
        virtual void Visit(const FeatureSpecExpression &p_expr) override;
        virtual void Visit(const FeatureGroupSpecExpression &p_expr) override;
        virtual void Visit(const PhiNodeExpression &p_expr) override;
        virtual void Visit(const PublishExpression &p_expr) override;
        virtual void Visit(const ReturnExpression &p_expr) override;
        virtual void Visit(const StreamDataExpression &p_expr) override;
        virtual void Visit(const UpdateStreamDataExpression &p_expr) override;
        virtual void Visit(const VariableRefExpression &p_expr) override;
        virtual void Visit(const ImportFeatureExpression &p_expr) override;
        virtual void Visit(const StateExpression &p_expr) override;
        virtual bool AlternativeVisit(const StateMachineExpression &p_expr) override;
        virtual void Visit(const StateMachineExpression &p_expr) override;
        virtual void Visit(const ExecuteStreamRewritingStateMachineGroupExpression &p_expr) override;
        virtual void Visit(const ExecuteMachineExpression &p_expr) override;
        virtual void Visit(const ExecuteMachineGroupExpression &p_expr) override;
        virtual void Visit(const YieldExpression &p_expr) override;
        virtual void Visit(const RandFloatExpression &p_expr) override;
        virtual void Visit(const RandIntExpression &p_expr) override;
        virtual void Visit(const ThisExpression &p_expr) override;
        virtual void Visit(const UnresolvedAccessExpression &p_expr) override;
        virtual bool AlternativeVisit(const TypeInitializerExpression &p_expr) override;
        virtual void Visit(const TypeInitializerExpression &p_expr) override;
        // virtual void Visit(const DocumentContextExpression& p_expr) override;
        // virtual void Visit(const DocumentAggregationCacheExpression& p_expr) override;
        virtual void Visit(const AggregateContextExpression &p_expr) override;
        // virtual void Visit(const DocumentListExpression& p_expr) override;
        virtual void Visit(const DebugExpression &p_expr) override;

        virtual void VisitReference(const ArrayDereferenceExpression &p_expr) override;
        virtual void VisitReference(const VariableRefExpression &p_expr) override;
        virtual void VisitReference(const MemberAccessExpression &p_expr) override;
        virtual void VisitReference(const ThisExpression &p_expr) override;
        virtual void VisitReference(const UnresolvedAccessExpression &p_expr) override;

        virtual size_t StackSize() const override;
        virtual size_t StackIncrement() const override;

        // Copy a word constraint, including all Expressions within the
        // constraint.
        // FSM::WordConstraint CopyWordConstraint(const FSM::WordConstraint& p_constraint);

    protected:
        // Copy a type so that it is owned by this copy's TypeManager.
        const TypeImpl &CopyType(const TypeImpl &p_type);

        // Expression owner for the new AST.
        boost::shared_ptr<SimpleExpressionOwner> m_owner;

        // The type manager used with both the new and old AST.
        boost::shared_ptr<TypeManager> m_typeManager;

        // Temporary expression stack.
        std::vector<const Expression *> m_stack;

        // A map of pointers to FunctionExpressions in the old tree and the new one.
        std::map<const FunctionExpression *, const FunctionExpression *> m_functionTranslation;
    };
}

#endif
