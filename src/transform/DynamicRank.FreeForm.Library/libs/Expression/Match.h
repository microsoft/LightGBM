/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#pragma once

#ifndef FREEFORM2_MATCH_H
#define FREEFORM2_MATCH_H

#include "Expression.h"
#include "MatchSub.h"

namespace FreeForm2
{
    // A match expression represents a used-entered match statement that
    // evaluates a pattern against a given value, and takes the given action if
    // the pattern matches.
    class MatchExpression : public Expression
    {
    public:
        // Create a match expression from the value to match, the
        // pattern, and the corresponding action.
        MatchExpression(const Annotations &p_annotations,
                        const Expression &p_value,
                        const MatchSubExpression &p_pattern,
                        const Expression &p_action,
                        bool p_isOverlapping);

        // Virtual methods inherited from Expression.
        virtual void Accept(Visitor &p_visitor) const override;
        virtual const TypeImpl &GetType() const override;
        virtual size_t GetNumChildren() const override;

        const Expression &GetValue() const;
        const MatchSubExpression &GetPattern() const;
        const Expression &GetAction() const;
        bool IsOverlapping() const;

    private:
        // Value to be matched against.
        const Expression &m_value;

        // Pattern to match.
        const MatchSubExpression &m_pattern;

        // Action to take.
        const Expression &m_action;

        // Flag to indicate whether or not the match should overlap.
        bool m_isOverlapping;
    };

    // A MatchOperatorExpression combines matching constraints with a variety of
    // operators.  We translate these to finite-state matchines before issuing
    // code, so the operations aren't very distinct in the syntax tree.
    class MatchOperatorExpression : public MatchSubExpression
    {
    public:
        // Enumeration of different match operations.
        enum Operator
        {
            // Kleene star, unbounded repetition.
            kleene,

            // '+' operation, which matches at least one repetition.
            atLeastOne,

            // Alternation, allowing any of a given set of matching constraints.
            alternation,

            // Concatenation, matching constraints in sequence.
            concatenation,

            invalid
        };

        // Methods inherited from Expression.
        virtual void Accept(Visitor &p_visitor) const override;
        virtual const TypeImpl &GetType() const override;
        virtual size_t GetNumChildren() const override;

        // Methods inherited from MatchSubExpression.
        virtual Info GetInfo() const override;

        // Methods to allocate MatchOperatorExpression objects.
        static boost::shared_ptr<MatchOperatorExpression>
        Alloc(const Annotations &p_annotations,
              const MatchSubExpression **p_children,
              size_t p_numChildren,
              Operator p_op);
        static boost::shared_ptr<MatchOperatorExpression>
        Alloc(const Annotations &p_annotations,
              const MatchSubExpression &p_left,
              const MatchSubExpression &p_right,
              Operator p_op);
        static boost::shared_ptr<MatchOperatorExpression>
        Alloc(const Annotations &p_annotations,
              const MatchSubExpression &p_expr,
              Operator p_op);

        // Get the operator represented by this expression.
        Operator GetOperator() const;

        // Get the children of this expression.
        const MatchSubExpression *const *GetChildren() const;

    private:
        MatchOperatorExpression(const Annotations &p_annotations,
                                const MatchSubExpression **p_children,
                                size_t p_numChildren,
                                Operator p_op);

        static void DeleteAlloc(MatchOperatorExpression *p_allocated);

        // Matching operator used.
        const Operator m_op;

        // Number of children of this node.
        size_t m_numChildren;

        // Array of children of this node, allocated using struct hack.
        const MatchSubExpression *m_children[1];
    };

    // A MatchGuardExpression represents guarding of a pattern by an arbitrary
    // statement evaluating to a bool
    class MatchGuardExpression : public MatchSubExpression
    {
    public:
        MatchGuardExpression(const Annotations &p_annotations,
                             const Expression &p_guard);

        // Methods inherited from Expression.
        virtual void Accept(Visitor &p_visitor) const override;
        virtual const TypeImpl &GetType() const override;
        virtual size_t GetNumChildren() const override;

        // Methods inherited from MatchSubExpression.
        virtual Info GetInfo() const override;

        const Expression &m_guard;
    };

    // Expression to allow binding of variables during matching.  Note that we
    // need to handle this separately from normal binding, due to extensive
    // state machine transformations.
    class MatchBindExpression : public MatchSubExpression
    {
    public:
        MatchBindExpression(const Annotations &p_annotations,
                            const MatchSubExpression &p_value,
                            VariableID p_id);

        // Virtual methods inherited from Expression.
        virtual void Accept(Visitor &p_visitor) const override;
        virtual const TypeImpl &GetType() const override;
        virtual size_t GetNumChildren() const override;

        // Methods inherited from MatchSubExpression.
        virtual Info GetInfo() const override;

        const MatchSubExpression &m_value;

        const VariableID m_id;
    };
};

#endif
