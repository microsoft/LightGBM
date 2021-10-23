#pragma once
#ifndef FREEFORM2_OBJECT_RESOLUTION_VISITOR_H
#define FREEFORM2_OBJECT_RESOLUTION_VISITOR_H

#include "CopyingVisitor.h"
#include <stack>

namespace FreeForm2
{
    class CompoundType;

    // The ObjectResolutionVisitor is responsible for two-pass object type
    // annotating. For example with StateMachineExpressions, when parsed all
    // ThisExpressions that are children of the StateMachineExpression are of 
    // type Unknown, and all unknown variables turn into 
    // UnresolveAccessExpressions. Since the type can only be created after
    // parsing is complete, this information must be added in a second pass.
    class ObjectResolutionVisitor : public CopyingVisitor
    {
    public:
        // Create an ObjectResolutionVisitor using a new ExpressionOwner
        // and TypeManager.
        ObjectResolutionVisitor();

        virtual bool AlternativeVisit(const StateMachineExpression& p_expr);
        virtual void Visit(const ThisExpression& p_expr);
        virtual void Visit(const UnresolvedAccessExpression& p_expr);
        virtual bool AlternativeVisit(const TypeInitializerExpression& p_expr);

    private:
        // The current type with which a ThisExpression should be annotated.
        std::stack<const CompoundType*> m_thisTypeStack;
    };
}

#endif
