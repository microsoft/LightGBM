#include "ObjectResolutionVisitor.h"

#include <boost/cast.hpp>
#include <boost/foreach.hpp>
#include <boost/make_shared.hpp>
#include "CompoundType.h"
#include "FreeForm2Assert.h"
#include "MemberAccessExpression.h"
#include "Mutation.h"
#include "RefExpression.h"
#include <sstream>
#include "StateMachine.h"
#include "TypeManager.h"

FreeForm2::ObjectResolutionVisitor::ObjectResolutionVisitor()
{
}


bool 
FreeForm2::ObjectResolutionVisitor::AlternativeVisit(const StateMachineExpression& p_expr)
{
    FF2_ASSERT(p_expr.GetType().Primitive() == Type::StateMachine);
    const TypeImpl& copiedType = CopyType(p_expr.GetType());
    FF2_ASSERT(copiedType.IsSameAs(p_expr.GetType(), false));
    const StateMachineType& machineType = static_cast<const StateMachineType&>(copiedType);

    m_thisTypeStack.push(&machineType);
    const bool result = CopyingVisitor::AlternativeVisit(p_expr);
    FF2_ASSERT(m_thisTypeStack.top() == &machineType);
    m_thisTypeStack.pop();

    // Assert that we are correct in not calling the CopyingVisitor::Visit 
    // method for this expression.
    FF2_ASSERT(result);
    return true;
}


void 
FreeForm2::ObjectResolutionVisitor::Visit(const ThisExpression& p_expr)
{
    if (m_thisTypeStack.empty())
    {
        std::ostringstream err;
        err << "Invalid this reference: not in object scope";
        throw ParseError(err.str(), p_expr.GetSourceLocation());
    }

    const CompoundType& currentThisType = *m_thisTypeStack.top();
    if (p_expr.GetType().Primitive() != Type::Unknown
        && !p_expr.GetType().IsSameAs(currentThisType, false))
    {
        std::ostringstream err;
        err << "Object types not compatible. Expected type: "
            << currentThisType << "; found type: "
            << p_expr.GetType();
        throw ParseError(err.str(), p_expr.GetSourceLocation());
    }

    AddExpression(boost::make_shared<ThisExpression>(p_expr.GetAnnotations(),
                                                     currentThisType));
}


void 
FreeForm2::ObjectResolutionVisitor::Visit(const UnresolvedAccessExpression& p_expr)
{
    const Expression& object = *m_stack.back();
    m_stack.pop_back();

    if (object.GetType().Primitive() != Type::StateMachine)
    {
        std::ostringstream err;
        err << "Unable to resolve member" << p_expr.GetMemberName()
            << " on type " << object.GetType();
        throw ParseError(err.str(), p_expr.GetSourceLocation());
    }

    const StateMachineType& type = static_cast<const StateMachineType&>(object.GetType());
    
    const std::string memberName
        = StateMachineExpression::GetAugmentedMemberName(type.GetName(), p_expr.GetMemberName());

    const CompoundType::Member* member = type.FindMember(memberName);
    if (member == NULL)
    {
        std::ostringstream err;
        err << "Unable to resolve member " << p_expr.GetMemberName()
            << " on type " << object.GetType();
        throw ParseError(err.str(), p_expr.GetSourceLocation());
    }

    if (!member->m_type->IsSameAs(p_expr.GetType(), false) 
        && p_expr.GetType().Primitive() != Type::Unknown)
    {
        std::ostringstream err;
        err << "expected member " << p_expr.GetMemberName()
            << " to be type " << p_expr.GetType()
            << " but encountered type " << *member->m_type;
        throw ParseError(err.str(), p_expr.GetSourceLocation());
    }

    AddExpression(boost::make_shared<MemberAccessExpression>(p_expr.GetAnnotations(),
                                                             object,
                                                             *member,
                                                             0));
}


bool 
FreeForm2::ObjectResolutionVisitor::AlternativeVisit(const TypeInitializerExpression& p_expr)
{
    FF2_ASSERT(p_expr.GetType().Primitive() == Type::StateMachine);
    const TypeImpl& copiedType = CopyType(p_expr.GetType());
    FF2_ASSERT(copiedType.IsSameAs(p_expr.GetType(), false));
    const StateMachineType& machineType = static_cast<const StateMachineType&>(copiedType);

    m_thisTypeStack.push(&machineType);
    const bool result = CopyingVisitor::AlternativeVisit(p_expr);
    FF2_ASSERT(m_thisTypeStack.top() == &machineType);
    m_thisTypeStack.pop();

    // Assert that we are correct in not calling the CopyingVisitor::Visit 
    // method for this expression.
    FF2_ASSERT(result);
    return true;
}
