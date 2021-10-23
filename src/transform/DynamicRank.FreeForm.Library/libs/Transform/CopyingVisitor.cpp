#include "CopyingVisitor.h"

#include "SimpleExpressionOwner.h"

#include <boost/cast.hpp>
#include <boost/foreach.hpp>
#include <boost/make_shared.hpp>
#include <boost/ref.hpp>
#include <boost/shared_ptr.hpp>
#include "ArrayDereferenceExpression.h"
#include "ArrayLength.h"
#include "ArrayLiteralExpression.h"
#include "BlockExpression.h"
#include "Conditional.h"
#include "ConvertExpression.h"
#include "DebugExpression.h"
#include "Declaration.h"
#include "Expression.h"
#include "Extern.h"
#include "FeatureSpec.h"
#include "FreeForm2Assert.h"
#include "Function.h"
#include "LetExpression.h"
#include "LiteralExpression.h"
#include <map>
#include "Match.h"
#include "MemberAccessExpression.h"
#include "Mutation.h"
#include "ObjectType.h"
#include "OperatorExpression.h"
#include "PhiNode.h"
#include "Publish.h"
#include "RangeReduceExpression.h"
#include "RandExpression.h"
#include "RefExpression.h"
#include "SelectNth.h"
#include "StateMachine.h"
#include "StateMachineType.h"
#include "StreamData.h"
#include <sstream>
#include "TypeManager.h"
#include "TypeUtil.h"


FreeForm2::CopyingVisitor::CopyingVisitor()
    : m_owner(boost::make_shared<SimpleExpressionOwner>()), 
      m_typeManager(TypeManager::CreateTypeManager().release())
{
}


FreeForm2::CopyingVisitor::CopyingVisitor(const boost::shared_ptr<SimpleExpressionOwner>& p_owner,
                                          const boost::shared_ptr<TypeManager>& p_typeManager)
    : m_owner(p_owner), m_typeManager(p_typeManager)
{
}


boost::shared_ptr<FreeForm2::ExpressionOwner>
FreeForm2::CopyingVisitor::GetExpressionOwner() const
{
    return m_owner;
}


boost::shared_ptr<FreeForm2::TypeManager>
FreeForm2::CopyingVisitor::GetTypeManager() const
{
    return m_typeManager;
}


const FreeForm2::Expression*
FreeForm2::CopyingVisitor::GetSyntaxTree() const
{
    FF2_ASSERT(m_stack.size() == 1);
    return m_stack.back();
}


std::vector<const FreeForm2::Expression*>&
FreeForm2::CopyingVisitor::GetStack()
{
    return m_stack;
}


void 
FreeForm2::CopyingVisitor::AddExpression(
    const boost::shared_ptr<Expression>& p_expr)
{
    m_owner->AddExpression(p_expr);
    m_stack.push_back(p_expr.get());
}


void
FreeForm2::CopyingVisitor::AddExpressionToOwner(
    const boost::shared_ptr<Expression>& p_expr)
{
    m_owner->AddExpression(p_expr);
}


void 
FreeForm2::CopyingVisitor::Visit(const SelectNthExpression& p_expr)
{
    std::vector<const Expression*> children(p_expr.GetNumChildren());

    children[0] = m_stack.back();
    m_stack.pop_back();

    for (unsigned int i = 0; i < p_expr.GetNumChildren() - 1; i++)
    {
        // Children are pushed on the stack in the reverse order from what
        // SelectNthExpression::Alloc expects.
        children[p_expr.GetNumChildren() - i - 1] = m_stack.back();
        m_stack.pop_back();
    }

    AddExpression(SelectNthExpression::Alloc(p_expr.GetAnnotations(), children));
}


void 
FreeForm2::CopyingVisitor::Visit(const SelectRangeExpression& p_expr)
{
    const Expression& arrayExp = *m_stack.back();
    m_stack.pop_back();

    const Expression& count = *m_stack.back();
    m_stack.pop_back();

    const Expression& start = *m_stack.back();
    m_stack.pop_back();

    AddExpression(boost::make_shared<SelectRangeExpression>(p_expr.GetAnnotations(), 
                                                            start,
                                                            count,
                                                            arrayExp,
                                                            *m_typeManager));
}


void
FreeForm2::CopyingVisitor::Visit(const ConditionalExpression& p_expr)
{
    const Expression& conditionExpression = *m_stack.back();
    m_stack.pop_back();
    const Expression& thenExpression = *m_stack.back();
    m_stack.pop_back();
    const Expression& elseExpression = *m_stack.back();
    m_stack.pop_back();

    AddExpression(boost::make_shared<ConditionalExpression>(p_expr.GetAnnotations(),
                                                            conditionExpression,
                                                            thenExpression,
                                                            elseExpression));
}


void
FreeForm2::CopyingVisitor::Visit(const ArrayLiteralExpression& p_expr)
{
    std::vector<const Expression*> children;
    children.reserve(p_expr.GetNumChildren());

    for (size_t i = 0; i < p_expr.GetNumChildren(); i++)
    {
        children.push_back(m_stack.back());
        m_stack.pop_back();
    }

    FF2_ASSERT(p_expr.GetType().Primitive() == Type::Array);
    const ArrayType& exprType = static_cast<const ArrayType&>(CopyType(p_expr.GetType()));

    AddExpression(ArrayLiteralExpression::Alloc(p_expr.GetAnnotations(),
                                                exprType, 
                                                children,
                                                p_expr.GetId()));
}


void
FreeForm2::CopyingVisitor::Visit(const LetExpression& p_expr)
{
    std::vector<LetExpression::IdExpressionPair> children(p_expr.GetNumChildren() - 1);
    const Expression& value = *m_stack.back();
    m_stack.pop_back();

    for (size_t i = 0; i < p_expr.GetNumChildren() - 1; i++)
    {
        const size_t index = p_expr.GetNumChildren() - i - 2;
        children[index] = std::make_pair(p_expr.GetBound()[index].first, m_stack.back());
        m_stack.pop_back();
    }

    AddExpression(LetExpression::Alloc(p_expr.GetAnnotations(), children, &value));
}


void 
FreeForm2::CopyingVisitor::Visit(const BlockExpression& p_expr)
{
    std::vector<const Expression*> children(p_expr.GetNumChildren());

    for (size_t i = 0; i < p_expr.GetNumChildren(); i++)
    {
        const size_t index = p_expr.GetNumChildren() - i - 1;
        children[index] = m_stack.back();
        m_stack.pop_back();
    }

    AddExpression(BlockExpression::Alloc(p_expr.GetAnnotations(),
                                         &children[0], 
                                         static_cast<unsigned int>(p_expr.GetNumChildren()), 
                                         p_expr.GetNumBound()));
}


void
FreeForm2::CopyingVisitor::Visit(const BinaryOperatorExpression& p_expr)
{
    const size_t numChildren = p_expr.GetNumChildren();
    std::vector<const Expression*> children(numChildren);

    for (size_t i = 0; i < numChildren; i++)
    {
        children[numChildren - i - 1] = m_stack.back();
        m_stack.pop_back();
    }

    AddExpression(BinaryOperatorExpression::Alloc(p_expr.GetAnnotations(),
                                                  children, 
                                                  p_expr.GetOperator(),
                                                  *m_typeManager));
}


void
FreeForm2::CopyingVisitor::Visit(const UnaryOperatorExpression& p_expr)
{
    const Expression& child = *m_stack.back();
    m_stack.pop_back();

    AddExpression(boost::make_shared<UnaryOperatorExpression>(p_expr.GetAnnotations(), child, p_expr.m_op));
}


void
FreeForm2::CopyingVisitor::Visit(const RangeReduceExpression& p_expr)
{
    const Expression& reduce = *m_stack.back();
    m_stack.pop_back();
    const Expression& low = *m_stack.back();
    m_stack.pop_back();
    const Expression& high = *m_stack.back();
    m_stack.pop_back();
    const Expression& initial = *m_stack.back();
    m_stack.pop_back();

    AddExpression(boost::make_shared<RangeReduceExpression>(p_expr.GetAnnotations(),
                                                            low, 
                                                            high,
                                                            initial, 
                                                            reduce,
                                                            p_expr.GetStepId(),
                                                            p_expr.GetReduceId()));
}


void
FreeForm2::CopyingVisitor::Visit(const ForEachLoopExpression& p_expr)
{
    const Expression& body = *m_stack.back();
    m_stack.pop_back();
    const Expression& next = *m_stack.back();
    m_stack.pop_back();
    const Expression& end = *m_stack.back();
    m_stack.pop_back();
    const Expression& begin = *m_stack.back();
    m_stack.pop_back();

    AddExpression(boost::make_shared<ForEachLoopExpression>(p_expr.GetAnnotations(),
                                                            std::make_pair(&begin, &end), 
                                                            next,
                                                            body, 
                                                            p_expr.GetIteratorId(),
                                                            p_expr.GetVersion(),
                                                            p_expr.GetHint(),
                                                            boost::ref(*m_typeManager)));
}


void
FreeForm2::CopyingVisitor::Visit(const ComplexRangeLoopExpression& p_expr)
{
    const Expression& loopCondition = *m_stack.back();
    m_stack.pop_back();
    const Expression& body = *m_stack.back();
    m_stack.pop_back();
    const Expression& step = *m_stack.back();
    m_stack.pop_back();
    const Expression& high = *m_stack.back();
    m_stack.pop_back();
    const Expression& low = *m_stack.back();
    m_stack.pop_back();
    const Expression& precondition = *m_stack.back();
    m_stack.pop_back();

    AddExpression(boost::make_shared<ComplexRangeLoopExpression>(p_expr.GetAnnotations(),
                                                                 std::make_pair(&low, &high), 
                                                                 step,
                                                                 body, 
                                                                 precondition,
                                                                 loopCondition,
                                                                 CopyType(p_expr.GetStepType()),
                                                                 p_expr.GetStepId(),
                                                                 p_expr.GetVersion()));
}


void
FreeForm2::CopyingVisitor::Visit(const MutationExpression& p_expr)
{
    const Expression* right = m_stack.back();
    m_stack.pop_back();
    const Expression* left = m_stack.back();
    m_stack.pop_back();

    AddExpression(boost::make_shared<MutationExpression>(p_expr.GetAnnotations(), *left, *right));
}


void
FreeForm2::CopyingVisitor::Visit(const MatchExpression& p_expr)
{
    const Expression* action = m_stack.back();
    m_stack.pop_back();
    const MatchSubExpression* pattern 
        = boost::polymorphic_downcast<const MatchSubExpression*>(m_stack.back());
    m_stack.pop_back();
    const Expression* value = m_stack.back();
    m_stack.pop_back();

    AddExpression(boost::make_shared<MatchExpression>(p_expr.GetAnnotations(), *value, *pattern, *action, p_expr.IsOverlapping()));
}


// bool 
// FreeForm2::CopyingVisitor::AlternativeVisit(const MatchWordExpression& p_expr)
// {
//     FSM::WordConstraint constraint = CopyWordConstraint(p_expr.GetConstraint());

//     const size_t numEffects = p_expr.GetNumEffects();
//     std::vector<const DeclarationExpression*> children(numEffects);

//     for (size_t i = 0; i < numEffects; i++)
//     {
//         p_expr.BeginEffects()[i]->Accept(*this);
//         children[i] 
//             = boost::polymorphic_downcast<const DeclarationExpression*>(m_stack.back());
//         m_stack.pop_back();
//     }

//     AddExpression(MatchWordExpression::Alloc(p_expr.GetAnnotations(),
//                                              constraint, 
//                                              children.size(), 
//                                              children.empty() ? NULL : &children[0]));
//     return true;
// }


// void 
// FreeForm2::CopyingVisitor::Visit(const MatchWordExpression& p_expr)
// {
//     // Should be handled by AlternativeVisit, above.
//     FF2_ASSERT(false);
// }


// void 
// FreeForm2::CopyingVisitor::Visit(const MatchLiteralExpression& p_expr)
// {
//     FF2_ASSERT(p_expr.GetNumChildren() == 1);
//     const Expression* child = m_stack.back();
//     m_stack.pop_back();
 
//     AddExpression(boost::make_shared<MatchLiteralExpression>(p_expr.GetAnnotations(), *child, p_expr.m_int));
// }


// void 
// FreeForm2::CopyingVisitor::Visit(const MatchCurrentWordExpression& p_expr)
// {
//     FF2_ASSERT(p_expr.GetNumChildren() == 0);
//     AddExpression(boost::make_shared<MatchCurrentWordExpression>(p_expr.GetAnnotations(), p_expr.m_offset, p_expr.m_matchType));
// }


void 
FreeForm2::CopyingVisitor::Visit(const MatchOperatorExpression& p_expr)
{
    const size_t numChildren = p_expr.GetNumChildren();
    std::vector<const MatchSubExpression*> children(numChildren);
    FF2_ASSERT(numChildren > 0);

    for (size_t i = 0; i < numChildren; i++)
    {
        children[numChildren - i - 1] 
            = boost::polymorphic_downcast<const MatchSubExpression*>(m_stack.back());
        m_stack.pop_back();
    }

    AddExpression(MatchOperatorExpression::Alloc(p_expr.GetAnnotations(),
                                                 &children[0], 
                                                 children.size(),
                                                 p_expr.GetOperator()));
}


void 
FreeForm2::CopyingVisitor::Visit(const MatchGuardExpression& p_expr)
{
    FF2_ASSERT(p_expr.GetNumChildren() == 1);
    const Expression* guard = m_stack.back();
    m_stack.pop_back();

    AddExpression(boost::make_shared<MatchGuardExpression>(p_expr.GetAnnotations(), *guard));
}



void 
FreeForm2::CopyingVisitor::Visit(const MatchBindExpression& p_expr)
{
    const MatchSubExpression* init 
        = boost::polymorphic_downcast<const MatchSubExpression*>(m_stack.back());
    m_stack.pop_back();

    AddExpression(boost::make_shared<MatchBindExpression>(p_expr.GetAnnotations(), *init, p_expr.m_id));
}


void
FreeForm2::CopyingVisitor::Visit(const MemberAccessExpression& p_expr)
{
    const Expression& container = *m_stack.back();
    m_stack.pop_back();

    // Look up the CompoundType::Member struct in the next expression type. 
    // Becase types are being copied, the member from p_expr.GetType() will not
    // be the same as the one from the expression at the top of the stack.
    FF2_ASSERT(CompoundType::IsCompoundType(container.GetType()));
    const CompoundType& type = static_cast<const CompoundType&>(container.GetType());
    const CompoundType::Member* member = type.FindMember(p_expr.GetMemberInfo().m_name);
    FF2_ASSERT(member != NULL);

    AddExpression(boost::make_shared<MemberAccessExpression>(p_expr.GetAnnotations(), container, *member, p_expr.GetVersion()));
}


//void 
//FreeForm2::CopyingVisitor::Visit(const NeuralInputResultExpression& p_expr)
//{
//    FF2_ASSERT(p_expr.GetNumChildren() == 1);
//    const Expression* child = m_stack.back();
//    m_stack.pop_back();
// 
//    AddExpression(boost::make_shared<NeuralInputResultExpression>(p_expr.GetAnnotations(), p_expr.m_index, *child));
//}


//void 
//FreeForm2::CopyingVisitor::Visit(const ObjectMethodExpression& p_expr)
//{
//    FF2_ASSERT(p_expr.GetNumChildren() == 1 + p_expr.m_numParameters);
//
//    const Expression* child = m_stack.back();
//    m_stack.pop_back();
//
//    if (child->GetType().Primitive() == Type::Object)
//    {
//        std::vector<const Expression*> parameters(p_expr.m_numParameters);
//        for (size_t i = 0; i < p_expr.m_numParameters; i++)
//        {
//            parameters[p_expr.m_numParameters - i - 1] = m_stack.back();
//            m_stack.pop_back();
//        }
//
//        // Look up the CompoundType::Member in the next expression type. 
//        // Because types are being copied, the member from p_expr.GetType() will not
//        // be the same as the one from the expression at the top of the stack.
//        FF2_ASSERT(p_expr.GetType() != child->GetType());
//        FF2_ASSERT(CompoundType::IsCompoundType(child->GetType()));
//        const CompoundType& type = static_cast<const CompoundType&>(child->GetType());
//        const CompoundType::Member* member = type.FindMember(p_expr.m_member->m_name);
//        FF2_ASSERT(member != NULL);
//        AddExpression(ObjectMethodExpression::Alloc(p_expr.GetAnnotations(), *child, *member, parameters));
//    }
//    else
//    {
//        AddExpression(ObjectMethodExpression::Alloc(p_expr.GetAnnotations(), *child, p_expr.m_method));
//    }
//}


void
FreeForm2::CopyingVisitor::Visit(const LiteralIntExpression& p_expr)
{
    AddExpression(boost::make_shared<LiteralIntExpression>(p_expr.GetAnnotations(), p_expr.GetConstantValue().m_int));
}


void
FreeForm2::CopyingVisitor::Visit(const LiteralUInt64Expression& p_expr)
{
    AddExpression(boost::make_shared<LiteralUInt64Expression>(p_expr.GetAnnotations(), p_expr.GetConstantValue().m_uint64));
}


void
FreeForm2::CopyingVisitor::Visit(const LiteralInt32Expression& p_expr)
{
    AddExpression(boost::make_shared<LiteralInt32Expression>(p_expr.GetAnnotations(), p_expr.GetConstantValue().m_int32));
}


void
FreeForm2::CopyingVisitor::Visit(const LiteralUInt32Expression& p_expr)
{
    AddExpression(boost::make_shared<LiteralUInt32Expression>(p_expr.GetAnnotations(), p_expr.GetConstantValue().m_uint32));
}


void
FreeForm2::CopyingVisitor::Visit(const ArrayLengthExpression& p_expr)
{
    const Expression* arrayLiteral = m_stack.back();
    m_stack.pop_back();

    AddExpression(boost::make_shared<ArrayLengthExpression>(p_expr.GetAnnotations(), *arrayLiteral));
}


void
FreeForm2::CopyingVisitor::Visit(const ArrayDereferenceExpression& p_expr)
{
    const Expression* index = m_stack.back();
    m_stack.pop_back();
    const Expression* arrayExpression = m_stack.back();
    m_stack.pop_back();

    AddExpression(boost::make_shared<ArrayDereferenceExpression>(p_expr.GetAnnotations(), 
                                                                 *arrayExpression, 
                                                                 *index,
                                                                 p_expr.GetVersion()));
}


void
FreeForm2::CopyingVisitor::Visit(const ConvertToFloatExpression& p_expr)
{
    const Expression* value = m_stack.back();
    m_stack.pop_back();

    AddExpression(boost::make_shared<ConvertToFloatExpression>(p_expr.GetAnnotations(), *value));
}


void
FreeForm2::CopyingVisitor::Visit(const ConvertToIntExpression& p_expr)
{
    const Expression* value = m_stack.back();
    m_stack.pop_back();

    AddExpression(boost::make_shared<ConvertToIntExpression>(p_expr.GetAnnotations(), *value));
}


void
FreeForm2::CopyingVisitor::Visit(const ConvertToUInt64Expression& p_expr)
{
    const Expression* value = m_stack.back();
    m_stack.pop_back();

    AddExpression(boost::make_shared<ConvertToUInt64Expression>(p_expr.GetAnnotations(), *value));
}


void
FreeForm2::CopyingVisitor::Visit(const ConvertToInt32Expression& p_expr)
{
    const Expression* value = m_stack.back();
    m_stack.pop_back();

    AddExpression(boost::make_shared<ConvertToInt32Expression>(p_expr.GetAnnotations(), *value));
}


void
FreeForm2::CopyingVisitor::Visit(const ConvertToUInt32Expression& p_expr)
{
    const Expression* value = m_stack.back();
    m_stack.pop_back();

    AddExpression(boost::make_shared<ConvertToUInt32Expression>(p_expr.GetAnnotations(), *value));
}


void
FreeForm2::CopyingVisitor::Visit(const ConvertToBoolExpression& p_expr)
{
    const Expression* value = m_stack.back();
    m_stack.pop_back();

    AddExpression(boost::make_shared<ConvertToBoolExpression>(p_expr.GetAnnotations(), *value));
}


void 
FreeForm2::CopyingVisitor::Visit(const ConvertToImperativeExpression& p_expr)
{
    const Expression* value = m_stack.back();
    m_stack.pop_back();

    AddExpression(boost::make_shared<ConvertToImperativeExpression>(p_expr.GetAnnotations(), *value));
}


void 
FreeForm2::CopyingVisitor::Visit(const DeclarationExpression& p_expr)
{
    const Expression& init = *m_stack.back();
    m_stack.pop_back();

    boost::shared_ptr<DeclarationExpression> expr(
        boost::make_shared<DeclarationExpression>(p_expr.GetAnnotations(),
                                                  CopyType(p_expr.GetDeclaredType()), 
                                                  init, 
                                                  p_expr.HasVoidValue(),
                                                  p_expr.GetId(),
                                                  p_expr.GetVersion()));
    AddExpression(expr);
}


void
FreeForm2::CopyingVisitor::Visit(const DirectPublishExpression& p_expr)
{
    const Expression& value = *m_stack.back();
    m_stack.pop_back();

    const size_t numIndices = p_expr.GetNumIndices();
    std::vector<const Expression*> indices(numIndices);

    for (size_t i = 0; i < numIndices; i++)
    {
        indices[numIndices - i - 1] = m_stack.back();
        m_stack.pop_back();
    }

    AddExpression(DirectPublishExpression::Alloc(p_expr.GetAnnotations(),
                                                 p_expr.GetFeatureName(),
                                                 &indices[0],
                                                 static_cast<unsigned int>(indices.size()),
                                                 value));
}


void 
FreeForm2::CopyingVisitor::Visit(const ExternExpression& p_expr)
{
    AddExpression(boost::make_shared<ExternExpression>(p_expr.GetAnnotations(),
                                                       p_expr.GetData(), 
                                                       CopyType(p_expr.GetType()), 
                                                       p_expr.GetId(),
                                                       boost::ref(*m_typeManager)));
}


void 
FreeForm2::CopyingVisitor::Visit(const FunctionExpression& p_expr)
{
    const size_t numParameters = p_expr.GetNumParameters();
    std::vector<FunctionExpression::Parameter> parameters(numParameters);

    const Expression* body = m_stack.back();
    m_stack.pop_back();

    for (size_t i = 0; i < numParameters; i++)
    {
        parameters[numParameters - i - 1]
            = p_expr.GetParameters()[numParameters - i - 1];
        parameters[numParameters - i - 1].m_parameter
            = static_cast<const VariableRefExpression*>(m_stack.back());
        m_stack.pop_back();
    }

    boost::shared_ptr<FunctionExpression> expr(new FunctionExpression(p_expr.GetAnnotations(),
                                                                      static_cast<const FunctionType&>(CopyType(p_expr.GetFunctionType())),
                                                                      p_expr.GetName(),
                                                                      parameters,
                                                                      *body));
    AddExpression(expr);
}


bool
FreeForm2::CopyingVisitor::AlternativeVisit(const FunctionCallExpression& p_expr)
{
    const size_t numParameters = p_expr.GetNumParameters();
    std::vector<const Expression*> parameters(numParameters);

    for (size_t i = 0; i < numParameters; i++)
    {
        p_expr.GetParameters()[i]->Accept(*this);
        parameters[i] = m_stack.back();
        m_stack.pop_back();
    }

    const Expression* function;
    const FunctionExpression* functionExpression = dynamic_cast<const FunctionExpression*>(&p_expr.GetFunction());

    if (functionExpression != nullptr)
    {
        if (m_functionTranslation.find(functionExpression) == m_functionTranslation.end())
        {
            p_expr.GetFunction().Accept(*this);
            m_functionTranslation.insert(std::make_pair(functionExpression, static_cast<const FunctionExpression*>(m_stack.back())));
            m_stack.pop_back();
        }

        function = m_functionTranslation[functionExpression];
    }
    else
    {
        p_expr.GetFunction().Accept(*this);
        function = m_stack.back();
        m_stack.pop_back();
    }

    AddExpression(FunctionCallExpression::Alloc(p_expr.GetAnnotations(),
                                                *function,
                                                parameters));

    return true;
}


void
FreeForm2::CopyingVisitor::Visit(const FunctionCallExpression& p_expr)
{
    // Handled in AlternativeVisit.
    FF2_UNREACHABLE();
}


void
FreeForm2::CopyingVisitor::Visit(const LiteralFloatExpression& p_expr)
{
    AddExpression(boost::make_shared<LiteralFloatExpression>(p_expr.GetAnnotations(),
                                                             p_expr.GetConstantValue().m_float));
}


void
FreeForm2::CopyingVisitor::Visit(const LiteralBoolExpression& p_expr)
{
    AddExpression(boost::make_shared<LiteralBoolExpression>(p_expr.GetAnnotations(),
                                                            p_expr.GetConstantValue().m_bool));
}


void 
FreeForm2::CopyingVisitor::Visit(const LiteralVoidExpression& p_expr)
{
    m_stack.push_back(&LiteralVoidExpression::GetInstance());
}


void 
FreeForm2::CopyingVisitor::Visit(const LiteralStreamExpression& p_expr)
{
    const size_t numChildren = p_expr.GetNumChildren();
    std::vector<const Expression*> children(numChildren);

    for (size_t i = 0; i < numChildren; i++)
    {
        children[numChildren - i - 1] = m_stack.back();
        m_stack.pop_back();
    }

    AddExpression(LiteralStreamExpression::Alloc(p_expr.GetAnnotations(),
                                                 &children[0],
                                                 numChildren,
                                                 p_expr.GetId()));
}


void 
FreeForm2::CopyingVisitor::Visit(const LiteralWordExpression& p_expr)
{
    const Expression* word = NULL;
    const Expression* offset = NULL;
    const Expression* attribute = NULL;
    const Expression* length = NULL;
    const Expression* candidate = NULL;

    if (p_expr.m_candidate != NULL)
    {
        candidate = m_stack.back();
        m_stack.pop_back();
    }

    if (p_expr.m_length != NULL)
    {
        length = m_stack.back();
        m_stack.pop_back();
    }

    if (p_expr.m_attribute != NULL)
    {
        attribute = m_stack.back();
        m_stack.pop_back();
    }

    offset = m_stack.back();
    m_stack.pop_back();
    word = m_stack.back();
    m_stack.pop_back();

    if (p_expr.m_isHeader)
    {
        FF2_ASSERT(attribute == NULL && length == NULL && candidate == NULL);
        AddExpression(
            boost::make_shared<LiteralWordExpression>(p_expr.GetAnnotations(),
                                                      *word, 
                                                      *offset,
                                                      p_expr.GetId()));
    }
    else
    {
        AddExpression(
            boost::make_shared<LiteralWordExpression>(p_expr.GetAnnotations(),
                                                      *word, 
                                                      *offset, 
                                                      attribute, 
                                                      length, 
                                                      candidate,
                                                      p_expr.GetId()));
    }
}


void
FreeForm2::CopyingVisitor::Visit(const LiteralInstanceHeaderExpression& p_expr)
{
    const Expression* instanceLength = m_stack.back();
    m_stack.pop_back();
    const Expression* rank = m_stack.back();
    m_stack.pop_back();
    const Expression* instanceCount = m_stack.back();
    m_stack.pop_back();

    AddExpression(boost::make_shared<LiteralInstanceHeaderExpression>(p_expr.GetAnnotations(),
                                                                      *instanceCount,
                                                                      *rank,
                                                                      *instanceLength));
}


void
FreeForm2::CopyingVisitor::Visit(const FeatureRefExpression& p_expr)
{
    AddExpression(boost::make_shared<FeatureRefExpression>(p_expr.GetAnnotations(),
                                                           p_expr.m_index));
}


//bool
//FreeForm2::CopyingVisitor::AlternativeVisit(const FSMExpression& p_expr)
//{
//    size_t stackSize = m_stack.size();
//    // CopyFSM copy(*this, p_expr);
//    // p_expr.Accept(copy);
//    // m_stack.push_back(&copy.GetCopy(*m_owner));
//    FF2_ASSERT(m_stack.size() == stackSize + 1);
//    return true;
//}
//
//
//void
//FreeForm2::CopyingVisitor::Visit(const FSMExpression& p_expr)
//{
//    // We handle FSMExpressions via AlternativeVisit.
//    Unreachable(__FILE__, __LINE__);
//}


void 
FreeForm2::CopyingVisitor::Visit(const FeatureSpecExpression& p_expr)
{
    const Expression& body = *m_stack.back();
    m_stack.pop_back();

    boost::shared_ptr<FeatureSpecExpression::PublishFeatureMap> featureMapCopy =
        boost::make_shared<FeatureSpecExpression::PublishFeatureMap>();

    BOOST_FOREACH (const FeatureSpecExpression::PublishFeatureMap::value_type& featureNameToType, *p_expr.GetPublishFeatureMap())
    {
        featureMapCopy->insert(FeatureSpecExpression::PublishFeatureMap::value_type(featureNameToType.first, 
                                                                                    CopyType(featureNameToType.second)));
    }

    AddExpression(boost::make_shared<FeatureSpecExpression>(p_expr.GetAnnotations(),
                                                            featureMapCopy,
                                                            body,
                                                            p_expr.GetFeatureSpecType(),
                                                            p_expr.GetType().Primitive() != Type::Void));
}


void 
FreeForm2::CopyingVisitor::Visit(const FeatureGroupSpecExpression& p_expr)
{
    std::vector<const FeatureSpecExpression*> featureSpecs;

    for (int i = 0; i < p_expr.GetFeatureSpecs().size(); ++i)
    {
        featureSpecs.insert(featureSpecs.begin(), boost::polymorphic_downcast<const FeatureSpecExpression*>(m_stack.back()));
        m_stack.pop_back();
    }

    AddExpression(boost::make_shared<FeatureGroupSpecExpression>(p_expr.GetAnnotations(),
                                                                 p_expr.GetName(),
                                                                 featureSpecs,
                                                                 p_expr.IsExtendedExperimental(),
                                                                 p_expr.IsSmallExperimental(),
                                                                 p_expr.IsBlockLevelFeature(),
                                                                 p_expr.IsBodyBlockFeature(),
                                                                 p_expr.IsForwardIndexFeature(),
                                                                 p_expr.GetMetaStreamName()));
}


void
FreeForm2::CopyingVisitor::Visit(const PhiNodeExpression& p_expr)
{
    AddExpression(PhiNodeExpression::Alloc(p_expr.GetAnnotations(),
                                           p_expr.GetVersion(),
                                           p_expr.GetIncomingVersionsCount(),
                                           p_expr.GetIncomingVersions()));
}


void
FreeForm2::CopyingVisitor::Visit(const PublishExpression& p_expr)
{
    const Expression& value = *m_stack.back();
    m_stack.pop_back();

    AddExpression(boost::make_shared<PublishExpression>(p_expr.GetAnnotations(),
                                                        p_expr.GetFeatureName(), value));
}


void
FreeForm2::CopyingVisitor::Visit(const ReturnExpression& p_expr)
{
    const Expression& value = *m_stack.back();
    m_stack.pop_back();

    AddExpression(boost::make_shared<ReturnExpression>(p_expr.GetAnnotations(),
                                                       value));
}


void 
FreeForm2::CopyingVisitor::Visit(const StreamDataExpression& p_expr)
{
    AddExpression(boost::make_shared<StreamDataExpression>(p_expr.GetAnnotations(),
                                                           p_expr.m_requestsLength));
}


void 
FreeForm2::CopyingVisitor::Visit(const UpdateStreamDataExpression& p_expr)
{
    m_stack.push_back(&UpdateStreamDataExpression::GetInstance());
}


void 
FreeForm2::CopyingVisitor::Visit(const VariableRefExpression& p_expr)
{
    AddExpression(boost::make_shared<VariableRefExpression>(p_expr.GetAnnotations(),
                                                            p_expr.GetId(),
                                                            p_expr.GetVersion(),
                                                            CopyType(p_expr.GetType())));
}


void 
FreeForm2::CopyingVisitor::Visit(const ImportFeatureExpression& p_expr)
{
    if (p_expr.GetType().Primitive() == Type::Array)
    {
        const ArrayType& type = static_cast<const ArrayType&>(p_expr.GetType());
        const std::vector<UInt32> dimensions(type.GetDimensions(), 
                                             type.GetDimensions() + type.GetDimensionCount());
        AddExpression(
            boost::make_shared<ImportFeatureExpression>(p_expr.GetAnnotations(),
                                                        p_expr.GetFeatureName(), 
                                                        dimensions,
                                                        p_expr.GetId(),
                                                        boost::ref(*m_typeManager)));
    }
    else
    {
        AddExpression(boost::make_shared<ImportFeatureExpression>(p_expr.GetAnnotations(),
                                                                  p_expr.GetFeatureName(),
                                                                  p_expr.GetId()));
    }
}


void 
FreeForm2::CopyingVisitor::Visit(const StateExpression& p_expr)
{
    boost::shared_ptr<StateExpression> expr(new StateExpression(p_expr.GetAnnotations()));
    m_owner->AddExpression(expr);

    expr->m_id = p_expr.m_id;

    // Copy state actions.
    {
        std::list<StateExpression::Action>::const_iterator iter = p_expr.m_actions.begin();
        for (; iter != p_expr.m_actions.end(); ++iter)
        {
            iter->m_action->Accept(*this);
            StateExpression::Action a = { iter->m_matchType, m_stack.back() };
            m_stack.pop_back();
            expr->m_actions.push_back(a);
        }
    }

    // Copy state transitions.
    {
        std::list<StateExpression::Transition>::const_iterator iter = p_expr.m_transitions.begin();
        for (; iter != p_expr.m_transitions.end(); ++iter)
        {
            StateExpression::Transition t = { iter->m_matchType, NULL, iter->m_destinationId, NULL };
            if (iter->m_condition)
            {
                iter->m_condition->Accept(*this);
                t.m_condition = m_stack.back();
                m_stack.pop_back();
            }

            if (iter->m_leavingAction)
            {
                iter->m_leavingAction->Accept(*this);
                t.m_leavingAction = m_stack.back();
                m_stack.pop_back();
            }

            expr->m_transitions.push_back(t);
        }
    }

    m_stack.push_back(expr.get());
}


bool
FreeForm2::CopyingVisitor::AlternativeVisit(const StateMachineExpression& p_expr)
{
    FF2_ASSERT(p_expr.GetType().Primitive() == Type::StateMachine);
    const TypeImpl& copiedType = CopyType(p_expr.GetType());
    const StateMachineType& type = static_cast<const StateMachineType&>(copiedType);

    // The state machine has already been copied; do not re-copy.
    if (type.HasDefinition())
    {
        boost::shared_ptr<const StateMachineExpression> expr = type.GetDefinition();
        FF2_ASSERT(expr.get() != nullptr);

        m_stack.push_back(expr.get());
    }
    else
    {
        FF2_ASSERT(type.IsSameAs(type, false));
        p_expr.GetInitializer().Accept(*this);
        const TypeInitializerExpression& init 
            = *boost::polymorphic_downcast<const TypeInitializerExpression*>(m_stack.back());
        m_stack.pop_back();

        const size_t numStates = p_expr.GetNumChildren() - 1;
        std::vector<const StateExpression*> states;
        states.reserve(numStates);
        for (size_t i = 0; i < numStates; i++)
        {
            p_expr.GetChildren()[i]->Accept(*this);
            states.push_back(boost::polymorphic_downcast<const StateExpression*>(m_stack.back()));
            m_stack.pop_back();
        }

        boost::shared_ptr<Expression> expr(
            StateMachineExpression::Alloc(p_expr.GetAnnotations(),
                                          type, 
                                          init, 
                                          states.size() > 0 ? &states[0] : NULL, 
                                          states.size(), 
                                          p_expr.GetStartStateId()));
        AddExpression(expr);
        FF2_ASSERT(type.HasDefinition());
    }
    return true;
}


void 
FreeForm2::CopyingVisitor::Visit(const StateMachineExpression& p_expr)
{
    // Handled by AlternativeVisit.
    FF2_UNREACHABLE();
}


void
FreeForm2::CopyingVisitor::Visit(const ExecuteStreamRewritingStateMachineGroupExpression& p_expr)
{
    std::vector<ExecuteStreamRewritingStateMachineGroupExpression::MachineInstance> machineInstances(p_expr.GetNumMachineInstances());

    for (size_t i = 0; i < p_expr.GetNumMachineInstances(); ++i)
    {
        size_t index = p_expr.GetNumMachineInstances() - i - 1;
        machineInstances[index].m_machineExpression = boost::polymorphic_downcast<const ExecuteMachineExpression*>(m_stack.back());
        m_stack.pop_back();
        machineInstances[index].m_machineDeclaration = boost::polymorphic_downcast<const DeclarationExpression*>(m_stack.back());
        m_stack.pop_back();
    }

    const Expression* duplicateTermInformation = nullptr;

    if (p_expr.GetDuplicateTermInformation() != nullptr)
    {
        p_expr.GetDuplicateTermInformation()->Accept(*this);
        duplicateTermInformation = m_stack.back();
        m_stack.pop_back();
    }

    const Expression* numQueryPaths = nullptr;

    if (p_expr.GetNumQueryPaths() != nullptr)
    {
        p_expr.GetNumQueryPaths()->Accept(*this);
        numQueryPaths = m_stack.back();
        m_stack.pop_back();
    }

    const Expression* queryPathCandidates = nullptr;

    if (p_expr.GetQueryPathCandidates() != nullptr)
    {
        p_expr.GetQueryPathCandidates()->Accept(*this);
        queryPathCandidates = m_stack.back();
        m_stack.pop_back();
    }

    const Expression* queryLength = nullptr;

    if (p_expr.GetQueryLength() != nullptr)
    {
        p_expr.GetQueryLength()->Accept(*this);
        queryLength = m_stack.back();
        m_stack.pop_back();
    }

    const Expression* tupleOfInterestCount = nullptr;

    if (p_expr.GetTupleOfInterestCount() != nullptr)
    {
        p_expr.GetTupleOfInterestCount()->Accept(*this);
        tupleOfInterestCount = m_stack.back();
        m_stack.pop_back();
    }

    const Expression* tuplesOfInterest = nullptr;

    if (p_expr.GetTuplesOfInterest() != nullptr)
    {
        p_expr.GetTuplesOfInterest()->Accept(*this);
        tuplesOfInterest = m_stack.back();
        m_stack.pop_back();
    }

    AddExpression(ExecuteStreamRewritingStateMachineGroupExpression::Alloc(p_expr.GetAnnotations(),
                                                                           &machineInstances[0],
                                                                           static_cast<unsigned int>(machineInstances.size()),
                                                                           p_expr.GetNumBound(),
                                                                           p_expr.GetMachineIndexID(),
                                                                           p_expr.GetMachineArraySize(),
                                                                           p_expr.GetStreamRewritingType(),
                                                                           duplicateTermInformation,
                                                                           numQueryPaths,
                                                                           queryPathCandidates,
                                                                           queryLength,
                                                                           tupleOfInterestCount,
                                                                           tuplesOfInterest,
                                                                           p_expr.IsNearChunk(),
                                                                           p_expr.GetMinChunkNumber()));
}


void
FreeForm2::CopyingVisitor::Visit(const ExecuteMachineExpression& p_expr)
{
    std::vector<std::pair<std::string, const Expression*>> yieldActions(p_expr.GetNumYieldActions());

    for (size_t i = 0; i < p_expr.GetNumYieldActions(); ++i)
    {
        size_t index = p_expr.GetNumYieldActions() - i - 1;
        yieldActions[index].first = p_expr.GetYieldActions()[index].first;
        yieldActions[index].second = m_stack.back();
        m_stack.pop_back();
    }

    const Expression& machine = *m_stack.back();
    m_stack.pop_back();

    const Expression& stream = *m_stack.back();
    m_stack.pop_back();

    AddExpression(ExecuteMachineExpression::Alloc(p_expr.GetAnnotations(),
                                                  stream, 
                                                  machine, 
                                                  yieldActions.size() > 0 ? &yieldActions[0] : NULL, 
                                                  yieldActions.size()));
}


void
FreeForm2::CopyingVisitor::Visit(const ExecuteMachineGroupExpression& p_expr)
{
    std::vector<ExecuteMachineGroupExpression::MachineInstance> machineInstances(p_expr.GetNumMachineInstances());

    for (size_t i = 0; i < p_expr.GetNumMachineInstances(); ++i)
    {
        size_t index = p_expr.GetNumMachineInstances() - i - 1;
        machineInstances[index].m_machineExpression = boost::polymorphic_downcast<const ExecuteMachineExpression*>(m_stack.back());
        m_stack.pop_back();
        machineInstances[index].m_machineDeclaration = m_stack.back();
        m_stack.pop_back();
    }

    AddExpression(ExecuteMachineGroupExpression::Alloc(p_expr.GetAnnotations(),
                                                       &machineInstances[0],
                                                       static_cast<unsigned int>(machineInstances.size()),
                                                       p_expr.GetNumBound()));
}


void
FreeForm2::CopyingVisitor::Visit(const YieldExpression& p_expr)
{
    AddExpression(boost::make_shared<YieldExpression>(p_expr.GetAnnotations(),
                                                      p_expr.GetMachineName(),
                                                      p_expr.GetName()));
}


void
FreeForm2::CopyingVisitor::Visit(const RandFloatExpression& p_expr)
{
    m_stack.push_back(&RandFloatExpression::GetInstance());
}


void
FreeForm2::CopyingVisitor::Visit(const RandIntExpression& p_expr)
{
    const Expression& upperBound = *m_stack.back();
    m_stack.pop_back();
    const Expression& lowerBound = *m_stack.back();
    m_stack.pop_back();

    AddExpression(boost::make_shared<RandIntExpression>(p_expr.GetAnnotations(),
                                                        lowerBound,
                                                        upperBound));
}


void
FreeForm2::CopyingVisitor::Visit(const ThisExpression& p_expr)
{
    FF2_ASSERT(CompoundType::IsCompoundType(p_expr.GetType()) 
        || p_expr.GetType().Primitive() == Type::Unknown);
    AddExpression(boost::make_shared<ThisExpression>(p_expr.GetAnnotations(),
                                                     CopyType(p_expr.GetType())));
}


void
FreeForm2::CopyingVisitor::Visit(const UnresolvedAccessExpression& p_expr)
{
    const Expression& object = *m_stack.back();
    m_stack.pop_back();
    AddExpression(boost::make_shared<UnresolvedAccessExpression>(p_expr.GetAnnotations(),
                                                                 object, 
                                                                 p_expr.GetMemberName(),
                                                                 CopyType(p_expr.GetType())));
}


bool
FreeForm2::CopyingVisitor::AlternativeVisit(const TypeInitializerExpression& p_expr)
{
    FF2_ASSERT(CompoundType::IsCompoundType(p_expr.GetType()));
    const CompoundType& type = static_cast<const CompoundType&>(CopyType(p_expr.GetType()));

    if (type.Primitive() == Type::StateMachine)
    {
        const StateMachineType& stateMachine = static_cast<const StateMachineType&>(type);
        if (stateMachine.HasDefinition())
        {
            const StateMachineExpression& expr = *stateMachine.GetDefinition();
            m_stack.push_back(&expr.GetInitializer());
            return true;
        }
    }

    std::vector<TypeInitializerExpression::Initializer> inits(p_expr.BeginInitializers(), 
                                                                p_expr.EndInitializers());

    BOOST_FOREACH(TypeInitializerExpression::Initializer& init, inits)
    {
        const CompoundType::Member* member = type.FindMember(init.m_member->m_name);
        FF2_ASSERT(member != NULL);
        init.m_member = member;

        init.m_initializer->Accept(*this);
        init.m_initializer = m_stack.back();
        m_stack.pop_back();
    }

    AddExpression(TypeInitializerExpression::Alloc(p_expr.GetAnnotations(),
                                                    type,
                                                    inits.size() > 0 ? &inits[0] : NULL,
                                                    inits.size()));
    return true;
}


void
FreeForm2::CopyingVisitor::Visit(const TypeInitializerExpression& p_expr)
{
    // Handled by AlternativeVisit.
    Unreachable(__FILE__, __LINE__);
}


void
FreeForm2::CopyingVisitor::Visit(const AggregateContextExpression& p_expr)
{
    const Expression& body = *m_stack.back();
    m_stack.pop_back();

    AddExpression(boost::make_shared<AggregateContextExpression>(p_expr.GetAnnotations(), body));
}


void
FreeForm2::CopyingVisitor::Visit(const DebugExpression& p_expr)
{
    const Expression& child = *m_stack.back();
    m_stack.pop_back();

    AddExpression(boost::make_shared<DebugExpression>(p_expr.GetAnnotations(), child, p_expr.GetChildText()));
}


void
FreeForm2::CopyingVisitor::VisitReference(const ArrayDereferenceExpression& p_expr)
{
    Visit(p_expr);
}


void 
FreeForm2::CopyingVisitor::VisitReference(const VariableRefExpression& p_expr)
{
    Visit(p_expr);
}


void 
FreeForm2::CopyingVisitor::VisitReference(const MemberAccessExpression& p_expr)
{
    Visit(p_expr);
}


void 
FreeForm2::CopyingVisitor::VisitReference(const ThisExpression& p_expr)
{
    Visit(p_expr);
}


void 
FreeForm2::CopyingVisitor::VisitReference(const UnresolvedAccessExpression& p_expr)
{
    Visit(p_expr);
}


size_t 
FreeForm2::CopyingVisitor::StackSize() const 
{
    return m_stack.size();
}


size_t 
FreeForm2::CopyingVisitor::StackIncrement() const 
{
    return 1;
}
 

const FreeForm2::TypeImpl& 
FreeForm2::CopyingVisitor::CopyType(const TypeImpl& p_type)
{
    switch (p_type.Primitive())
    {
        case Type::Float:
        {
            FF2_ASSERT(&p_type == &TypeImpl::GetFloatInstance(p_type.IsConst()));
            return p_type;
        }

        case Type::Int:
        {
            FF2_ASSERT(&p_type == &TypeImpl::GetIntInstance(p_type.IsConst()));
            return p_type;
        }

        case Type::UInt64:
        {
            FF2_ASSERT(&p_type == &TypeImpl::GetUInt64Instance(p_type.IsConst()));
            return p_type;
        }

        case Type::Int32:
        {
            FF2_ASSERT(&p_type == &TypeImpl::GetInt32Instance(p_type.IsConst()));
            return p_type;
        }

        case Type::UInt32:
        {
            FF2_ASSERT(&p_type == &TypeImpl::GetUInt32Instance(p_type.IsConst()));
            return p_type;
        }

        case Type::Bool:
        {
            FF2_ASSERT(&p_type == &TypeImpl::GetBoolInstance(p_type.IsConst()));
            return p_type;
        }

        case Type::Array:
        {
            const ArrayType& type = static_cast<const ArrayType&>(p_type);
            return m_typeManager->GetArrayType(type);
        }

        case Type::Struct:
        {
            const StructType& type = static_cast<const StructType&>(p_type);
            return m_typeManager->GetStructType(type);
        }

        case Type::Object:
        {
            const ObjectType& type = static_cast<const ObjectType&>(p_type);
            return m_typeManager->GetObjectType(type);
        }

        case Type::Function:
        {
            const FunctionType& type = static_cast<const FunctionType&>(p_type);
            return m_typeManager->GetFunctionType(type);
        }

        case Type::StateMachine:
        {
            // Check if the type has already been copied.
            const StateMachineType& type = static_cast<const StateMachineType&>(p_type);
            const TypeImpl* copied = m_typeManager->GetTypeInfo(p_type.GetName());
            if (copied != NULL)
            {
                FF2_ASSERT(copied->Primitive() == Type::StateMachine);
                return *copied;
            }
            else
            {
                // Copy the type without the implementing expression.
                std::vector<CompoundType::Member> members(type.BeginMembers(), type.EndMembers());
                BOOST_FOREACH (CompoundType::Member& member, members)
                {
                    // Prevent self-reference.
                    FF2_ASSERT(!member.m_type->IsSameAs(type, true));

                    member.m_type = &CopyType(*member.m_type);
                }
                const StateMachineType& copiedType 
                    = m_typeManager->GetStateMachineType(type.GetName(), 
                                                         members.size() > 0 ? &members[0] : NULL, 
                                                         members.size(), 
                                                         boost::weak_ptr<StateMachineExpression>());

                // Copy the StateMachineExpression, which should set the definition.
                type.GetDefinition()->Accept(*this);
                FF2_ASSERT(&m_stack.back()->GetType() == &copiedType);
                FF2_ASSERT(copiedType.GetDefinition().get() != NULL);
                m_stack.pop_back();
                return copiedType;
            }
        }

        case Type::Void:
        {
            FF2_ASSERT(&p_type == &TypeImpl::GetVoidInstance());
            return p_type;
        }

        case Type::Stream:
        {
            FF2_ASSERT(&p_type == &TypeImpl::GetStreamInstance(p_type.IsConst()));
            return p_type;
        }

        case Type::Word:
        {
            FF2_ASSERT(&p_type == &TypeImpl::GetWordInstance(p_type.IsConst()));
            return p_type;
        }

        case Type::InstanceHeader:
        {
            FF2_ASSERT(&p_type == &TypeImpl::GetInstanceHeaderInstance(p_type.IsConst()));
            return p_type;
        }

        case Type::BodyBlockHeader:
        {
            FF2_ASSERT(&p_type == &TypeImpl::GetBodyBlockHeaderInstance(p_type.IsConst()));
            return p_type;
        }

        case Type::Unknown:
        {
            FF2_ASSERT(&p_type == &TypeImpl::GetUnknownType());
            return p_type;
        }

        case Type::Invalid:
        {
            FF2_ASSERT(&p_type == &TypeImpl::GetInvalidType());
            return p_type;
        }

        default:
        {
            Unreachable(__FILE__, __LINE__);
        }
    }
}
