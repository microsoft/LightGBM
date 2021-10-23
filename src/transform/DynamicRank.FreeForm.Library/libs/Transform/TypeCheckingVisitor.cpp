#include "TypeCheckingVisitor.h"

#include "ArrayType.h"
#include "BlockExpression.h"
#include <boost/foreach.hpp>
#include "Conditional.h"
#include "ConvertExpression.h"
#include "DebugExpression.h"
#include "Declaration.h"
#include "Expression.h"
#include "FeatureSpec.h"
#include "FreeForm2Assert.h"
#include "Function.h"
#include "LetExpression.h"
#include "LiteralExpression.h"
#include "Match.h"
#include "Mutation.h"
#include "NoOpVisitor.h"
#include "FunctionType.h"
#include "Publish.h"
#include "RangeReduceExpression.h"
#include "RefExpression.h"
#include <sstream>
#include "StateMachine.h"
#include "TypeImpl.h"
#include "TypeUtil.h"

FreeForm2::TypeCheckingVisitor::TypeCheckingVisitor()
    : m_lastExpressionReturns(true),
      m_hasSideEffects(false)
{
}


void
FreeForm2::TypeCheckingVisitor::AssertSideEffects(const FreeForm2::SourceLocation& p_sourceLocation) const
{
    if (!m_hasSideEffects)
    {
        std::ostringstream err;
        err << "Statement does not have side effects.";
        throw ParseError(err.str(), p_sourceLocation);
    }
}


bool 
FreeForm2::TypeCheckingVisitor::AlternativeVisit(const BlockExpression& p_expr)
{
    unsigned int numChildren = static_cast<unsigned int>(p_expr.GetNumChildren());

    if (m_lastExpressionReturns)
    {
        numChildren--;
    }

    for (unsigned int i = 0; i < numChildren; i++)
    {
        const FreeForm2::Expression& child = p_expr.GetChild(i);

        if (m_functions.size() > 0 && m_functions.top().m_allPathsReturn)
        {
            std::ostringstream err;
            err << "Unreachable code detected.";
            throw ParseError(err.str(), child.GetSourceLocation());
        }

        m_hasSideEffects = false;
        child.Accept(*this);
        AssertSideEffects(child.GetSourceLocation());
    }

    if (m_lastExpressionReturns)
    {
        if (m_functions.size() > 0 && m_functions.top().m_allPathsReturn)
        {
            std::ostringstream err;
            err << "Unreachable code detected.";
            throw ParseError(err.str(), p_expr.GetChild(numChildren).GetSourceLocation());
        }

        p_expr.GetChild(numChildren).Accept(*this);
    }

    UniformExpressionVisitor::Visit(p_expr);
    return true;
}


bool
FreeForm2::TypeCheckingVisitor::AlternativeVisit(const ConditionalExpression& p_expr) 
{
    const bool lastSiblingReturns = m_lastExpressionReturns;
    const bool conditionalReturnsValue = p_expr.GetType().Primitive() != Type::Void;

    // We don't care if the condition has side effects, only the then and 
    // else clauses.
    m_lastExpressionReturns = conditionalReturnsValue;
    m_hasSideEffects = false;
    p_expr.GetThen().Accept(*this);
    if (!(m_hasSideEffects || m_lastExpressionReturns))
    {
        AssertSideEffects(p_expr.GetThen().GetSourceLocation());
    }

    const bool ifExprReturns = m_functions.size() > 0 && m_functions.top().m_allPathsReturn;

    if (ifExprReturns)
    {
        m_functions.top().m_allPathsReturn = false;
    }

    if (&p_expr.GetElse() != &FreeForm2::LiteralVoidExpression::GetInstance())
    {
        m_lastExpressionReturns = conditionalReturnsValue;
        m_hasSideEffects = false;
        p_expr.GetElse().Accept(*this);
        if (!(m_hasSideEffects || m_lastExpressionReturns))
        {
            AssertSideEffects(p_expr.GetElse().GetSourceLocation());
        }

        if (m_functions.size() > 0)
        {
            m_functions.top().m_allPathsReturn &= ifExprReturns;
        }
    }

    UniformExpressionVisitor::Visit(p_expr);

    m_lastExpressionReturns = lastSiblingReturns;
    return true;
}


bool
FreeForm2::TypeCheckingVisitor::AlternativeVisit(const MatchExpression& p_expr)
{
    const bool lastSiblingReturns = m_lastExpressionReturns;

    p_expr.GetPattern().Accept(*this);
    p_expr.GetValue().Accept(*this);

    // Ignore the value and pattern of the match; they are not expected to
    // have side effects.
    m_lastExpressionReturns = false;
    m_hasSideEffects = false;
    p_expr.GetAction().Accept(*this);
    AssertSideEffects(p_expr.GetAction().GetSourceLocation());

    UniformExpressionVisitor::Visit(p_expr);

    m_lastExpressionReturns = lastSiblingReturns;
    return true;
}


bool
FreeForm2::TypeCheckingVisitor::AlternativeVisit(const ConvertToImperativeExpression& p_expr)
{
    m_hasSideEffects = false;
    p_expr.GetChild().Accept(*this);
    AssertSideEffects(p_expr.GetChild().GetSourceLocation());
    UniformExpressionVisitor::Visit(p_expr);
    return true;
}


bool 
FreeForm2::TypeCheckingVisitor::AlternativeVisit(const RangeReduceExpression& p_expr)
{
    FF2_ASSERT(m_variableTypes.find(p_expr.GetReduceId()) == m_variableTypes.end()
        && m_variableTypes.find(p_expr.GetStepId()) == m_variableTypes.end());
    const auto reduceKey = p_expr.GetReduceId();
    m_variableTypes.insert(
        std::make_pair(reduceKey, &p_expr.GetReduceExpression().GetType()));
    const auto stepKey = p_expr.GetStepId();
    m_variableTypes.insert(
        std::make_pair(stepKey, &p_expr.GetLow().GetType()));

    const bool lastSiblingReturns = m_lastExpressionReturns;

    // Range-reduce expression only returns something if the reduction 
    // variable is non-void.
    m_lastExpressionReturns = p_expr.GetType().Primitive() != Type::Void;

    p_expr.GetReduceExpression().Accept(*this);
    m_hasSideEffects |= m_lastExpressionReturns;
    AssertSideEffects(p_expr.GetReduceExpression().GetSourceLocation());

    m_lastExpressionReturns = lastSiblingReturns;

    m_variableTypes.erase(reduceKey);
    m_variableTypes.erase(stepKey);

    UniformExpressionVisitor::Visit(p_expr);
    return true;
}


bool 
FreeForm2::TypeCheckingVisitor::AlternativeVisit(const ForEachLoopExpression& p_expr)
{
    FF2_ASSERT(m_variableTypes.find(p_expr.GetIteratorId()) == m_variableTypes.end());
    const auto insertKey = p_expr.GetIteratorId();
    m_variableTypes.insert(
        std::make_pair(insertKey, &p_expr.GetIteratorType()));

    const bool lastSiblingReturns = m_lastExpressionReturns;
    m_lastExpressionReturns = false;
    p_expr.GetBody().Accept(*this);
    AssertSideEffects(p_expr.GetBody().GetSourceLocation());
    m_lastExpressionReturns = lastSiblingReturns;

    m_variableTypes.erase(insertKey);

    UniformExpressionVisitor::Visit(p_expr);
    return true;
}


bool 
FreeForm2::TypeCheckingVisitor::AlternativeVisit(const ComplexRangeLoopExpression& p_expr)
{
    FF2_ASSERT(m_variableTypes.find(p_expr.GetStepId()) == m_variableTypes.end());
    const auto insertKey = p_expr.GetStepId();
    m_variableTypes.insert(std::make_pair(insertKey, &p_expr.GetStepType()));

    const bool lastSiblingReturns = m_lastExpressionReturns;
    m_lastExpressionReturns = false;
    p_expr.GetBody().Accept(*this);
    AssertSideEffects(p_expr.GetBody().GetSourceLocation());
    m_lastExpressionReturns = lastSiblingReturns;

    m_variableTypes.erase(insertKey);

    UniformExpressionVisitor::Visit(p_expr);
    return true;
}


bool 
FreeForm2::TypeCheckingVisitor::AlternativeVisit(const AggregateContextExpression& p_expr)
{
    const bool lastSiblingReturns = m_lastExpressionReturns;
    m_lastExpressionReturns = false;
    p_expr.GetBody().Accept(*this);
    AssertSideEffects(p_expr.GetBody().GetSourceLocation());
    m_lastExpressionReturns = lastSiblingReturns;

    UniformExpressionVisitor::Visit(p_expr);
    return true;
}


bool 
FreeForm2::TypeCheckingVisitor::AlternativeVisit(const FunctionExpression& p_expr)
{
    // Save the state of the name->type mapping to restore it after the function body
    // is evaluated.
    const auto savedVariableTypesMap = m_variableTypes;

    // Bind all parameters for type checking purposes.
    BOOST_FOREACH (auto& param, p_expr.GetParameters())
    {
        FF2_ASSERT(m_variableTypes.find(param.m_parameter->GetId()) == m_variableTypes.end());
        m_variableTypes.insert(std::make_pair(param.m_parameter->GetId(), &param.m_parameter->GetType()));
    }

    FunctionState functionState;
    functionState.m_returnType = &p_expr.GetFunctionType().GetReturnType();
	functionState.m_allPathsReturn = false;

    m_functions.push(functionState);
    
    p_expr.GetBody().Accept(*this);

    UniformExpressionVisitor::Visit(p_expr);

    if (!m_functions.top().m_allPathsReturn)
    {
        std::ostringstream err;
        err << "Not all code paths return a value.";
        throw ParseError(err.str(), p_expr.GetSourceLocation());
    }

    m_functions.pop();
    
    // Restore the name->type mapping.
    m_variableTypes = savedVariableTypesMap;

    return true;
}


bool 
FreeForm2::TypeCheckingVisitor::AlternativeVisit(const FunctionCallExpression& p_expr)
{
    const FunctionType& functionType = p_expr.GetFunctionType();

    FF2_ASSERT(p_expr.GetNumParameters() == functionType.GetParameterCount());

    for (unsigned int i = 0; i < functionType.GetParameterCount(); ++i)
    {
        p_expr.GetParameters()[i]->Accept(*this);

        // If a parameter in a function is marked as mutable, the parameter passed
        // into a function call must be a mutable l-value.
        if (!functionType.BeginParameters()[i]->IsConst())
        {
            if (p_expr.GetParameters()[i]->GetType().IsConst())
            {
                throw ParseError("Parameter must be a mutable l-value.",
                        p_expr.GetParameters()[i]->GetSourceLocation());
            }

            // Ref-parameters must be variable ref expressions for the moment.
            const VariableRefExpression* refExpression
                = dynamic_cast<const VariableRefExpression*>(p_expr.GetParameters()[i]);

            if (refExpression == nullptr)
            {
                throw ParseError("Parameter must be a non-array variable name.",
                        p_expr.GetParameters()[i]->GetSourceLocation());
            }
        }
    }

    p_expr.GetFunction().Accept(*this);

    return true;
}


bool 
FreeForm2::TypeCheckingVisitor::AlternativeVisit(const FeatureSpecExpression& p_expr)
{
    m_publishFeatureMap = p_expr.GetPublishFeatureMap().get();
    
    m_lastExpressionReturns = p_expr.GetType().Primitive() != Type::Void;
    p_expr.GetBody().Accept(*this);

    UniformExpressionVisitor::Visit(p_expr);
    return true;
}


bool
FreeForm2::TypeCheckingVisitor::AlternativeVisit(const StateMachineExpression& p_expr)
{
    m_hasSideEffects = true;
    UniformExpressionVisitor::Visit(p_expr);
    return true;
}


bool 
FreeForm2::TypeCheckingVisitor::AlternativeVisit(const LetExpression& p_expr)
{
    for (unsigned int i = 0; i < p_expr.GetNumChildren() - 1; i++)
    {
        FF2_ASSERT(m_variableTypes.find(p_expr.GetBound()[i].first) == m_variableTypes.end());
        m_variableTypes.insert(
            std::make_pair(p_expr.GetBound()[i].first, &p_expr.GetBound()[i].second->GetType()));
    }

    const bool lastSiblingReturns = m_lastExpressionReturns;
    m_lastExpressionReturns = true;
    p_expr.GetValue().Accept(*this);
    m_lastExpressionReturns = lastSiblingReturns;

    for (unsigned int i = 0; i < p_expr.GetNumChildren() - 1; i++)
    {
        auto find = m_variableTypes.find(p_expr.GetBound()[i].first);
        FF2_ASSERT(find != m_variableTypes.end());
        m_variableTypes.erase(find);
    }

    UniformExpressionVisitor::Visit(p_expr);
    return true;
}


bool
FreeForm2::TypeCheckingVisitor::AlternativeVisit(const ExecuteStreamRewritingStateMachineGroupExpression& p_expr) 
{
    const bool lastSiblingReturns = m_lastExpressionReturns;
    m_lastExpressionReturns = false;

    // Process all machine declarations before processing machine execution 
    // expressions, otherwise machine execution will reference undeclared 
    // variables.
    for (unsigned int i = 0; i < p_expr.GetNumMachineInstances(); i++)
    {
        m_hasSideEffects = false;
        p_expr.GetMachineInstances()[i].m_machineDeclaration->Accept(*this);
        AssertSideEffects(p_expr.GetMachineInstances()[i].m_machineDeclaration->GetSourceLocation());
    }

    FF2_ASSERT(m_variableTypes.find(p_expr.GetMachineIndexID()) == m_variableTypes.end());
    const auto insertKey = p_expr.GetMachineIndexID();
    m_variableTypes.insert(std::make_pair(insertKey, &FreeForm2::TypeImpl::GetUInt32Instance(true)));

    for (unsigned int i = 0; i < p_expr.GetNumMachineInstances(); i++)
    {
        p_expr.GetMachineInstances()[i].m_machineExpression->Accept(*this);
    }

    m_variableTypes.erase(insertKey);

    m_lastExpressionReturns = lastSiblingReturns;
    
    return true;
}


bool
FreeForm2::TypeCheckingVisitor::AlternativeVisit(const ExecuteMachineGroupExpression& p_expr) 
{
    const bool lastSiblingReturns = m_lastExpressionReturns;
    m_lastExpressionReturns = false;

    // Process all machine declarations before processing machine execution 
    // expressions, otherwise machine execution will reference undeclared 
    // variables.
    for (unsigned int i = 0; i < p_expr.GetNumMachineInstances(); i++)
    {
        m_hasSideEffects = false;
        p_expr.GetMachineInstances()[i].m_machineDeclaration->Accept(*this);
        AssertSideEffects(p_expr.GetMachineInstances()[i].m_machineDeclaration->GetSourceLocation());
    }

    for (unsigned int i = 0; i < p_expr.GetNumMachineInstances(); i++)
    {
        p_expr.GetMachineInstances()[i].m_machineExpression->Accept(*this);
    }

    m_lastExpressionReturns = lastSiblingReturns;
    return true;
}


void
FreeForm2::TypeCheckingVisitor::Visit(const MutationExpression& p_expr) 
{
    m_hasSideEffects = true;
    UniformExpressionVisitor::Visit(p_expr);
} 


void
FreeForm2::TypeCheckingVisitor::Visit(const DeclarationExpression& p_expr)
{
    FF2_ASSERT(m_variableTypes.find(p_expr.GetId()) == m_variableTypes.end());
    m_variableTypes.insert(std::make_pair(p_expr.GetId(), &p_expr.GetDeclaredType()));

    m_hasSideEffects = true;
    UniformExpressionVisitor::Visit(p_expr);
}


void
FreeForm2::TypeCheckingVisitor::Visit(const DirectPublishExpression& p_expr)
{
    m_hasSideEffects = true;
    
    FF2_ASSERT(m_publishFeatureMap != NULL);
    FeatureSpecExpression::PublishFeatureMap::const_iterator featureNameToType = 
        m_publishFeatureMap->find(FeatureSpecExpression::FeatureName(p_expr.GetFeatureName()));

    // The lemon file should have already checked that the feature names
    // being published are valid.
    FF2_ASSERT(featureNameToType != m_publishFeatureMap->end());
    FF2_ASSERT(featureNameToType->second.Primitive() == Type::Array);

    const ArrayType& type = static_cast<const ArrayType&>(featureNameToType->second);

    if (!TypeUtil::IsAssignable(type.GetChildType(), p_expr.GetValue().GetType()))
    {
        std::ostringstream err;
        err << "Invalid publish type: " << p_expr.GetValue().GetType()
            << "; expected type: " << type.GetChildType();
        throw ParseError(err.str(), p_expr.GetSourceLocation());
    }

    UniformExpressionVisitor::Visit(p_expr);
}


void
FreeForm2::TypeCheckingVisitor::Visit(const PublishExpression& p_expr)
{
    m_hasSideEffects = true;
    
    FF2_ASSERT(m_publishFeatureMap != NULL);
    FeatureSpecExpression::PublishFeatureMap::const_iterator featureNameToType = 
        m_publishFeatureMap->find(FeatureSpecExpression::FeatureName(p_expr.GetFeatureName()));

    // The lemon file should have already checked that the feature names
    // being published are valid.
    FF2_ASSERT(featureNameToType != m_publishFeatureMap->end());

    if (!TypeUtil::IsAssignable(featureNameToType->second, p_expr.GetValue().GetType()))
    {
        std::ostringstream err;
        err << "Invalid publish type: " << p_expr.GetValue().GetType()
            << "; expected type: " << featureNameToType->second;
        throw ParseError(err.str(), p_expr.GetSourceLocation());
    }

    UniformExpressionVisitor::Visit(p_expr);
}


void
FreeForm2::TypeCheckingVisitor::Visit(const ReturnExpression& p_expr)
{
    if (m_functions.size() == 0)
    {
        std::ostringstream err;
        err << "Return statements can only be used in functions.";
        throw ParseError(err.str(), p_expr.GetSourceLocation());
    }

    if (!TypeUtil::IsAssignable(*m_functions.top().m_returnType, p_expr.GetValue().GetType()))
    {
        std::ostringstream err;
        err << "Invalid return type: " << p_expr.GetValue().GetType()
            << "; expected type: " << *m_functions.top().m_returnType;
        throw ParseError(err.str(), p_expr.GetSourceLocation());
    }

    m_hasSideEffects = true;
    m_functions.top().m_allPathsReturn = true;
    UniformExpressionVisitor::Visit(p_expr);
}


void
FreeForm2::TypeCheckingVisitor::Visit(const ImportFeatureExpression& p_expr)
{
    FF2_ASSERT(m_variableTypes.find(p_expr.GetId()) == m_variableTypes.end());
    m_variableTypes.insert(std::make_pair(p_expr.GetId(), &p_expr.GetType()));

    m_hasSideEffects = true;
    UniformExpressionVisitor::Visit(p_expr);
}


void 
FreeForm2::TypeCheckingVisitor::Visit(const VariableRefExpression& p_expr)
{
    auto find = m_variableTypes.find(p_expr.GetId());

    if (find == m_variableTypes.end())
    {
        std::ostringstream err;
        err << "Variable referenced before its declaration (ID "
            << p_expr.GetId() << ")";
        throw ParseError(err.str(), p_expr.GetSourceLocation());
    }

    FF2_ASSERT(find->second != nullptr);
    if (!find->second->IsSameAs(p_expr.GetType(), true))
    {
        std::ostringstream err;
        err << "Variable declaration and reference have different types. "
            << "Got " << *find->second << " and " << p_expr.GetType()
            << " for the declaration and reference, respectively "
            << "(ID " << p_expr.GetId() << ")";
        throw ParseError(err.str(), p_expr.GetSourceLocation());
    }

    UniformExpressionVisitor::Visit(p_expr);
}


void
FreeForm2::TypeCheckingVisitor::Visit(const DebugExpression& p_expr)
{
    m_hasSideEffects = true;
    UniformExpressionVisitor::Visit(p_expr);
}


void
FreeForm2::TypeCheckingVisitor::VisitReference(const VariableRefExpression& p_expr)
{
    Visit(p_expr);
}


void 
FreeForm2::TypeCheckingVisitor::Visit(const Expression& p_expr)
{
    const TypeImpl& type = p_expr.GetType();
    FF2_ASSERT(type.Primitive() != Type::Unknown);
    FF2_ASSERT(type.IsValid());
}
