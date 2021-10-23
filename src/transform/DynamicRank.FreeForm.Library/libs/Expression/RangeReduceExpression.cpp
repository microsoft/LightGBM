#include "RangeReduceExpression.h"

#include "BinaryOperator.h"
#include <boost/make_shared.hpp>
#include <boost/tuple/tuple.hpp>
#include "Conditional.h"
#include "LiteralExpression.h"
#include "FreeForm2Assert.h"
#include "OperatorExpression.h"
#include "RefExpression.h"
#include "SimpleExpressionOwner.h"
#include "Visitor.h"
#include "TypeUtil.h"
#include "UnaryOperator.h"
#include <sstream>

namespace
{
    // This method creates the following precondition expression:
    //    (max(low, high) - abs(step) < max(low, high)) 
    // && (min(low, high) + abs(step) > min(low, high))
    // && ((step > 0) == (high > low))
    // && step != 0)
    // and the following loop condition:
    // (step > 0 ? loopVar <= high - step
    //           : loopVar >= high - step).

    boost::tuple<const FreeForm2::Expression*, const FreeForm2::Expression*>
    CreateGenericLoopConditions(
        const std::pair<const FreeForm2::Expression*, const FreeForm2::Expression*>& p_range,
        const FreeForm2::Expression& p_step,
        const FreeForm2::Expression& p_loopVar,
        FreeForm2::SimpleExpressionOwner& p_owner,
        FreeForm2::TypeManager& p_typeManager)
    {
        using namespace FreeForm2;

        // Common expressions.
        auto zero = boost::make_shared<LiteralInt32Expression>(p_loopVar.GetAnnotations(), 0);
        p_owner.AddExpression(zero);
        auto stepSign 
            = BinaryOperatorExpression::Alloc(p_loopVar.GetAnnotations(), p_step, *zero, BinaryOperator::gt, p_typeManager);
        p_owner.AddExpression(stepSign);

        // Create the precondition
        const Expression* precondition = nullptr;
        {
            auto high 
                = BinaryOperatorExpression::Alloc(p_loopVar.GetAnnotations(), *p_range.first, *p_range.second, BinaryOperator::max, p_typeManager);
            p_owner.AddExpression(high);
            auto low 
                = BinaryOperatorExpression::Alloc(p_loopVar.GetAnnotations(), *p_range.first, *p_range.second, BinaryOperator::min, p_typeManager);
            p_owner.AddExpression(low);
            auto step 
                = boost::make_shared<UnaryOperatorExpression>(p_loopVar.GetAnnotations(), p_step, UnaryOperator::abs);
            p_owner.AddExpression(step);
            auto highMinusStep 
                = BinaryOperatorExpression::Alloc(p_loopVar.GetAnnotations(), *high, *step, BinaryOperator::minus, p_typeManager);
            p_owner.AddExpression(highMinusStep);
            auto lowPlusStep 
                = BinaryOperatorExpression::Alloc(p_loopVar.GetAnnotations(), *low, *step, BinaryOperator::plus, p_typeManager);
            p_owner.AddExpression(lowPlusStep);
            auto underflowCheck 
                = BinaryOperatorExpression::Alloc(p_loopVar.GetAnnotations(), *highMinusStep, *high, BinaryOperator::lt, p_typeManager);
            p_owner.AddExpression(underflowCheck);
            auto overflowCheck
                = BinaryOperatorExpression::Alloc(p_loopVar.GetAnnotations(), *lowPlusStep, *low, BinaryOperator::gt, p_typeManager);
            p_owner.AddExpression(overflowCheck);
            auto stepMoving 
                = BinaryOperatorExpression::Alloc(p_loopVar.GetAnnotations(), p_step, *zero, BinaryOperator::neq, p_typeManager);
            p_owner.AddExpression(stepMoving);
            auto rangeSign 
                = BinaryOperatorExpression::Alloc(p_loopVar.GetAnnotations(), *p_range.second, *p_range.first, BinaryOperator::gt, p_typeManager);
            p_owner.AddExpression(rangeSign);
            auto rangeCheck 
                = BinaryOperatorExpression::Alloc(p_loopVar.GetAnnotations(), *stepSign, *rangeSign, BinaryOperator::eq, p_typeManager);
            p_owner.AddExpression(rangeCheck);
            auto and1 
                = BinaryOperatorExpression::Alloc(p_loopVar.GetAnnotations(), *underflowCheck, *overflowCheck, BinaryOperator::_and, p_typeManager);
            p_owner.AddExpression(and1);
            auto and2 
                = BinaryOperatorExpression::Alloc(p_loopVar.GetAnnotations(), *and1, *rangeCheck, BinaryOperator::_and, p_typeManager);
            p_owner.AddExpression(and2);
            auto and3 
                = BinaryOperatorExpression::Alloc(p_loopVar.GetAnnotations(), *and2, *stepMoving, BinaryOperator::_and, p_typeManager);
            p_owner.AddExpression(and3);
            precondition = and3.get();
        }

        // Create the loop condition.
        const Expression* condition = nullptr;
        {
            auto endMinusStep 
                = BinaryOperatorExpression::Alloc(p_loopVar.GetAnnotations(), *p_range.second, p_step, BinaryOperator::minus, p_typeManager);
            p_owner.AddExpression(endMinusStep);
            auto incRange 
                = BinaryOperatorExpression::Alloc(p_loopVar.GetAnnotations(), p_loopVar, *endMinusStep, BinaryOperator::lte, p_typeManager);
            p_owner.AddExpression(incRange);
            auto decRange 
                = BinaryOperatorExpression::Alloc(p_loopVar.GetAnnotations(), p_loopVar, *endMinusStep, BinaryOperator::gte, p_typeManager);
            p_owner.AddExpression(decRange);
            auto cond = boost::make_shared<ConditionalExpression>(p_loopVar.GetAnnotations(), *stepSign, *incRange, *decRange);
            p_owner.AddExpression(cond);
            condition = cond.get();
        }

        return boost::make_tuple(precondition, condition);
    }


    boost::tuple<const FreeForm2::Expression*, const FreeForm2::Expression*>
    CreateConditionsForKnownStep(
        const std::pair<const FreeForm2::Expression*, const FreeForm2::Expression*>& p_range,
        const FreeForm2::Expression& p_step,
        const FreeForm2::Expression& p_loopVar,
        FreeForm2::SimpleExpressionOwner& p_owner,
        FreeForm2::TypeManager& p_typeManager)
    {
        using namespace FreeForm2;
        FF2_ASSERT(p_step.IsConstant() && p_step.GetConstantValue().GetInt(p_step.GetType()) != 0);
        const Result::IntType stepVal = p_step.GetConstantValue().GetInt(p_step.GetType());
        const bool isIncreasing = stepVal > 0;

        // For increasing ranges, create the expression:
        // ((high - step < high) && (low + step > low) && high > low)
        // For decreasing ranges, create the expression:
        // ((high - step > high) && (low + step < low) && high < low)
        const Expression* precondition = nullptr;
        {
            const Expression& low = *p_range.first;
            const Expression& high = *p_range.second;
            auto highMinusStep = BinaryOperatorExpression::Alloc(p_loopVar.GetAnnotations(), high, p_step, BinaryOperator::minus, p_typeManager);
            p_owner.AddExpression(highMinusStep);
            auto lowPlusStep = BinaryOperatorExpression::Alloc(p_loopVar.GetAnnotations(), low, p_step, BinaryOperator::plus, p_typeManager);
            p_owner.AddExpression(lowPlusStep);
            const BinaryOperator::Operation check1Op = isIncreasing ? BinaryOperator::lt : BinaryOperator::gt;
            auto check1 = BinaryOperatorExpression::Alloc(p_loopVar.GetAnnotations(), *highMinusStep, high, check1Op, p_typeManager);
            p_owner.AddExpression(check1);
            const BinaryOperator::Operation check2Op = isIncreasing ? BinaryOperator::gt : BinaryOperator::lt;
            auto check2 = BinaryOperatorExpression::Alloc(p_loopVar.GetAnnotations(), *lowPlusStep, low, check2Op, p_typeManager);
            p_owner.AddExpression(check2);
            const BinaryOperator::Operation rangeOp = isIncreasing ? BinaryOperator::gt : BinaryOperator::lt;
            auto rangeCheck = BinaryOperatorExpression::Alloc(p_loopVar.GetAnnotations(), high, low, rangeOp, p_typeManager);
            p_owner.AddExpression(rangeCheck);
            auto and1 = BinaryOperatorExpression::Alloc(p_loopVar.GetAnnotations(), *check1, *check2, BinaryOperator::_and, p_typeManager);
            p_owner.AddExpression(and1);
            auto and2 = BinaryOperatorExpression::Alloc(p_loopVar.GetAnnotations(), *and1, *rangeCheck, BinaryOperator::_and, p_typeManager);
            p_owner.AddExpression(and2);
            precondition = and2.get();
        }

        // For increasing ranges, create the expression: loopVar <= high - step.
        // For decreasing ranges, create the expression: loopVar >= high - step.
        const Expression* condition = nullptr;
        {
            const Expression& high = *p_range.second;
            auto highMinusStep 
                = BinaryOperatorExpression::Alloc(p_loopVar.GetAnnotations(), high, p_step, BinaryOperator::minus, p_typeManager);
            p_owner.AddExpression(highMinusStep);
            const BinaryOperator::Operation compareOp = isIncreasing ? BinaryOperator::lte : BinaryOperator::gte;
            auto compare = BinaryOperatorExpression::Alloc(p_loopVar.GetAnnotations(), p_loopVar, *highMinusStep, compareOp, p_typeManager);
            p_owner.AddExpression(compare);
            condition = compare.get();
        }

        return boost::make_tuple(precondition, condition);
    }
}


FreeForm2::RangeReduceExpression::RangeReduceExpression(
    const Annotations& p_annotations,
    const Expression& p_low,
    const Expression& p_high,
    const Expression& p_initial,
    const Expression& p_reduce,
    VariableID p_stepId,
    VariableID p_reduceId)
    : Expression(p_annotations),
      m_low(p_low),
      m_high(p_high),
      m_initial(p_initial),
      m_reduce(p_reduce),
      m_stepId(p_stepId),
      m_reduceId(p_reduceId),
      m_type(InferType())
{
}

const FreeForm2::TypeImpl&
FreeForm2::RangeReduceExpression::InferType() const
{
    if (!m_low.GetType().IsIntegerType() || !m_high.GetType().IsIntegerType())
    {
        std::ostringstream err;
        err << "Expected low range and high range arguments to be of compatible integer types;"
            << " got " << m_low.GetType() << ", " << m_high.GetType()
            << " respectively.";
        throw ParseError(err.str(), GetSourceLocation());
    }

    if (!(m_initial.GetType().IsSameAs(m_reduce.GetType(), true)))
    {
        std::ostringstream err;
        err << "Expected initial reduction argument to range-reduce to "
                "be of the same type as the reduction expression.  Got "
            << m_initial.GetType() << " and "
            << m_reduce.GetType() << " respectively.";
        throw ParseError(err.str(), GetSourceLocation());
    }

    if (m_reduce.GetType().Primitive() == Type::Array)
    {
        std::ostringstream err;
        err << "An array cannot be the result of a looping expression, "
                "such as range-reduce, as our array representation "
                "relies on reusing array space (and thus uses constant "
                "space).  If arrays were the result of loops using this "
                "representation, dangling pointers would result.";
        throw ParseError(err.str(), GetSourceLocation());
    }
    return m_reduce.GetType().AsConstType();
}


size_t
FreeForm2::RangeReduceExpression::GetNumChildren() const
{
    return 4;
}


void
FreeForm2::RangeReduceExpression::Accept(Visitor& p_visitor) const
{
    size_t stackSize = p_visitor.StackSize();

    if (!p_visitor.AlternativeVisit(*this))
    {
        m_initial.Accept(p_visitor);
        m_high.Accept(p_visitor);
        m_low.Accept(p_visitor);
        m_reduce.Accept(p_visitor);

        p_visitor.Visit(*this);
    }

    FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}


const FreeForm2::TypeImpl&
FreeForm2::RangeReduceExpression::GetType() const
{
    return m_type;
}


const FreeForm2::Expression&
FreeForm2::RangeReduceExpression::GetLow() const
{
    return m_low;
}


const FreeForm2::Expression&
FreeForm2::RangeReduceExpression::GetHigh() const
{
    return m_high;
}


const FreeForm2::Expression&
FreeForm2::RangeReduceExpression::GetInitial() const
{
    return m_initial;
}


const FreeForm2::Expression&
FreeForm2::RangeReduceExpression::GetReduceExpression() const
{
    return m_reduce;
}


FreeForm2::VariableID
FreeForm2::RangeReduceExpression::GetReduceId() const
{
    return m_reduceId;
}


FreeForm2::VariableID
FreeForm2::RangeReduceExpression::GetStepId() const
{
    return m_stepId;
}


FreeForm2::ForEachLoopExpression::ForEachLoopExpression(
    const Annotations& p_annotations,
    const std::pair<const Expression*, const Expression*>& p_bounds,
    const Expression& p_next,
    const Expression& p_body,
    VariableID p_iteratorId,
    size_t p_version,
    LoopHint p_hint,
    TypeManager& p_typeManager)
    : Expression(p_annotations),
      m_begin(*p_bounds.first),
      m_end(*p_bounds.second),
      m_next(p_next),
      m_body(p_body),
      m_iteratorType(nullptr),
      m_iteratorId(p_iteratorId),
      m_version(p_version),
      m_hint(p_hint)
{
    FF2_ASSERT(p_bounds.first && p_bounds.second);
    m_iteratorType = &TypeUtil::Unify(m_begin.GetType(), m_end.GetType(), p_typeManager, false, true);
    m_iteratorType = &TypeUtil::Unify(m_next.GetType(), *m_iteratorType, p_typeManager, false, true);

    if (!m_iteratorType->IsValid())
    {
        std::ostringstream err;
        err << "For-each bounds must have unifiable types. Got "
            << m_begin.GetType() << ", " << m_end.GetType()
            << ", and " << m_next.GetType()
            << " for beginning, ending, and step values respectively.";
        throw ParseError(err.str(), GetSourceLocation());
    }
}

size_t
FreeForm2::ForEachLoopExpression::GetNumChildren() const
{
    return 4;
}


void
FreeForm2::ForEachLoopExpression::Accept(Visitor& p_visitor) const
{
    size_t stackSize = p_visitor.StackSize();

    if (!p_visitor.AlternativeVisit(*this))
    {
        m_begin.Accept(p_visitor);
        m_end.Accept(p_visitor);
        m_next.Accept(p_visitor);
        m_body.Accept(p_visitor);

        p_visitor.Visit(*this);
    }

    FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}


const FreeForm2::TypeImpl&
FreeForm2::ForEachLoopExpression::GetType() const
{
    return TypeImpl::GetVoidInstance();
}


const FreeForm2::Expression&
FreeForm2::ForEachLoopExpression::GetBegin() const
{
    return m_begin;
}


const FreeForm2::Expression&
FreeForm2::ForEachLoopExpression::GetEnd() const
{
    return m_end;
}


const FreeForm2::Expression&
FreeForm2::ForEachLoopExpression::GetNext() const
{
    return m_next;
}


const FreeForm2::Expression&
FreeForm2::ForEachLoopExpression::GetBody() const
{
    return m_body;
}


const FreeForm2::TypeImpl&
FreeForm2::ForEachLoopExpression::GetIteratorType() const
{
    return *m_iteratorType;
}


FreeForm2::VariableID
FreeForm2::ForEachLoopExpression::GetIteratorId() const
{
    return m_iteratorId;
}


size_t
FreeForm2::ForEachLoopExpression::GetVersion() const
{
    return m_version;
}


FreeForm2::ForEachLoopExpression::LoopHint
FreeForm2::ForEachLoopExpression::GetHint() const
{
    return m_hint;
}


FreeForm2::ComplexRangeLoopExpression::ComplexRangeLoopExpression(
    const Annotations& p_annotations,
    const std::pair<const Expression*, const Expression*>& p_range,
    const Expression& p_step,
    const Expression& p_body,
    const Expression& p_precondition,
    const Expression& p_loopCondition,
    const TypeImpl& p_stepType,
    VariableID p_stepId,
    size_t p_version)
    : Expression(p_annotations),
      m_low(*p_range.first),
      m_high(*p_range.second),
      m_step(p_step),
      m_body(p_body),
      m_precondition(p_precondition),
      m_loopCondition(p_loopCondition),
      m_stepType(p_stepType),
      m_stepId(p_stepId),
      m_version(p_version)
{
    FF2_ASSERT(p_range.first && p_range.second);
    FF2_ASSERT(p_stepId != VariableID::c_invalidID);
    if (!p_range.first->GetType().IsIntegerType() || !p_range.second->GetType().IsIntegerType()
        || !p_step.GetType().IsIntegerType())
    {
        std::ostringstream err;
        err << "Range bounds and step value must all be integral types. Got "
            << p_range.first->GetType() << ", " << p_range.second->GetType() 
            << ", and " << p_step.GetType() << " for low, high, and step respectively.";
        throw ParseError(err.str(), GetSourceLocation());
    }

    if (p_precondition.GetType().Primitive() != Type::Bool
        || p_loopCondition.GetType().Primitive() != Type::Bool)
    {
        std::ostringstream err;
        err << "Loop conditions must evaluate to boolean types. Got "
            << p_precondition.GetType() << " and " << p_loopCondition.GetType()
            << " for the precondition and loop condition respectively.";
        throw ParseError(err.str(), GetSourceLocation());
    }
}


const FreeForm2::ComplexRangeLoopExpression&    
FreeForm2::ComplexRangeLoopExpression::Create(
    const Annotations& p_annotations,
    const std::pair<const Expression*, const Expression*>& p_range,
    const Expression& p_step,
    const Expression& p_body,
    const Expression& p_loopVar,
    VariableID p_stepId,
    size_t p_version,
    SimpleExpressionOwner& p_owner,
    TypeManager& p_typeManager)
{
    FF2_ASSERT(p_range.first && p_range.second);
    const TypeImpl* stepType 
        = &TypeUtil::Unify(p_range.first->GetType(), p_range.second->GetType(), p_typeManager, false, true);
    stepType = &TypeUtil::Unify(*stepType, p_step.GetType(), p_typeManager, false, true);

    if (!p_range.first->GetType().IsIntegerType() || !p_range.second->GetType().IsIntegerType()
        || !p_step.GetType().IsIntegerType() || !p_loopVar.GetType().IsIntegerType()
        || !TypeUtil::IsAssignable(p_loopVar.GetType(), *stepType))
    {
        std::ostringstream err;
        err << "Range bounds, step value, and loop variable must all be integral types. Got "
            << p_range.first->GetType() << ", " << p_range.second->GetType() 
            << ", " << p_step.GetType() << ", and " << p_loopVar.GetType()
            << " for low, high, and step respectively.";
        throw ParseError(err.str(), p_annotations.m_sourceLocation);
    }

    const Expression* precondition = nullptr;
    const Expression* condition = nullptr;
    if (p_step.IsConstant())
    {
        boost::tie(precondition, condition)
            = CreateConditionsForKnownStep(p_range, p_step, p_loopVar, p_owner, p_typeManager);
    }
    else
    {
        boost::tie(precondition, condition)
            = CreateGenericLoopConditions(p_range, p_step, p_loopVar, p_owner, p_typeManager);
    }

    FF2_ASSERT(precondition && condition);
    auto loop = boost::make_shared<ComplexRangeLoopExpression>(
        p_annotations, p_range, p_step, p_body, *precondition, *condition, p_loopVar.GetType(), p_stepId, p_version);
    p_owner.AddExpression(loop);
    return *loop;
}


size_t
FreeForm2::ComplexRangeLoopExpression::GetNumChildren() const
{
    return 6;
}


void
FreeForm2::ComplexRangeLoopExpression::Accept(Visitor& p_visitor) const
{
    size_t stackSize = p_visitor.StackSize();

    if (!p_visitor.AlternativeVisit(*this))
    {
        m_precondition.Accept(p_visitor);
        m_low.Accept(p_visitor);
        m_high.Accept(p_visitor);
        m_step.Accept(p_visitor);
        m_body.Accept(p_visitor);
        m_loopCondition.Accept(p_visitor);

        p_visitor.Visit(*this);
    }

    FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}


const FreeForm2::TypeImpl&
FreeForm2::ComplexRangeLoopExpression::GetType() const
{
    return TypeImpl::GetVoidInstance();
}


const FreeForm2::Expression&
FreeForm2::ComplexRangeLoopExpression::GetPrecondition() const
{
    return m_precondition;
}


const FreeForm2::Expression&
FreeForm2::ComplexRangeLoopExpression::GetLow() const
{
    return m_low;
}


const FreeForm2::Expression&
FreeForm2::ComplexRangeLoopExpression::GetHigh() const
{
    return m_high;
}


const FreeForm2::Expression&
FreeForm2::ComplexRangeLoopExpression::GetStep() const
{
    return m_step;
}


const FreeForm2::Expression&
FreeForm2::ComplexRangeLoopExpression::GetBody() const
{
    return m_body;
}


const FreeForm2::Expression&
FreeForm2::ComplexRangeLoopExpression::GetLoopCondition() const
{
    return m_loopCondition;
}


const FreeForm2::TypeImpl&
FreeForm2::ComplexRangeLoopExpression::GetStepType() const
{
    return m_stepType;
}


FreeForm2::VariableID
FreeForm2::ComplexRangeLoopExpression::GetStepId() const
{
    return m_stepId;
}


size_t
FreeForm2::ComplexRangeLoopExpression::GetVersion() const
{
    return m_version;
}

