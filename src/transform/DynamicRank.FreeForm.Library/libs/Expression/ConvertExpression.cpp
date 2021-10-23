#include "ConvertExpression.h"

#include "FreeForm2Assert.h"
#include "SimpleExpressionOwner.h"
#include "Visitor.h"
#include "TypeUtil.h"
#include <sstream>
#include <stdexcept>

static
void
CheckType(const FreeForm2::TypeImpl& p_type, 
          const FreeForm2::TypeImpl& p_child,
          const FreeForm2::SourceLocation& p_sourceLocation)
{
    if (!FreeForm2::TypeUtil::IsConvertible(p_child, p_type) 
        && p_child.Primitive() != FreeForm2::Type::Unknown)
    {
        std::ostringstream err;
        err << "Expression type " << p_child 
            << " cannot be converted to type " << p_type;
        throw FreeForm2::ParseError(err.str(), p_sourceLocation);
    }    
}


FreeForm2::ConversionExpression::ConversionExpression(const Annotations& p_annotations,
                                                      const Expression& p_child)
    : Expression(p_annotations),
      m_child(p_child)
{
}


FreeForm2::ConversionExpression::~ConversionExpression()
{
}


size_t
FreeForm2::ConversionExpression::GetNumChildren() const
{
    return 1;
}


const FreeForm2::TypeImpl&
FreeForm2::ConversionExpression::GetChildType() const
{
    return m_child.GetType();
}


const FreeForm2::Expression&
FreeForm2::ConversionExpression::GetChild() const
{
    return m_child;
}


template <typename Derived>
void FreeForm2::ConversionExpression::AcceptDerived(Visitor& p_visitor) const
{
    size_t stackSize = p_visitor.StackSize();

    if (!p_visitor.AlternativeVisit(static_cast<const Derived&>(*this)))
    {
        m_child.Accept(p_visitor);

        p_visitor.Visit(static_cast<const Derived&>(*this));
    }

    FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}


FreeForm2::ConvertToFloatExpression::ConvertToFloatExpression(const Annotations& p_annotations,
                                                              const Expression& p_child)
    : ConversionExpression(p_annotations, p_child)
{
}


const FreeForm2::TypeImpl&
FreeForm2::ConvertToFloatExpression::GetType() const
{
    const TypeImpl& type = TypeImpl::GetFloatInstance(true);
    CheckType(type, GetChildType(), GetSourceLocation());

    return type;
}


void
FreeForm2::ConvertToFloatExpression::Accept(Visitor& p_visitor) const
{
    AcceptDerived<ConvertToFloatExpression>(p_visitor);
}


FreeForm2::ConvertToIntExpression::ConvertToIntExpression(const Annotations& p_annotations,
                                                          const Expression& p_child)
    : ConversionExpression(p_annotations, p_child)
{
}


const FreeForm2::TypeImpl&
FreeForm2::ConvertToIntExpression::GetType() const
{
    const TypeImpl& type = TypeImpl::GetIntInstance(true);
    CheckType(type, GetChildType(), GetSourceLocation());

    return type;
}


void
FreeForm2::ConvertToIntExpression::Accept(Visitor& p_visitor) const
{
    AcceptDerived<ConvertToIntExpression>(p_visitor);
}


FreeForm2::ConvertToUInt64Expression::ConvertToUInt64Expression(const Annotations& p_annotations,
                                                                const Expression& p_child)
    : ConversionExpression(p_annotations, p_child)
{
}


const FreeForm2::TypeImpl&
FreeForm2::ConvertToUInt64Expression::GetType() const
{
    const TypeImpl& type = TypeImpl::GetUInt64Instance(true);
    CheckType(type, GetChildType(), GetSourceLocation());

    return type;
}


void
FreeForm2::ConvertToUInt64Expression::Accept(Visitor& p_visitor) const
{
    AcceptDerived<ConvertToUInt64Expression>(p_visitor);
}


FreeForm2::ConvertToInt32Expression::ConvertToInt32Expression(const Annotations& p_annotations,
                                                              const Expression& p_child)
    : ConversionExpression(p_annotations, p_child)
{
}


const FreeForm2::TypeImpl&
FreeForm2::ConvertToInt32Expression::GetType() const
{
    const TypeImpl& type = TypeImpl::GetInt32Instance(true);
    CheckType(type, GetChildType(), GetSourceLocation());

    return type;
}


void
FreeForm2::ConvertToInt32Expression::Accept(Visitor& p_visitor) const
{
    AcceptDerived<ConvertToInt32Expression>(p_visitor);
}


FreeForm2::ConvertToUInt32Expression::ConvertToUInt32Expression(const Annotations& p_annotations,
                                                                const Expression& p_child)
    : ConversionExpression(p_annotations, p_child)
{
}


const FreeForm2::TypeImpl&
FreeForm2::ConvertToUInt32Expression::GetType() const
{
    const TypeImpl& type = TypeImpl::GetUInt32Instance(true);
    CheckType(type, GetChildType(), GetSourceLocation());

    return type;
}


void
FreeForm2::ConvertToUInt32Expression::Accept(Visitor& p_visitor) const
{
    AcceptDerived<ConvertToUInt32Expression>(p_visitor);
}


FreeForm2::ConvertToBoolExpression::ConvertToBoolExpression(const Annotations& p_annotations,
                                                            const Expression& p_child)
    : ConversionExpression(p_annotations, p_child)
{
}


const FreeForm2::TypeImpl&
FreeForm2::ConvertToBoolExpression::GetType() const
{
    const TypeImpl& type = TypeImpl::GetBoolInstance(true);
    CheckType(type, GetChildType(), GetSourceLocation());

    return type;
}


void
FreeForm2::ConvertToBoolExpression::Accept(Visitor& p_visitor) const
{
    AcceptDerived<ConvertToBoolExpression>(p_visitor);
}


FreeForm2::ConvertToImperativeExpression::ConvertToImperativeExpression(const Annotations& p_annotations,
                                                                        const Expression& p_child)
    : ConversionExpression(p_annotations, p_child)
{
}


const FreeForm2::TypeImpl&
FreeForm2::ConvertToImperativeExpression::GetType() const
{
    return TypeImpl::GetVoidInstance();
}


void 
FreeForm2::ConvertToImperativeExpression::Accept(Visitor& p_visitor) const
{
    AcceptDerived<ConvertToImperativeExpression>(p_visitor);
}
