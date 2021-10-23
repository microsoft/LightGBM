#include "LiteralExpression.h"

#include "Allocation.h"
#include "FreeForm2Assert.h"
#include "Visitor.h"


FreeForm2::LeafTypeLiteralExpression::LeafTypeLiteralExpression(const Annotations& p_annotations,
                                                                Result::IntType p_value)
    : Expression(Annotations(p_annotations.m_sourceLocation, ValueBounds(p_value, p_value)))
{
    m_value.m_int = p_value;
}


FreeForm2::LeafTypeLiteralExpression::LeafTypeLiteralExpression(const Annotations& p_annotations,
                                                                Result::UInt64Type p_value)
    : Expression(Annotations(p_annotations.m_sourceLocation, ValueBounds(p_value, p_value)))
{
    m_value.m_uint64 = p_value;
}


FreeForm2::LeafTypeLiteralExpression::LeafTypeLiteralExpression(const Annotations& p_annotations,
                                                                Result::Int32Type p_value)
    : Expression(Annotations(p_annotations.m_sourceLocation, ValueBounds(p_value, p_value)))
{
    m_value.m_int32 = p_value;
}


FreeForm2::LeafTypeLiteralExpression::LeafTypeLiteralExpression(const Annotations& p_annotations,
                                                                Result::UInt32Type p_value)
    : Expression(Annotations(p_annotations.m_sourceLocation, ValueBounds(p_value, p_value)))
{
    m_value.m_uint32 = p_value;
}


FreeForm2::LeafTypeLiteralExpression::LeafTypeLiteralExpression(const Annotations& p_annotations,
                                                                Result::FloatType p_value)
    : Expression(p_annotations)
{
    m_value.m_float = p_value;
}


FreeForm2::LeafTypeLiteralExpression::LeafTypeLiteralExpression(const Annotations& p_annotations,
                                                                Result::BoolType p_value)
    : Expression(Annotations(p_annotations.m_sourceLocation, ValueBounds(p_value, p_value)))
{
    m_value.m_bool = p_value;
}


bool
FreeForm2::LeafTypeLiteralExpression::IsConstant() const
{
    return true;
}


FreeForm2::ConstantValue
FreeForm2::LeafTypeLiteralExpression::GetConstantValue() const
{
    return m_value;
}


FreeForm2::LiteralIntExpression::LiteralIntExpression(const Annotations& p_annotations,
                                                      Result::IntType p_val)
    : LeafTypeLiteralExpression(p_annotations, p_val)
{
}


const FreeForm2::TypeImpl&
FreeForm2::LiteralIntExpression::GetType() const
{
    return TypeImpl::GetIntInstance(true);
}


size_t
FreeForm2::LiteralIntExpression::GetNumChildren() const
{
    return 0;
}


void
FreeForm2::LiteralIntExpression::Accept(Visitor& p_visitor) const
{
    const size_t stackSize = p_visitor.StackSize();

    if (!p_visitor.AlternativeVisit(*this))
    {
       p_visitor.Visit(*this);
    }

    FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}


FreeForm2::LiteralUInt64Expression::LiteralUInt64Expression(const Annotations& p_annotations,
                                                            Result::UInt64Type p_val)
    : LeafTypeLiteralExpression(p_annotations, p_val)
{
}


const FreeForm2::TypeImpl&
FreeForm2::LiteralUInt64Expression::GetType() const
{
    return TypeImpl::GetUInt64Instance(true);
}


size_t
FreeForm2::LiteralUInt64Expression::GetNumChildren() const
{
    return 0;
}


void
FreeForm2::LiteralUInt64Expression::Accept(Visitor& p_visitor) const
{
    const size_t stackSize = p_visitor.StackSize();

    if (!p_visitor.AlternativeVisit(*this))
    {
       p_visitor.Visit(*this);
    }

    FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}


FreeForm2::LiteralInt32Expression::LiteralInt32Expression(const Annotations& p_annotations,
                                                          int p_val)
    : LeafTypeLiteralExpression(p_annotations, p_val)
{
}


const FreeForm2::TypeImpl&
FreeForm2::LiteralInt32Expression::GetType() const
{
    return TypeImpl::GetInt32Instance(true);
}


size_t
FreeForm2::LiteralInt32Expression::GetNumChildren() const
{
    return 0;
}


void
FreeForm2::LiteralInt32Expression::Accept(Visitor& p_visitor) const
{
    const size_t stackSize = p_visitor.StackSize();

    if (!p_visitor.AlternativeVisit(*this))
    {
       p_visitor.Visit(*this);
    }

    FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}


FreeForm2::LiteralUInt32Expression::LiteralUInt32Expression(const Annotations& p_annotations,
                                                            unsigned int p_val)
    : LeafTypeLiteralExpression(p_annotations, p_val)
{
}


const FreeForm2::TypeImpl&
FreeForm2::LiteralUInt32Expression::GetType() const
{
    return TypeImpl::GetUInt32Instance(true);
}


size_t
FreeForm2::LiteralUInt32Expression::GetNumChildren() const
{
    return 0;
}


void
FreeForm2::LiteralUInt32Expression::Accept(Visitor& p_visitor) const
{
    const size_t stackSize = p_visitor.StackSize();

    if (!p_visitor.AlternativeVisit(*this))
    {
       p_visitor.Visit(*this);
    }

    FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}


FreeForm2::LiteralFloatExpression::LiteralFloatExpression(const Annotations& p_annotations,
                                                          Result::FloatType p_val)
    : LeafTypeLiteralExpression(p_annotations, p_val)
{
}


const FreeForm2::TypeImpl&
FreeForm2::LiteralFloatExpression::GetType() const
{
    return TypeImpl::GetFloatInstance(true);
}


size_t
FreeForm2::LiteralFloatExpression::GetNumChildren() const
{
    return 0;
}


void
FreeForm2::LiteralFloatExpression::Accept(Visitor& p_visitor) const
{
    const size_t stackSize = p_visitor.StackSize();

    if (!p_visitor.AlternativeVisit(*this))
    {
        p_visitor.Visit(*this);
    }

    FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}


FreeForm2::LiteralBoolExpression::LiteralBoolExpression(const Annotations& p_annotations,
                                                        bool p_val)
    : LeafTypeLiteralExpression(p_annotations, p_val)
{
}


const FreeForm2::TypeImpl&
FreeForm2::LiteralBoolExpression::GetType() const
{
    return TypeImpl::GetBoolInstance(true);
}


size_t
FreeForm2::LiteralBoolExpression::GetNumChildren() const
{
    return 0;
}


void 
FreeForm2::LiteralBoolExpression::Accept(Visitor& p_visitor) const
{
    const size_t stackSize = p_visitor.StackSize();

    if (!p_visitor.AlternativeVisit(*this))
    {
        p_visitor.Visit(*this);
    }

    FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}


const FreeForm2::LiteralVoidExpression& 
FreeForm2::LiteralVoidExpression::GetInstance()
{
    static const Annotations s_annotations;
    static const LiteralVoidExpression s_expr(s_annotations);
    return s_expr;
}


const FreeForm2::TypeImpl&
FreeForm2::LiteralVoidExpression::GetType() const
{
    return TypeImpl::GetVoidInstance();
}


size_t 
FreeForm2::LiteralVoidExpression::GetNumChildren() const
{
    return 0;
}


void 
FreeForm2::LiteralVoidExpression::Accept(Visitor& p_visitor) const
{
    const size_t stackSize = p_visitor.StackSize();

    if (!p_visitor.AlternativeVisit(*this))
    {
        p_visitor.Visit(*this);
    }

    FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}


FreeForm2::LiteralVoidExpression::LiteralVoidExpression(const Annotations& p_annotations)
    : Expression(p_annotations)
{
}


FreeForm2::LiteralWordExpression::LiteralWordExpression(
    const Annotations& p_annotations,
    const Expression& p_word, 
    const Expression& p_offset,
    const Expression* p_attribute,
    const Expression* p_length, 
    const Expression* p_candidate,
    VariableID p_variableID)
    : Expression(p_annotations),
      m_isHeader(false), 
      m_word(p_word), 
      m_offset(p_offset), 
      m_attribute(p_attribute), 
      m_length(p_length), 
      m_candidate(p_candidate),
      m_variableID(p_variableID)
{
}


FreeForm2::LiteralWordExpression::LiteralWordExpression(
    const Annotations& p_annotations,
    const Expression& p_length, 
    const Expression& p_count,
    VariableID p_variableID)
    : Expression(p_annotations),
      m_isHeader(true), 
      m_word(p_length), 
      m_offset(p_count), 
      m_attribute(NULL), 
      m_length(NULL), 
      m_candidate(NULL),
      m_variableID(p_variableID)
{
}


const FreeForm2::TypeImpl&
FreeForm2::LiteralWordExpression::GetType() const
{
    return TypeImpl::GetWordInstance(true);
}


size_t 
FreeForm2::LiteralWordExpression::GetNumChildren() const
{
    return 2 
        + (m_attribute != NULL ? 1 : 0)
        + (m_length != NULL ? 1 : 0)
        + (m_candidate != NULL ? 1 : 0);
}


FreeForm2::VariableID
FreeForm2::LiteralWordExpression::GetId() const
{
    return m_variableID;
}


void 
FreeForm2::LiteralWordExpression::Accept(Visitor& p_visitor) const
{
    const size_t stackSize = p_visitor.StackSize();

    if (!p_visitor.AlternativeVisit(*this))
    {
        m_word.Accept(p_visitor);
        m_offset.Accept(p_visitor);
        if (m_attribute != NULL)
        {
            m_attribute->Accept(p_visitor);
        }
        if (m_length != NULL)
        {
            m_length->Accept(p_visitor);
        }
        if (m_candidate != NULL)
        {
            m_candidate->Accept(p_visitor);
        }

        p_visitor.Visit(*this);
    }

    FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}


const FreeForm2::TypeImpl&
FreeForm2::LiteralStreamExpression::GetType() const
{
    return TypeImpl::GetStreamInstance(true);
}


size_t 
FreeForm2::LiteralStreamExpression::GetNumChildren() const
{
    return m_numChildren;
}


void 
FreeForm2::LiteralStreamExpression::Accept(Visitor& p_visitor) const
{
    const size_t stackSize = p_visitor.StackSize();

    if (!p_visitor.AlternativeVisit(*this))
    {
        for (unsigned int i = 0; i < m_numChildren; i++)
        {
            m_children[i]->Accept(p_visitor);
        }

        p_visitor.Visit(*this);
    }

    FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}


const FreeForm2::Expression* const* 
FreeForm2::LiteralStreamExpression::GetChildren() const
{
    return m_children;
}


FreeForm2::VariableID
FreeForm2::LiteralStreamExpression::GetId() const
{
    return m_id;
}


FreeForm2::LiteralStreamExpression::LiteralStreamExpression(const Annotations& p_annotations,
                                                            const Expression** p_children, 
                                                            size_t p_numChildren,
                                                            VariableID p_id)
    : Expression(p_annotations),
      m_numChildren(p_numChildren),
      m_id(p_id)
{
    for (unsigned int i = 0; i < m_numChildren; i++)
    {
        m_children[i] = p_children[i];
    }
}


boost::shared_ptr<FreeForm2::LiteralStreamExpression> 
FreeForm2::LiteralStreamExpression::Alloc(const Annotations& p_annotations,
                                          const Expression** p_children, 
                                          size_t p_numChildren,
                                          VariableID p_id)
{
    FF2_ASSERT(p_numChildren > 0);
    size_t bytes = sizeof(LiteralStreamExpression) 
        + (p_numChildren - 1) * sizeof(Expression*);

    // Allocate a shared_ptr that deletes an LiteralStreamExpression 
    // allocated in a char[].
    boost::shared_ptr<LiteralStreamExpression> exp(new (new char[bytes]) 
        LiteralStreamExpression(p_annotations, p_children, p_numChildren, p_id), DeleteAlloc);
    return exp;
}


void 
FreeForm2::LiteralStreamExpression::DeleteAlloc(LiteralStreamExpression* p_allocated)
{
    // Manually call dtor for stream expression.
    p_allocated->~LiteralStreamExpression();

    // Dispose of memory, which we allocated in a char[].
    char* mem = reinterpret_cast<char*>(p_allocated);
    delete[] mem;
}


FreeForm2::LiteralInstanceHeaderExpression::LiteralInstanceHeaderExpression(const Annotations& p_annotations,
                                                                            const Expression& p_instanceCount,
                                                                            const Expression& p_rank,
                                                                            const Expression& p_instanceLength)
    : Expression(p_annotations),
      m_instanceCount(p_instanceCount),
      m_rank(p_rank),
      m_instanceLength(p_instanceLength)
{
}


const FreeForm2::TypeImpl&
FreeForm2::LiteralInstanceHeaderExpression::GetType() const
{
    return TypeImpl::GetInstanceHeaderInstance(true);
}


size_t
FreeForm2::LiteralInstanceHeaderExpression::GetNumChildren() const
{
    return 3;
}


void
FreeForm2::LiteralInstanceHeaderExpression::Accept(Visitor& p_visitor) const
{
    const size_t stackSize = p_visitor.StackSize();

    if (!p_visitor.AlternativeVisit(*this))
    {
        m_instanceCount.Accept(p_visitor);
        m_rank.Accept(p_visitor);
        m_instanceLength.Accept(p_visitor);
        p_visitor.Visit(*this);
    }

    FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}
