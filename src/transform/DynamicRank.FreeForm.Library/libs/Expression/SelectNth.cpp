#include "SelectNth.h"

#include "ArrayType.h"
#include "Expression.h"
#include "FreeForm2Assert.h"
#include "SimpleExpressionOwner.h"
#include "Visitor.h"
#include "TypeManager.h"
#include "TypeUtil.h"
#include <sstream>

FreeForm2::SelectNthExpression::SelectNthExpression(
    const Annotations& p_annotations,
    const std::vector<const Expression*>& p_children)
    : Expression(p_annotations),
      m_index(*p_children[0]),
      m_numChildren(static_cast<unsigned int>(p_children.size()) - 1),
      m_type(NULL)
{
    FF2_ASSERT(p_children.size() >= 2);
    for (unsigned int i = 0; i + 1 < p_children.size(); i++)
    {
        m_children[i] = p_children[i + 1];
    }

    m_type = &InferType();
}


void
FreeForm2::SelectNthExpression::Accept(Visitor& p_visitor) const
{
    size_t stackSize = p_visitor.StackSize();

    if (!p_visitor.AlternativeVisit(*this))
    {
        for (unsigned int i = 0; i < m_numChildren; i++)
        {        
            m_children[i]->Accept(p_visitor);
        }

        m_index.Accept(p_visitor);

        p_visitor.Visit(*this);
    }

    FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}


const FreeForm2::TypeImpl&
FreeForm2::SelectNthExpression::GetType() const
{
    return *m_type;
}


const FreeForm2::TypeImpl&
FreeForm2::SelectNthExpression::InferType() const
{
    FF2_ASSERT(m_numChildren > 0);

    if (!m_index.GetType().IsIntegerType())
    {
        std::ostringstream err;
        err << "First argument to selection expression must be "
            << " an integer (got type '"
            << m_index.GetType() << "')";
        throw ParseError(err.str(), GetSourceLocation());
    }

    const TypeImpl& unifiedType = m_children[0]->GetType().AsConstType();
    for (unsigned int i = 1; i < m_numChildren; i++)
    {
        if (!unifiedType.IsSameAs(m_children[i]->GetType(), true))
        {
            std::ostringstream err;
            err << "All arguments to selection expression (except index) "
                << "are expected to be of the same type.  The first "
                << i - 1 << " elements are of type '" << unifiedType
                << "', element " << i << " is of type '" 
                << m_children[i]->GetType() << "'";
            throw ParseError(err.str(), GetSourceLocation());
        }
    }

    return unifiedType;
}


size_t
FreeForm2::SelectNthExpression::GetNumChildren() const
{
    return m_numChildren + 1;
}


const FreeForm2::Expression& 
FreeForm2::SelectNthExpression::GetIndex() const
{
    return m_index;
}


const FreeForm2::Expression& 
FreeForm2::SelectNthExpression::GetChild(size_t p_index) const
{
    FF2_ASSERT(p_index < m_numChildren);
    return *m_children[p_index];
}


boost::shared_ptr<FreeForm2::SelectNthExpression> 
FreeForm2::SelectNthExpression::Alloc(const Annotations& p_annotations,
                                      const std::vector<const Expression*>& p_children)
{
    size_t bytes = sizeof(SelectNthExpression) 
        + (p_children.size() - 2) * sizeof(Expression*);

    // Allocate a shared_ptr that deletes an SelectNthExpression 
    // allocated in a char[].
    boost::shared_ptr<SelectNthExpression> exp(new (new char[bytes]) 
        SelectNthExpression(p_annotations, p_children), DeleteAlloc);
    return exp;
}


void
FreeForm2::SelectNthExpression::DeleteAlloc(SelectNthExpression* p_allocated)
{
    // Manually call dtor for arith expression.
    p_allocated->~SelectNthExpression();

    // Dispose of memory, which we allocated in a char[].
    char* mem = reinterpret_cast<char*>(p_allocated);
    delete[] mem;
}


FreeForm2::SelectRangeExpression::SelectRangeExpression(const Annotations& p_annotations,
                                                        const Expression& p_start,
                                                        const Expression& p_count,
                                                        const Expression& p_array,
                                                        TypeManager& p_typeManager)
    : Expression(p_annotations),
      m_type(nullptr),
      m_start(p_start),
      m_count(p_count),
      m_array(p_array)
{
    FF2_ASSERT(m_array.GetType().Primitive() == Type::Array);
    const ArrayType& type = static_cast<const ArrayType&>(m_array.GetType());
    m_type = &p_typeManager.GetArrayType(type.GetChildType(), 
                                         true, 
                                         type.GetDimensionCount(), 
                                         type.GetMaxElements());
}


void
FreeForm2::SelectRangeExpression::Accept(Visitor& p_visitor) const
{
    const size_t startSize = p_visitor.StackSize();

    if (!p_visitor.AlternativeVisit(*this))
    {
        m_start.Accept(p_visitor);
        m_count.Accept(p_visitor);
        m_array.Accept(p_visitor);

        p_visitor.Visit(*this);
    }

    FF2_ASSERT(p_visitor.StackSize() == startSize + p_visitor.StackIncrement());
}


size_t
FreeForm2::SelectRangeExpression::GetNumChildren() const
{
    return 3;
}


const FreeForm2::TypeImpl& 
FreeForm2::SelectRangeExpression::GetType() const
{
    return *m_type;
}


const FreeForm2::Expression&
FreeForm2::SelectRangeExpression::GetStart() const
{
    return m_start;
}


const FreeForm2::Expression&
FreeForm2::SelectRangeExpression::GetCount() const
{
    return m_count;
}


const FreeForm2::Expression&
FreeForm2::SelectRangeExpression::GetArray() const
{
    return m_array;
}
