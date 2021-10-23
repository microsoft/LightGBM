#include "BlockExpression.h"

#include "FreeForm2Assert.h"
#include "Visitor.h"

boost::shared_ptr<FreeForm2::BlockExpression>
FreeForm2::BlockExpression::Alloc(const Annotations& p_annotations,
                                  const Expression** p_children, 
                                  unsigned int p_numChildren,
                                  unsigned int p_numBound)
{
    FF2_ASSERT(p_numChildren > 0);
    
    size_t bytes = sizeof(BlockExpression) + (p_numChildren - 1) * sizeof(Expression*);

    // Allocate a shared_ptr that deletes an BlockExpression 
    // allocated in a char[].
    boost::shared_ptr<BlockExpression> exp;
    exp.reset(new (new char[bytes]) BlockExpression(p_annotations, p_children, p_numChildren, p_numBound), 
              DeleteAlloc);
    return exp;
}


const FreeForm2::TypeImpl&
FreeForm2::BlockExpression::GetType() const
{
    return *m_returnType;
}


size_t 
FreeForm2::BlockExpression::GetNumChildren() const
{
    return m_numChildren;
}


unsigned int
FreeForm2::BlockExpression::GetNumBound() const
{
    return m_numBound;
}


void 
FreeForm2::BlockExpression::Accept(Visitor& p_visitor) const
{
    size_t stackSize = p_visitor.StackSize();

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


const FreeForm2::Expression&
FreeForm2::BlockExpression::GetChild(unsigned int p_index) const
{
    return *m_children[p_index];
}


FreeForm2::BlockExpression::BlockExpression(const Annotations& p_annotations,
                                            const Expression** p_children,
                                            unsigned int p_numChildren,
                                            unsigned int p_numBound)
    : Expression(p_annotations),
      m_numChildren(p_numChildren),
      m_numBound(p_numBound),
      m_returnType(NULL)
{
    FF2_ASSERT(m_numChildren > 0);
    m_returnType = &p_children[p_numChildren - 1]->GetType().AsConstType();

    // We rely on the custom allocator Alloc to provide enough space
    // for all of the children.
    for (unsigned int i = 0; i < m_numChildren; i++)
    {
        m_children[i] = p_children[i];
    }
}


void 
FreeForm2::BlockExpression::DeleteAlloc(BlockExpression* p_allocated)
{
    // Manually call dtor for block expression.
    p_allocated->~BlockExpression();

    // Dispose of memory, which we allocated in a char[].
    char* mem = reinterpret_cast<char*>(p_allocated);
    delete[] mem;
}


