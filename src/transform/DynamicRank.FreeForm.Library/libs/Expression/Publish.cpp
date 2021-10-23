#include "Publish.h"
#include "Visitor.h"
#include "FreeForm2Assert.h"
#include <sstream>
#include "TypeUtil.h"

FreeForm2::PublishExpression::PublishExpression(const Annotations& p_annotations,
                                                const std::string& p_featureName,
                                                const Expression& p_value)
    : Expression(p_annotations),
      m_featureName(p_featureName),
      m_value(p_value)
{
}


void
FreeForm2::PublishExpression::Accept(Visitor& p_visitor) const 
{
    if (!p_visitor.AlternativeVisit(*this))
    {
        m_value.Accept(p_visitor);

        p_visitor.Visit(*this);
    }   
}


size_t
FreeForm2::PublishExpression::GetNumChildren() const
{
    return 1;
}


const FreeForm2::TypeImpl&
FreeForm2::PublishExpression::GetType() const
{
    return FreeForm2::TypeImpl::GetVoidInstance();
}


const FreeForm2::Expression&
FreeForm2::PublishExpression::GetValue() const
{
    return m_value;
}


const std::string&
FreeForm2::PublishExpression::GetFeatureName() const
{
    return m_featureName;
}


FreeForm2::DirectPublishExpression::DirectPublishExpression(const Annotations& p_annotations,
                                                            const std::string& p_featureName,
                                                            const Expression** p_indices,
                                                            const unsigned int p_numIndices,
                                                            const Expression& p_value)
    : Expression(p_annotations),
      m_featureName(p_featureName),
      m_numIndices(p_numIndices),
      m_value(p_value)
{
    for (size_t i = 0; i < m_numIndices; i++)
    {
        m_indices[i] = p_indices[i];
    }
}


boost::shared_ptr<FreeForm2::DirectPublishExpression>
FreeForm2::DirectPublishExpression::Alloc(const Annotations& p_annotations,
                                          const std::string& p_featureName,
                                          const Expression** p_indices,
                                          const unsigned int p_numIndices,
                                          const Expression& p_value)
{
    size_t bytes = sizeof(DirectPublishExpression) 
        + (p_numIndices - 1) * sizeof(Expression*);

    // Allocate a shared_ptr that deletes an DirectPublishExpression 
    // allocated in a char[].
    boost::shared_ptr<DirectPublishExpression> exp(new (new char[bytes]) 
        DirectPublishExpression(p_annotations, p_featureName, p_indices, p_numIndices, p_value), DeleteAlloc);
    return exp;
}


void 
FreeForm2::DirectPublishExpression::DeleteAlloc(DirectPublishExpression* p_allocated)
{
    // Manually call dtor for expression.
    p_allocated->~DirectPublishExpression();

    // Dispose of memory, which we allocated in a char[].
    char* mem = reinterpret_cast<char*>(p_allocated);
    delete[] mem;
}


void
FreeForm2::DirectPublishExpression::Accept(Visitor& p_visitor) const 
{
    if (!p_visitor.AlternativeVisit(*this))
    {
        for (unsigned int i = 0; i < m_numIndices; ++i)
        {
            m_indices[i]->Accept(p_visitor);
        }

        m_value.Accept(p_visitor);

        p_visitor.Visit(*this);
    }   
}


size_t
FreeForm2::DirectPublishExpression::GetNumChildren() const
{
    return 1 + m_numIndices;
}


const FreeForm2::TypeImpl&
FreeForm2::DirectPublishExpression::GetType() const
{
    for (unsigned int i = 0; i < m_numIndices; ++i)
    {
        if (!m_indices[i]->GetType().IsIntegerType())
        {
            std::ostringstream err;
            err << "Index type in array publication is not an integer type "
                << "instead, it is a " << m_indices[i]->GetType() << ").";
            throw ParseError(err.str(), m_indices[i]->GetSourceLocation());
        }
    }

    return FreeForm2::TypeImpl::GetVoidInstance();
}


const FreeForm2::Expression&
FreeForm2::DirectPublishExpression::GetValue() const
{
    return m_value;
}


const std::string&
FreeForm2::DirectPublishExpression::GetFeatureName() const
{
    return m_featureName;
}


unsigned int
FreeForm2::DirectPublishExpression::GetNumIndices() const
{
    return m_numIndices;
}


const FreeForm2::Expression* const *
FreeForm2::DirectPublishExpression::GetIndices() const
{
    return m_indices;
}
