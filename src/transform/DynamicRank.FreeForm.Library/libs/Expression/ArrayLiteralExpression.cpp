#include "ArrayLiteralExpression.h"

#include "Allocation.h"
#include "ArrayType.h"
#include <boost/cast.hpp>
#include <boost/scoped_array.hpp>
#include "ConvertExpression.h"
#include "FreeForm2Assert.h"
#include "SimpleExpressionOwner.h"
#include "Visitor.h"
#include "TypeManager.h"
#include "TypeUtil.h"
#include <sstream>


FreeForm2::ArrayLiteralExpression::ArrayLiteralExpression(
    const Annotations& p_annotations,
    const TypeImpl& p_annotatedType, 
    const std::vector<const Expression*>& p_children,
    VariableID p_id,
    TypeManager& p_typeManager)
    : Expression(p_annotations),
      m_isFlat(false), 
      m_type(NULL),
      m_numChildren(static_cast<unsigned int>(p_children.size())),
      m_id(p_id)
{
    for (unsigned int i = 0; i < p_children.size(); i++)
    {
        m_children[i] = p_children[i];
    }

    m_type = &UnifyTypes(p_annotatedType, p_typeManager);
}


FreeForm2::ArrayLiteralExpression::ArrayLiteralExpression(
    const Annotations& p_annotations,
    const ArrayType& p_type,
    const std::vector<const Expression*>& p_elements,
    VariableID p_id)
    : Expression(p_annotations),
      m_type(&p_type),
      m_isFlat(true), 
      m_numChildren(static_cast<unsigned int>(p_elements.size())),
      m_id(p_id)
{
    FF2_ASSERT(p_type.Primitive() == Type::Array);
    FF2_ASSERT(m_type->GetChildType().Primitive() != Type::Array);

    for (unsigned int i = 0; i < p_elements.size(); i++)
    {
        m_children[i] = p_elements[i];
    }
}


const FreeForm2::ArrayType&
FreeForm2::ArrayLiteralExpression::UnifyTypes(const TypeImpl& p_annotatedType, 
                                              TypeManager& p_typeManager)
{
    if (m_numChildren == 0)
    {
        const unsigned int dimensions[] = { 0 };
        return p_typeManager.GetArrayType(p_annotatedType, false, 1, dimensions, 0);
    }

    const TypeImpl* childType = &m_children[0]->GetType();
    
    for (unsigned int i = 0; i < m_numChildren; i++)
    {
        const TypeImpl& nextType = m_children[i]->GetType();
        if (!childType->IsSameAs(nextType, true))
        {
            const TypeImpl& unifiedType 
                = TypeUtil::Unify(*childType, nextType, p_typeManager, true, false);
            if (!unifiedType.IsValid())
            {
                std::ostringstream err;
                err << "Can't unify " << childType << " and " << nextType;
                throw ParseError(err.str(), GetSourceLocation());
            }
            childType = &unifiedType;
        }
        else
        {
            childType = childType->IsConst() ? childType : &nextType;
        }
    }

    const TypeImpl* unifiedType = NULL;
    const TypeImpl* inferredType = NULL;
    std::vector<unsigned int> dimensions;
    unsigned int maxElements = 0;

    if (childType->Primitive() == Type::Array)
    {
        FF2_ASSERT(!IsFlat());
        const ArrayType* childArray = static_cast<const ArrayType*>(childType);
        
        dimensions.push_back(m_numChildren);
        dimensions.insert(dimensions.end(),
                          childArray->GetDimensions(), 
                          childArray->GetDimensions() + childArray->GetDimensionCount());

        inferredType = &childArray->GetChildType();
        unifiedType = &TypeUtil::Unify(p_annotatedType, 
                                       childArray->GetChildType(), 
                                       p_typeManager,
                                       false, 
                                       false);
        for (unsigned int i = 0; i < m_numChildren; i++)
        {
            FF2_ASSERT(m_children[i]->GetType().Primitive() == Type::Array);
            maxElements += static_cast<const ArrayType&>(m_children[i]->GetType()).GetMaxElements();
        }
    }
    else
    {
        unifiedType = &TypeUtil::Unify(*childType, p_annotatedType, p_typeManager, false, false);
        inferredType = childType;
        maxElements = m_numChildren;
        dimensions.push_back(m_numChildren);
    }

    if (!unifiedType->IsValid())
    {
        std::ostringstream err;
        err << "Annotated array type (" 
            << p_annotatedType
            << ") did not match inferred array type ("
            << *inferredType
            << ")";
        throw ParseError(err.str(), GetSourceLocation());
    }

    // Force array literals to be non-const. This is required for intialization
    // of array variables. This would theoretically allow an assignment to an 
    // array literal; however, this is disallowed by Visage grammar.
    return p_typeManager.GetArrayType(*unifiedType, 
                                      false,
                                      static_cast<unsigned int>(dimensions.size()), 
                                      &dimensions[0], 
                                      maxElements);
}


void
FreeForm2::ArrayLiteralExpression::Accept(Visitor& p_visitor) const
{
    size_t stackSize = p_visitor.StackSize();

    if (!p_visitor.AlternativeVisit(*this))
    {
        for (size_t i = 0; i < GetNumChildren(); i++)
        {
            size_t idx = GetNumChildren() - i - 1;

            m_children[idx]->Accept(p_visitor);
        }

        p_visitor.Visit(*this);
    }

    FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}


void 
FreeForm2::ArrayLiteralExpression::AcceptReference(Visitor& p_visitor) const
{
    std::ostringstream err;
    err << "Array literals cannot be used as l-values.";
    throw ParseError(err.str(), GetSourceLocation());
}


boost::shared_ptr<FreeForm2::ArrayLiteralExpression> 
FreeForm2::ArrayLiteralExpression::Alloc(const Annotations& p_annotations,
                                         const TypeImpl& p_childType, 
                                         const std::vector<const Expression*>& p_children,
                                         VariableID p_id,
                                         TypeManager& p_typeManager)
{
    size_t bytes = sizeof(ArrayLiteralExpression) 
        + (p_children.size() - 1) * sizeof(Expression*);

    if (p_children.size() > ArrayType::c_maxElementsPerDimension)
    {
        std::ostringstream err;
        err << "Array literals cannot have more than " << ArrayType::c_maxElementsPerDimension << " elements per dimension.";
        throw ParseError(err.str(), p_annotations.m_sourceLocation);
    }

    // Allocate a shared_ptr that deletes an ArrayLiteralExpression 
    // allocated in a char[].
    boost::shared_ptr<ArrayLiteralExpression> exp(new (new char[bytes]) 
        ArrayLiteralExpression(p_annotations, p_childType, p_children, p_id, p_typeManager), DeleteAlloc);
    return exp;
}


boost::shared_ptr<FreeForm2::ArrayLiteralExpression> 
FreeForm2::ArrayLiteralExpression::Alloc(const Annotations& p_annotations,
                                         const ArrayType& p_type, 
                                         const std::vector<const Expression*>& p_children,
                                         VariableID p_id)
{
    size_t bytes = sizeof(ArrayLiteralExpression) 
        + (p_children.size() - 1) * sizeof(Expression*);
    
    for (unsigned int i = 0; i < p_type.GetDimensionCount(); i++)
    {
        if (p_type.GetDimensions()[i] > ArrayType::c_maxElementsPerDimension)
        {
            std::ostringstream err;
            err << "Array literals cannot have more than " << ArrayType::c_maxElementsPerDimension << " elements per dimension.";
            throw ParseError(err.str(), p_annotations.m_sourceLocation);
        }
    }

    // Allocate a shared_ptr that deletes an ArrayLiteralExpression 
    // allocated in a char[].
    boost::shared_ptr<ArrayLiteralExpression> exp(new (new char[bytes]) 
        ArrayLiteralExpression(p_annotations, p_type, p_children, p_id), DeleteAlloc);
    return exp;
}


const FreeForm2::Expression* const* 
FreeForm2::ArrayLiteralExpression::Begin() const
{
    return &m_children[0];
}


const FreeForm2::Expression* const* 
FreeForm2::ArrayLiteralExpression::End() const
{
    return &m_children[GetNumChildren()];
}


void 
FreeForm2::ArrayLiteralExpression::DeleteAlloc(ArrayLiteralExpression* p_allocated)
{
    // Manually call dtor for array literal expression.
    p_allocated->~ArrayLiteralExpression();

    // Dispose of memory, which we allocated in a char[].
    char* mem = reinterpret_cast<char*>(p_allocated);
    delete[] mem;
}


const FreeForm2::ArrayLiteralExpression& 
FreeForm2::ArrayLiteralExpression::Flatten(SimpleExpressionOwner& p_owner) const
{
    return Flatten(p_owner, NULL, NULL);
}


const FreeForm2::ArrayLiteralExpression& 
FreeForm2::ArrayLiteralExpression::Flatten(SimpleExpressionOwner& p_owner,
                                           const TypeImpl* p_annotatedType,
                                           TypeManager* p_typeManager) const
{
    // We've chosen a pointer+length-based representation of an array
    // (pascal-style, sort-of).  That means we can't use 'compositional'
    // arrays, where multi-dimensional arrays are simply single-dimension 
    // arrays that hold other single-dimension arrays.  As such, we need to
    // ensure that every array is 'square', in that each element in each
    // dimension holds the same number of elements.  Thus, we can infer the
    // size of a multi-dimensional array by knowing the number of
    // sub-sub-elements per sub-element, and the number of sub-elements,
    // making our pointer+length representation practical.

    FF2_ASSERT(!IsFlat());

    // Check for empty arrays with no annotations.
    if (p_annotatedType && p_annotatedType->Primitive() == Type::Unknown)
    {
        p_annotatedType = NULL;
    }

    if (m_numChildren == 0 && !p_annotatedType)
    {
        throw ParseError("Can't infer array element type from empty array.", GetSourceLocation());
    }

    // Stack of unflattened array elements, along with the number of
    // dimensions of elements they contain.
    std::vector<std::pair<unsigned int, const ArrayLiteralExpression*>> 
        stack(1, std::make_pair(m_type->GetDimensionCount(), this));

    // Sizes of each dimension.
    std::vector<unsigned int> dimensions(m_type->GetDimensionCount(), 0);

    // Indication, for each dimension, of whether we currently know how big
    // the dimension is.
    std::vector<bool> dimensionSizeKnown(m_type->GetDimensionCount(), false);

    std::vector<unsigned int> elementCount(m_type->GetDimensionCount(), 0);

    // Flattened vector of elements.
    std::vector<const Expression*> elements;
    elements.reserve(m_type->GetMaxElements());

    while (!stack.empty())
    {
        FF2_ASSERT(stack.back().second != NULL);
        const ArrayLiteralExpression& current = *stack.back().second;
        unsigned int currentDimension = stack.back().first;
        unsigned int numChildren = static_cast<unsigned int>(current.GetNumChildren());
        stack.pop_back();

        // Check dimensions.
        if (dimensionSizeKnown[currentDimension - 1])
        {
            if (numChildren != dimensions[currentDimension - 1])
            {
                std::ostringstream err;
                err << "Array element ";
                for (unsigned int i = 0; i < currentDimension; i++)
                {
                    err << (i != 0 ? ", " : "") << elementCount[i];
                }
                err << " was expected to be a literal array (as all "
                    << "non-leaf elements of a literal array must be), "
                    << "but was not";
                throw ParseError(err.str(), GetSourceLocation());
            }
        }
        else
        {
            dimensionSizeKnown[currentDimension - 1] = true;
            dimensions[currentDimension - 1] = numChildren;
        }

        if (currentDimension > 1)
        {
            // Check that it's an array literal.  Note that we iterate
            // through elements in reverse order, so that they come off the
            // stack in the correct order.
            const Expression* const* iter = current.End();
            while (iter != current.Begin())
            {
                --iter;

                // We need to ensure that the child is a literal array, so
                // that we can check it's square (and hence giving us
                // assurance that our simple representation is valid).  This
                // might be better done with a type-indication returning
                // member function, but does require dynamic type information, 
                // short of making ArrayLiteralExpression aware of the 
                // types of children it has (not a good idea).
                const ArrayLiteralExpression* child 
                    = dynamic_cast<const ArrayLiteralExpression*>(*iter);
                if (child != NULL)
                {
                    stack.push_back(std::make_pair(currentDimension - 1, child));
                }
                else
                {
                    std::ostringstream err;
                    err << "Array element ";
                    for (unsigned int i = 0; i < currentDimension; i++)
                    {
                        err << (i != 0 ? ", " : "") << elementCount[i];
                    }
                    err << " was expected to be a literal array (as all "
                        << "non-leaf elements of a literal array must be), "
                        << "but was not";
                }
            }
        }
        else
        {
            // We're down to elements of the array, save them.
            FF2_ASSERT(currentDimension == 1);
            for (const Expression* const* iter = current.Begin(); 
                 iter != current.End(); 
                 ++iter)
            {
                elements.push_back(*iter);
            }
        }

        // Keep track of the number of elements processed, for decent error
        // messages.
        elementCount[currentDimension - 1]++;
    }

    // Ensure all dimensions are set.
    for (unsigned int i = 0; i < dimensions.size(); i++)
    {
        FF2_ASSERT(dimensionSizeKnown[i]);
    }

    FF2_ASSERT(elements.size() == m_type->GetMaxElements());

    const ArrayType* newType = m_type;
    if (p_annotatedType != NULL)
    {
        FF2_ASSERT(p_typeManager != NULL);
        const TypeImpl& type 
            = TypeUtil::Unify(*p_annotatedType, m_type->GetChildType(), *p_typeManager, false, false);
        if (type != m_type->GetChildType())
        {
            FF2_ASSERT(m_type->IsFixedSize());
            newType = &p_typeManager->GetArrayType(type, 
                                                   m_type->IsConst(), 
                                                   m_type->GetDimensionCount(), 
                                                   m_type->GetDimensions(), 
                                                   m_type->GetMaxElements());
        }
    }

    boost::shared_ptr<ArrayLiteralExpression> flat = Alloc(GetAnnotations(), *newType, elements, m_id);
    p_owner.AddExpression(flat);
    return *flat;
}


bool
FreeForm2::ArrayLiteralExpression::IsFlat() const
{
    return m_isFlat;
}


size_t
FreeForm2::ArrayLiteralExpression::GetNumChildren() const
{
    return m_numChildren;
}


const FreeForm2::TypeImpl&
FreeForm2::ArrayLiteralExpression::GetType() const
{
    return *m_type;
}


FreeForm2::VariableID
FreeForm2::ArrayLiteralExpression::GetId() const
{
    return m_id;
}
