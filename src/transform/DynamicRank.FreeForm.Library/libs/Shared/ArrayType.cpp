#include "ArrayType.h"

#include <boost/foreach.hpp>
#include "FreeForm2Assert.h"
#include <functional>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include "TypeManager.h"

FreeForm2::ArrayType::ArrayType(const TypeImpl& p_child, 
                                bool p_isConst,
                                unsigned int p_dimensions, 
                                unsigned int p_maxElements,
                                TypeManager& p_typeManager)
    : TypeImpl(Type::Array, p_isConst, &p_typeManager),
      m_typeManager(p_typeManager),
      m_derefType(NULL),
      m_oppositeConstnessType(NULL),
      m_isFixedSize(false),
      m_child(p_child), 
      m_dimensionCount(p_dimensions),
      m_maxElements(p_maxElements)
{
    FF2_ASSERT(m_maxElements <= c_maxElements);
    FF2_ASSERT(m_child.Primitive() != Type::Array && m_child.Primitive() != Type::Invalid);
    FF2_ASSERT(p_dimensions > 0);

    if (p_dimensions > c_maxDimensions)
    {
        std::ostringstream err;
        err << "The FreeForm2 language doesn't currently support more than "
            << c_maxDimensions << " dimensions per array.";
        throw std::runtime_error(err.str());
    }

    m_dimensions[0] = 0;
}


FreeForm2::ArrayType::ArrayType(const TypeImpl& p_child, 
                                bool p_isConst,
                                unsigned int p_dimensions,
                                const unsigned int p_elementCounts[],
                                unsigned int p_maxElements,
                                TypeManager& p_typeManager)
    : TypeImpl(Type::Array, p_isConst, &p_typeManager),
      m_typeManager(p_typeManager),
      m_derefType(NULL),
      m_oppositeConstnessType(NULL),
      m_isFixedSize(true),
      m_child(p_child), 
      m_dimensionCount(p_dimensions),
      m_maxElements(p_maxElements)
{
    FF2_ASSERT(m_maxElements <= c_maxElements);
    FF2_ASSERT(m_child.Primitive() != Type::Array && m_child.Primitive() != Type::Invalid);
    FF2_ASSERT(p_dimensions > 0);

    if (p_dimensions > c_maxDimensions)
    {
        std::ostringstream err;
        err << "The FreeForm2 language doesn't currently support more than "
            << c_maxDimensions << " dimensions per array.";
        throw std::runtime_error(err.str());
    }

    memcpy(m_dimensions, p_elementCounts, sizeof(unsigned int) * m_dimensionCount);
}


const FreeForm2::TypeImpl&
FreeForm2::ArrayType::GetChildType() const
{
    return m_child;
}


const unsigned int*
FreeForm2::ArrayType::GetDimensions() const
{
    FF2_ASSERT(IsFixedSize());
    return m_dimensions;
}


unsigned int
FreeForm2::ArrayType::GetDimensionCount() const
{
    return m_dimensionCount;
}


unsigned int
FreeForm2::ArrayType::GetMaxElements() const
{
    return m_maxElements;
}


bool
FreeForm2::ArrayType::IsFixedSize() const
{
    return m_isFixedSize;
}


bool
FreeForm2::ArrayType::IsSameSubType(const TypeImpl& p_other, bool p_ignoreConst) const
{
    FF2_ASSERT(p_other.Primitive() == Type::Array);

    const ArrayType& other = static_cast<const ArrayType&>(p_other);
    if ((IsFixedSize() != other.IsFixedSize()) || !GetChildType().IsSameAs(other.GetChildType(), p_ignoreConst))
    {
        return false;
    }

    if (GetDimensionCount() == other.GetDimensionCount())
    {
        if (IsFixedSize())
        {
            return memcmp(GetDimensions(), 
                               other.GetDimensions(), 
                               sizeof(unsigned int) * GetDimensionCount()) == 0;
        }
        else
        {
            return true;
        }
    }
    else
    {
        return false;
    }
}


std::string
FreeForm2::ArrayType::GetName(const TypeImpl& p_child,
                              bool p_isConst,
                              unsigned int p_dimensionCount,
                              const unsigned int* p_dimensions,
                              unsigned int p_maxElements)
{
    // Ignore the const-ness on arrays, because the const-ness of the child and
    // the constness of the array should be the same.
    std::ostringstream out;
    if (!p_isConst)
    {
        out << "mutable ";
    }

    out << p_child;

    for (unsigned int i = 0; i < p_dimensionCount; i++)
    {
        out << "[";
        if (p_dimensions != NULL)
        {
            out << p_dimensions[i];
        }
        out << "]";
    }
    return out.str();
}


const std::string&
FreeForm2::ArrayType::GetName() const
{
    if (m_name.empty())
    {
        m_name = GetName(GetChildType(), 
                         IsConst(), 
                         GetDimensionCount(), 
                         IsFixedSize() ? GetDimensions() : NULL, 
                         GetMaxElements());
    }
    return m_name;
}


const FreeForm2::TypeImpl& 
FreeForm2::ArrayType::AsConstType() const
{
    if (IsConst())
    {
        return *this;
    }
    else
    {
        if (m_oppositeConstnessType == NULL)
        {
            if (IsFixedSize())
            {
                m_oppositeConstnessType = &m_typeManager.GetArrayType(GetChildType().AsConstType(), 
                                                                      true, 
                                                                      GetDimensionCount(), 
                                                                      GetDimensions(), 
                                                                      GetMaxElements());
            }
            else
            {
                m_oppositeConstnessType = &m_typeManager.GetArrayType(GetChildType().AsConstType(), 
                                                                      true, 
                                                                      GetDimensionCount(), 
                                                                      GetMaxElements());
            }
        }
        return *m_oppositeConstnessType;
    }
}


const FreeForm2::TypeImpl& 
FreeForm2::ArrayType::AsMutableType() const
{
    if (!IsConst())
    {
        return *this;
    }
    else
    {
        if (m_oppositeConstnessType == NULL)
        {
            if (IsFixedSize())
            {
                m_oppositeConstnessType = &m_typeManager.GetArrayType(GetChildType().AsMutableType(), 
                                                                      false, 
                                                                      GetDimensionCount(), 
                                                                      GetDimensions(), 
                                                                      GetMaxElements());
            }
            else
            {
                m_oppositeConstnessType = &m_typeManager.GetArrayType(GetChildType().AsMutableType(), 
                                                                      false, 
                                                                      GetDimensionCount(), 
                                                                      GetMaxElements());
            }
        }
        return *m_oppositeConstnessType;
    }
}


const FreeForm2::TypeImpl&
FreeForm2::ArrayType::GetDerefType() const
{
    if (m_derefType == NULL)
    {
        if (GetDimensionCount() > 1)
        {
            if (IsFixedSize())
            {
                const unsigned int newMaxElements 
                    = GetMaxElements() / GetDimensions()[0];
                
                FF2_ASSERT(GetTypeManager() != NULL);
                m_derefType = &GetTypeManager()->GetArrayType(
                    GetChildType(), IsConst(), GetDimensionCount() - 1, GetDimensions() + 1, newMaxElements);
            }
            else
            {
                // Note that we lose a lot of information here about the
                // possible number of children, since we don't know how to
                // allocate them between the dimensions.
                FF2_ASSERT(GetTypeManager() != NULL);
                m_derefType = &GetTypeManager()->GetArrayType(
                    GetChildType(), IsConst(), GetDimensionCount() - 1, GetMaxElements());
            }
        }
        else
        {
            FF2_ASSERT(GetDimensionCount() == 1);
            m_derefType = &GetChildType();
        }
    }
    return *m_derefType;
}

