/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#pragma once

#include <boost/shared_ptr.hpp>
#include "TypeImpl.h"
#include <vector>

namespace FreeForm2
{
    class TypeFactory;

    class ArrayType : public TypeImpl
    {
    public:
        // Maximum number of dimensions per array.
        const static unsigned int c_maxDimensions = 7;

        // Maximum number of leaf elements per array.
        const static unsigned int c_maxElements = 16384;

        // Maximum number of elements per array dimension.
        const static unsigned int c_maxElementsPerDimension = 255;

        // Get the child type.
        const TypeImpl &GetChildType() const;

        // Get a vector containing the sizes of each dimension.
        const unsigned int *GetDimensions() const;

        // Get the number of dimensions. This is a convenience method for
        // getting the size of the vector returned from GetDimensions.
        unsigned int GetDimensionCount() const;

        // The the maximum number of elements in the array.
        unsigned int GetMaxElements() const;

        // Tests whether this array is fixed-size.
        bool IsFixedSize() const;

        // Get the name of an array type for the type desribed by the
        // parameters.
        static std::string
        GetName(const TypeImpl &p_child,
                bool p_isConst,
                unsigned int p_dimensionCount,
                const unsigned int *p_dimensions,
                unsigned int p_maxElements);

        // Get a string representation of the type.
        virtual const std::string &GetName() const override;

        // Methods to get types derived from this type.
        virtual const TypeImpl &AsConstType() const override;
        virtual const TypeImpl &AsMutableType() const override;
        const TypeImpl &GetDerefType() const;

    private:
        // Construct a variable-size array type.
        ArrayType(const TypeImpl &p_child,
                  bool p_isConst,
                  unsigned int p_dimensions,
                  unsigned int p_maxElements,
                  TypeManager &p_typeManager);

        // Construct a fixed-size array type.
        ArrayType(const TypeImpl &p_child,
                  bool p_isConst,
                  unsigned int p_dimensions,
                  const unsigned int p_elementCounts[],
                  unsigned int p_maxElements,
                  TypeManager &p_typeManager);

        virtual bool IsSameSubType(const TypeImpl &p_other, bool p_ignoreConst) const override;

        // Give the TypeManager access to the ArrayType constructor.
        friend class TypeManager;

        // The type manager that created this type.
        TypeManager &m_typeManager;

        // The name of this type.
        mutable std::string m_name;

        // Derived type references, stored for efficiency.
        mutable const TypeImpl *m_derefType;
        mutable const TypeImpl *m_oppositeConstnessType;

        // This flag indicates whether or not this array has a fixed size.
        bool m_isFixedSize;

        // Type of array.  Note that we only allow basic types as child of
        // an array, because we're using old-school c-style
        // multi-dimensional arrays, rather than something compositional.
        const TypeImpl &m_child;

        // Maximum number of elements of this array.  We use this to track
        // the maximum size of the array, and statically allocate space for it.
        unsigned int m_maxElements;

        // The number of dimensions contained in this array.
        unsigned int m_dimensionCount;

        // Number of elements in each dimension, allocated using the struct hack.
        unsigned int m_dimensions[1];
    };
}
