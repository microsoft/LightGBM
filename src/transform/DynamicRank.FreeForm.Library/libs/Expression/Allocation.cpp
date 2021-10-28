/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include "Allocation.h"

#include "ArrayType.h"
#include "FreeForm2Assert.h"

using namespace FreeForm2;

FreeForm2::Allocation::Allocation(AllocationType p_allocType, VariableID p_id,
                                  const TypeImpl &p_type)
    : m_allocType(p_allocType), m_id(p_id), m_type(p_type), m_children(0) {
  if (p_allocType == ArrayLiteral) {
    FF2_ASSERT(m_type.Primitive() == Type::Array);
    m_children = 1;

    const ArrayType &arrayType = static_cast<const ArrayType &>(p_type);
    for (UInt32 i = 0; i < arrayType.GetDimensionCount(); i++) {
      m_children *= arrayType.GetDimensions()[i];
    }
  }
}

FreeForm2::Allocation::Allocation(AllocationType p_allocType, VariableID p_id,
                                  const TypeImpl &p_type, size_t p_children)
    : m_allocType(p_allocType),
      m_id(p_id),
      m_type(p_type),
      m_children(p_children) {}

FreeForm2::Allocation::AllocationType FreeForm2::Allocation::GetAllocationType()
    const {
  return m_allocType;
}

const TypeImpl &FreeForm2::Allocation::GetType() const { return m_type; }

VariableID FreeForm2::Allocation::GetAllocationId() const { return m_id; }

size_t FreeForm2::Allocation::GetNumChildren() const { return m_children; }
