/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include "CompoundType.h"

FreeForm2::CompoundType::Member::Member(const std::string &p_name,
                                        const TypeImpl &p_type)
    : m_name(p_name), m_type(&p_type) {}

FreeForm2::CompoundType::Member::Member() : m_type(NULL) {}

FreeForm2::CompoundType::CompoundType(Type::TypePrimitive p_prim,
                                      bool p_isConst,
                                      TypeManager *p_typeManager)
    : TypeImpl(p_prim, p_isConst, p_typeManager) {}

bool FreeForm2::CompoundType::IsCompoundType(const TypeImpl &p_type) {
  return p_type.Primitive() == Type::Struct ||
         p_type.Primitive() == Type::StateMachine ||
         p_type.Primitive() == Type::Object;
}
