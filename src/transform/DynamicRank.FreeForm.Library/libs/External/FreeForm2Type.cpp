/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include "FreeForm2Type.h"

#include <stdlib.h>

#include <numeric>
#include <sstream>

#include "ArrayType.h"
#include "FreeForm2Assert.h"
#include "FunctionType.h"
#include "StructType.h"
#include "TypeImpl.h"
#include "TypeManager.h"

FreeForm2::Type::Type(const TypeImpl &p_impl) : m_impl(p_impl) {}

FreeForm2::Type::TypePrimitive FreeForm2::Type::Primitive() const {
  return m_impl.Primitive();
}

const char *FreeForm2::Type::Name(TypePrimitive p_type) {
  switch (p_type) {
    case Float:
      return "float";
    case Int:
      return "int";
    case UInt64:
      return "uint64";
    case Int32:
      return "int32";
    case UInt32:
      return "uint32";
    case Bool:
      return "bool";
    case Array:
      return "array";
    case Void:
      return "void";
    case Stream:
      return "stream";
    case Word:
      return "word";
    case InstanceHeader:
      return "instanceHeader";
    case BodyBlockHeader:
      return "bodyBlockHeader";
    case Unknown:
      return "unknown";
    default:
      return "<invalid type>";
  };
}

bool FreeForm2::Type::operator==(const Type &p_other) const {
  return GetImplementation() == p_other.GetImplementation();
}

const FreeForm2::TypeImpl &FreeForm2::Type::GetImplementation() const {
  return m_impl;
}

FreeForm2::Type::TypePrimitive FreeForm2::Type::ParsePrimitive(
    SIZED_STRING p_string) {
  for (int i = 0; i < Type::Invalid; i++) {
    Type::TypePrimitive prim = static_cast<Type::TypePrimitive>(i);
    const char *name = Name(prim);

    if (name != NULL) {
      size_t len = strlen(name);

      if (p_string.cbData == len && memcmp(p_string.pcData, name, len) == 0) {
        return prim;
      }
    }
  }

  return Type::Invalid;
}

std::ostream &FreeForm2::operator<<(std::ostream &p_out, const Type &p_type) {
  return p_out << p_type.GetImplementation();
}

FreeForm2::TypeFactory::TypeFactory(std::auto_ptr<TypeManager> p_typeManager)
    : m_typeManager(p_typeManager.release()) {}

const FreeForm2::TypeImpl &FreeForm2::TypeFactory::GetFloatType() {
  return TypeImpl::GetFloatInstance(true);
}

const FreeForm2::TypeImpl &FreeForm2::TypeFactory::GetIntType() {
  return TypeImpl::GetIntInstance(true);
}

const FreeForm2::TypeImpl &FreeForm2::TypeFactory::GetUInt64Type() {
  return TypeImpl::GetUInt64Instance(true);
}

const FreeForm2::TypeImpl &FreeForm2::TypeFactory::GetInt32Type() {
  return TypeImpl::GetInt32Instance(true);
}

const FreeForm2::TypeImpl &FreeForm2::TypeFactory::GetUInt32Type() {
  return TypeImpl::GetUInt32Instance(true);
}

const FreeForm2::TypeImpl &FreeForm2::TypeFactory::GetBoolType() {
  return TypeImpl::GetBoolInstance(true);
}

const FreeForm2::TypeImpl &FreeForm2::TypeFactory::GetVoidType() {
  return TypeImpl::GetVoidInstance();
}

const FreeForm2::TypeImpl &FreeForm2::TypeFactory::GetStreamType() {
  return TypeImpl::GetStreamInstance(true);
}

const FreeForm2::TypeImpl &FreeForm2::TypeFactory::GetWordType() {
  return TypeImpl::GetWordInstance(true);
}

const FreeForm2::TypeImpl &FreeForm2::TypeFactory::GetInstanceHeaderType() {
  return TypeImpl::GetInstanceHeaderInstance(true);
}

const FreeForm2::TypeImpl &FreeForm2::TypeFactory::GetBodyBlockHeaderType() {
  return TypeImpl::GetBodyBlockHeaderInstance(true);
}

const FreeForm2::TypeImpl &FreeForm2::TypeFactory::GetArrayType(
    const TypeImpl &p_child, const UInt32 *p_dimensions,
    UInt32 p_dimensionCount) {
  const UInt32 maxElements =
      std::accumulate(p_dimensions, p_dimensions + p_dimensionCount, 1,
                      std::multiplies<UInt32>());
  return m_typeManager->GetArrayType(p_child, true, p_dimensionCount,
                                     p_dimensions, maxElements);
}

const FreeForm2::TypeImpl &FreeForm2::TypeFactory::GetStructType(
    const std::string &p_name, const StructMember *p_members,
    UInt32 p_memberCount) {
  std::vector<StructType::MemberInfo> members;
  members.reserve(p_memberCount);
  for (UInt32 i = 0; i < p_memberCount; i++) {
    members.push_back(StructType::MemberInfo(
        p_members[i].first, p_members[i].second, p_members[i].first, 0, 0));
  }
  return m_typeManager->GetStructType(p_name, p_name, members, true);
}

const FreeForm2::TypeImpl &FreeForm2::TypeFactory::GetFunctionType(
    const TypeImpl &p_returnType, const TypeImpl *const *p_parameters,
    UInt32 p_parameterCount) {
  return m_typeManager->GetFunctionType(p_returnType, p_parameters,
                                        p_parameterCount);
}

const FreeForm2::TypeImpl *FreeForm2::TypeFactory::FindType(
    const std::string &p_name) const {
  return m_typeManager->GetTypeInfo(p_name);
}

const FreeForm2::TypeManager &FreeForm2::TypeFactory::GetTypeManager() const {
  return *m_typeManager;
}
