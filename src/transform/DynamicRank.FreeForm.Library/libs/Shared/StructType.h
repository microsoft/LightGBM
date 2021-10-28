/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#pragma once

#ifndef FREEFORM2_STRUCT_TYPE_H
#define FREEFORM2_STRUCT_TYPE_H

#include <map>
#include <string>
#include <vector>

#include "CompoundType.h"

namespace FreeForm2 {
// Structure types are C-like structs, which are named and contain named
// members.
class StructType : public CompoundType {
 public:
  // A struct that holds the information about members of a struct.
  class MemberInfo : public CompoundType::Member {
   public:
    // Constructor to initialize all members of the class.
    MemberInfo(const std::string &p_name, const TypeImpl &p_type,
               const std::string &p_externName, size_t p_offset, size_t p_size);

    // Default constructor to initialize members to empty values.
    MemberInfo();

    // The C++ name of the member.
    std::string m_externName;

    // The offset (in bytes) of the member from the beginning of the struct.
    size_t m_offset;

    // The size (in bytes) of this member.
    size_t m_size;

    // Equality operators.
    bool operator==(const MemberInfo &p_other) const;
    bool operator!=(const MemberInfo &p_other) const;
  };

  // Find a member of the given name within this struct.
  virtual const Member *FindMember(const std::string &p_name) const override;

  // Find a member info object of the given name within the struct.
  // Behaves similarly to FindMember.
  const MemberInfo *FindStructMember(const std::string &p_name) const;

  // Gets a vector of member information objects.
  const std::vector<MemberInfo> &GetMembers() const;

  // Gets the name of this struct.
  virtual const std::string &GetName() const override;

  // Gets the C++ name of this struct.
  const std::string &GetExternName() const;

  // Get a string representation of the type.
  virtual std::string GetString() const;

  // Create derived types based on this type.
  virtual const TypeImpl &AsConstType() const override;
  virtual const TypeImpl &AsMutableType() const override;

 private:
  // Creates a new StructInfo based on the list of members.
  StructType(const std::string &p_name, const std::string &p_externName,
             const std::vector<MemberInfo> &p_members, bool p_isConst,
             TypeManager &p_typeManager);

  friend class TypeManager;

  // Compare subclass type data.
  virtual bool IsSameSubType(const TypeImpl &p_type,
                             bool p_ignoreConst) const override;

  // The (ordered) list of members of this struct.
  std::vector<MemberInfo> m_members;

  // A mapping between names and MemberInfo structures.
  std::map<std::string, const MemberInfo *> m_memberMapping;

  // The name exposed to Visage.
  std::string m_name;

  // The C++ name of the struct.
  std::string m_externName;
};
}  // namespace FreeForm2

#endif