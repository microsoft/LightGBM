/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#pragma once

#ifndef FREEFORM2_STATE_MACHINE_TYPE_H
#define FREEFORM2_STATE_MACHINE_TYPE_H

#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <string>

#include "CompoundType.h"

namespace FreeForm2 {
class TypeManager;
class StateMachineExpression;

// This class represents the type of a state machine. State machine types
// contain the state variables which constitute the members of the type,
// as well as a weak reference to the definition of the state machine.
class StateMachineType : public CompoundType {
 public:
  // Destructor to correctly dispose of members.
  ~StateMachineType();

  // Get the name of the state machine.
  virtual const std::string &GetName() const override;

  // Find a member by name.
  virtual const Member *FindMember(const std::string &p_name) const override;

  // Iterate over members.
  typedef const Member *MemberIterator;
  MemberIterator BeginMembers() const;
  MemberIterator EndMembers() const;

  // Get the number of members in the type.
  size_t GetMemberCount() const;

  // Create derived types based on this type.
  virtual const TypeImpl &AsConstType() const override;
  virtual const TypeImpl &AsMutableType() const override;

  // Manipulate the definition of this StateMacineType.
  bool HasDefinition() const;
  boost::shared_ptr<const StateMachineExpression> GetDefinition() const;

 private:
  // This private constructor for StateMachineType requires memory
  // allocated for the size of the StateMachineType, plus the size of
  // the number of state variables and the size of the StateExpression
  // pointer, less the size of one char. This should be done
  // by the TypeManager.
  StateMachineType(TypeManager &p_typeManager, const std::string &p_name,
                   const Member *p_members, size_t p_numMembers,
                   boost::weak_ptr<const StateMachineExpression> p_expr);

  friend class TypeManager;
  friend class StateMachineExpression;

  // Test if this function type is the same as another.
  virtual bool IsSameSubType(const TypeImpl &p_type,
                             bool p_ignoreConst) const override;

  // The state machine name.
  std::string m_name;

  // The state machine definition associated with this type.
  mutable boost::weak_ptr<const StateMachineExpression> m_expr;

  // The number of state variables in the data blob.
  size_t m_numMembers;

  // A blob of data holding both the state variables followed by state
  // expression pointers, allocated using the struct hack.
  Member m_members[1];
};
}  // namespace FreeForm2

#endif
