/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#pragma once

#ifndef FREEFORM2_TYPEMANAGER_H
#define FREEFORM2_TYPEMANAGER_H

#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <memory>
#include <string>

#include "CompoundType.h"
#include "FunctionType.h"
#include "ObjectType.h"
#include "StructType.h"

namespace FreeForm2 {
class ArrayType;
class ExternalDataManager;
class StateMachineExpression;
class StateMachineType;
class TypeImpl;

// The TypeManager class acts as both an owner and a factory for TypeImpl
// objects. Type managers are implemented as a list, where each type
// manager has a parent (with the exception of the global type manager).
// If a type manager cannot find information about a type, it should query
// its parent type manager.
class TypeManager : boost::noncopyable {
 public:
  // Create a TypeManager with an optional parent.
  explicit TypeManager(const TypeManager *p_parent);

  // Unimplemented destructor.
  virtual ~TypeManager();

  // Gets the static instance of the TypeManager. This function is not
  // thread-safe.
  static const TypeManager &GetGlobalTypeManager();

  // Create a type manager with the default implementation.
  static std::auto_ptr<TypeManager> CreateTypeManager();

  // Create a type manager with the given parent.
  static std::auto_ptr<TypeManager> CreateTypeManager(
      const TypeManager &p_parent);

  // Create a type manager with the given parent.
  static std::auto_ptr<TypeManager> CreateTypeManager(
      const ExternalDataManager &p_parent);

  // Gets the type information for the provided name. Returns NULL if the
  // type is not found. This method should check this type manager as
  // well as its parent type manager.
  virtual const TypeImpl *GetTypeInfo(const std::string &p_name) const = 0;

  // Get a variable sized array type owned by this TypeManager.
  virtual const ArrayType &GetArrayType(const TypeImpl &p_child, bool p_isConst,
                                        unsigned int p_dimensions,
                                        unsigned int p_maxElements) = 0;

  // Get a fixed-sized array type owned by this TypeManager.
  virtual const ArrayType &GetArrayType(const TypeImpl &p_child, bool p_isConst,
                                        unsigned int p_dimensions,
                                        const unsigned int p_elementCounts[],
                                        unsigned int p_maxElements) = 0;

  // Returns an array type owned by this TypeManager which has the same
  // properties as another ArrayType.
  virtual const ArrayType &GetArrayType(const ArrayType &p_type) = 0;

  // Get a struct type owned by this TypeManager. The TypeManager is not
  // required to allow multiple non-unique names exist in the context of
  // its owned types.
  virtual const StructType &GetStructType(
      const std::string &p_name, const std::string &p_externName,
      const std::vector<StructType::MemberInfo> &p_members, bool p_isConst) = 0;

  // Returns a struct type owned by this TypeManager which has the same
  // properties as another StructType.
  virtual const StructType &GetStructType(const StructType &p_type) = 0;

  // Get an object type owned by this TypeManager. The TypeManager is not
  // required to allow multiple non-unique names exist in the context of
  // its owned types.
  virtual const ObjectType &GetObjectType(
      const std::string &p_name, const std::string &p_externName,
      const std::vector<ObjectType::ObjectMember> &p_members,
      bool p_isConst) = 0;

  // Returns an object type owned by this TypeManager which has the same
  // properties as another StructType.
  virtual const ObjectType &GetObjectType(const ObjectType &p_type) = 0;

  // Returns a state machine type owned by this TypeManager. The type
  // manager is not required to allow multiple non-unique names to exist
  // in the context of its owned types.
  virtual const StateMachineType &GetStateMachineType(
      const std::string &p_name, const CompoundType::Member *p_members,
      size_t p_numMembers,
      boost::weak_ptr<const StateMachineExpression> p_expr) = 0;

  // Returns a state machine type owned by this TypeManager which has the
  // same properties as another state machine type.
  virtual const StateMachineType &GetStateMachineType(
      const StateMachineType &p_type) = 0;

  // Get a function type owned by this TypeManager. The TypeManager will just
  // store one function type per signature.
  virtual const FunctionType &GetFunctionType(const TypeImpl &p_returnType,
                                              const TypeImpl *const *p_params,
                                              size_t p_numParams) = 0;

  // Returns a function type owned by this TypeManager which has the same
  // properties as another FunctionType.
  virtual const FunctionType &GetFunctionType(const FunctionType &p_type) = 0;

  // Get the parent of this type manager. This function may return NULL.
  const TypeManager *GetParent() const;

 protected:
  // Create an array type.
  boost::shared_ptr<const ArrayType> CreateArrayType(
      const TypeImpl &p_child, bool p_isConst, unsigned int p_dimensions,
      unsigned int p_maxElements);

  // Create a fixed-sized array type.
  boost::shared_ptr<const ArrayType> CreateArrayType(
      const TypeImpl &p_child, bool p_isConst, unsigned int p_dimensions,
      const unsigned int p_elementCounts[], unsigned int p_maxElements);

  // Create a deep copy of an array type. The type should not contain
  // cyclical dependencies.
  boost::shared_ptr<const ArrayType> CopyArrayType(
      const ArrayType &p_arrayType);

  // Create a struct type.
  boost::shared_ptr<const StructType> CreateStructType(
      const std::string &p_name, const std::string &p_externName,
      const std::vector<StructType::MemberInfo> &p_members, bool p_isConst);

  // Create a deep copy of a state machine type.
  boost::shared_ptr<const StructType> CopyStructType(const StructType &p_type);

  // Create an object type.
  boost::shared_ptr<const ObjectType> CreateObjectType(
      const std::string &p_name, const std::string &p_externName,
      const std::vector<ObjectType::ObjectMember> &p_members, bool p_isConst);

  // Create a deep copy of an object type.
  boost::shared_ptr<const ObjectType> CopyObjectType(const ObjectType &p_type);

  // Create a new state machine type.
  boost::shared_ptr<const StateMachineType> CreateStateMachineType(
      const std::string &p_name, const CompoundType::Member *p_members,
      size_t p_numMembers,
      boost::weak_ptr<const StateMachineExpression> p_expr);

  // Create a deep copy of a state machine type.
  boost::shared_ptr<const StateMachineType> CopyStateMachineType(
      const StateMachineType &p_type);

  // Create a function type.
  boost::shared_ptr<const FunctionType> CreateFunctionType(
      const TypeImpl &p_returnType, const TypeImpl *const *p_params,
      size_t p_numParams);

  // Create a deep copy of a state machine type.
  boost::shared_ptr<const FunctionType> CopyFunctionType(
      const FunctionType &p_type);

 private:
  // Get a copy of the specified type that is owned by this TypeManager.
  const TypeImpl &GetChildType(const TypeImpl &p_type);

  // The parent of this type manager.
  const TypeManager *m_parent;
};

// This class acts as a lightweight TypeManager to keep ownership of
// TypeImpl objects. This class does not provide any name lookup of types.
class AnonymousTypeManager : public TypeManager {
 public:
  // Construct an anonymous type manager with no parent.
  AnonymousTypeManager();

  // The anonymous type manager does not save any
  virtual const TypeImpl *GetTypeInfo(const std::string &p_name) const override;

  // Get a variable sized array type owned by this TypeManager.
  virtual const ArrayType &GetArrayType(const TypeImpl &p_child, bool p_isConst,
                                        unsigned int p_dimensions,
                                        unsigned int p_maxElements) override;

  // Get a fixed-sized array type owned by this TypeManager.
  virtual const ArrayType &GetArrayType(const TypeImpl &p_child, bool p_isConst,
                                        unsigned int p_dimensions,
                                        const unsigned int p_elementCounts[],
                                        unsigned int p_maxElements) override;

  // Returns an array type owned by this TypeManager which has the same
  // properties as another ArrayType.
  virtual const ArrayType &GetArrayType(const ArrayType &p_type) override;

  // Get a struct type owned by this TypeManager. The TypeManager is not
  // required to allow multiple non-unique names exist in the context of
  // its owned types.
  virtual const StructType &GetStructType(
      const std::string &p_name, const std::string &p_externName,
      const std::vector<StructType::MemberInfo> &p_members,
      bool p_isConst) override;

  // Returns a struct type owned by this TypeManager which has the same
  // properties as another StructType.
  virtual const StructType &GetStructType(const StructType &p_type) override;

  // Get an object type owned by this TypeManager. The TypeManager is not
  // required to allow multiple non-unique names exist in the context of
  // its owned types.
  virtual const ObjectType &GetObjectType(
      const std::string &p_name, const std::string &p_externName,
      const std::vector<ObjectType::ObjectMember> &p_members,
      bool p_isConst) override;

  // Returns an object type owned by this TypeManager which has the same
  // properties as another ObjectType.
  virtual const ObjectType &GetObjectType(const ObjectType &p_type) override;

  // Returns a state machine type owned by this TypeManager. The type
  // manager is not required to allow multiple non-unique names to exist
  // in the context of its owned types.
  virtual const StateMachineType &GetStateMachineType(
      const std::string &p_name, const CompoundType::Member *p_members,
      size_t p_numMembers,
      boost::weak_ptr<const StateMachineExpression> p_expr) override;

  // Returns a state machine type owned by this TypeManager which has the
  // same properties as another state machine type.
  virtual const StateMachineType &GetStateMachineType(
      const StateMachineType &p_type) override;

  // Get a function type owned by this TypeManager.
  virtual const FunctionType &GetFunctionType(const TypeImpl &p_returnType,
                                              const TypeImpl *const *p_params,
                                              size_t p_numParams) override;

  // Get a function type owned by this TypeManager.
  virtual const FunctionType &GetFunctionType(
      const FunctionType &p_type) override;

 private:
  // A vector containing all types created by this manager.
  std::vector<boost::shared_ptr<const TypeImpl> > m_types;
};
}  // namespace FreeForm2

#endif
