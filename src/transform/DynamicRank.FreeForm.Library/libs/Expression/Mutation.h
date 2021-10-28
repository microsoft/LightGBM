/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#pragma once

#ifndef FREEFORM2_MUTATION_H
#define FREEFORM2_MUTATION_H

#include "CompoundType.h"
#include "Expression.h"

namespace FreeForm2 {
class MutationExpression : public Expression {
 public:
  // Create a mutation expression from a type and an initialiser.
  MutationExpression(const Annotations &p_annotations,
                     const Expression &p_lvalue, const Expression &p_rvalue);

  // Virtual methods inherited from Expression.
  virtual void Accept(Visitor &p_visitor) const override;
  virtual const TypeImpl &GetType() const override;
  virtual size_t GetNumChildren() const override;

  // Get the l-value and r-value expressions.
  const Expression &GetLeftValue() const;
  const Expression &GetRightValue() const;

 private:
  // Expression to be mutated.
  const Expression &m_lvalue;

  // Value assigned to l-value after mutation.
  const Expression &m_rvalue;
};

class TypeInitializerExpression : public Expression {
 public:
  // This struct represents a member-initializer pair. Note that
  // std::pair is not used because it is a non-POD type, and doesn't
  // work with the struct hack.
  struct Initializer {
    const CompoundType::Member *m_member;
    const Expression *m_initializer;
    size_t m_version;
  };

  // Allocate a new TypeInitializerExpression.
  static boost::shared_ptr<TypeInitializerExpression> Alloc(
      const Annotations &p_annotations, const CompoundType &p_type,
      const Initializer *p_initializers, size_t p_numInitializers);

  // Virtual methods inherited from Expression.
  virtual void Accept(Visitor &p_visitor) const override;
  virtual const TypeImpl &GetType() const override;
  virtual size_t GetNumChildren() const override;

  // Access initializers.
  const Initializer *BeginInitializers() const;
  const Initializer *EndInitializers() const;

 private:
  // Create a type initializer with member-expression pairs to initialize
  // each member. All members in the CompoundType must be specified.
  TypeInitializerExpression(const Annotations &p_annotations,
                            const CompoundType &p_type,
                            const Initializer *p_initializers,
                            size_t p_initializerCount);

  // Validate that all members in the CompoundType are initialized.
  void ValidateMembers() const;

  // The type being initialized.
  const CompoundType &m_type;

  // The number of initializers.
  size_t m_numInitializers;

  // Initializer list, allocated using the struct hack.
  Initializer m_initializers[1];
};
};  // namespace FreeForm2

#endif
