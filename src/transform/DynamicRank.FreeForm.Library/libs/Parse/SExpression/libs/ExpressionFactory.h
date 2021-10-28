/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#pragma once

#ifndef FREEFORM2_EXPRESSIONFACTORY_H
#define FREEFORM2_EXPRESSIONFACTORY_H

#include <basic_types.h>

#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>

#include "ProgramParseState.h"
#include "TypeImpl.h"

namespace FreeForm2 {
class Expression;
class SimpleExpressionOwner;
class TypeManager;

// Base class that assists parsing by creating an expression from a given
// set of children.
class ExpressionFactory : boost::noncopyable {
 public:
  typedef boost::shared_ptr<ExpressionFactory> Ptr;

  typedef std::vector<const Expression *> ChildVec;

  // Creates an expression from the given children, with the returned
  // expression being owned by the given owner.  p_atom specifies the atom
  // with which this expression factory was identified during parsing,
  // which allows us to provide decent error messages.
  const Expression &Create(
      const ProgramParseState::ExpressionParseState &p_state,
      SimpleExpressionOwner &p_owner, TypeManager &p_typeManager) const;

 private:
  // Creates an expression from the given parse state.
  virtual const Expression &CreateExpression(
      const ProgramParseState::ExpressionParseState &p_state,
      SimpleExpressionOwner &p_owner, TypeManager &p_typeManager) const = 0;

  // Indicates the allowed arity of expressions produced from
  // this factory, in a [min, max] pair (both ends inclusive).
  virtual std::pair<unsigned int, unsigned int> Arity() const = 0;
};
}  // namespace FreeForm2

#endif
