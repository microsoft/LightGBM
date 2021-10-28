/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#pragma once

#ifndef FREEFORM2_SYMBOL_TABLE_H
#define FREEFORM2_SYMBOL_TABLE_H

#include <basic_types.h>

#include <boost/noncopyable.hpp>
#include <boost/operators.hpp>
#include <utility>
#include <vector>

namespace DynamicRank {
class IFeatureMap;
}

namespace FreeForm2 {
class Expression;
class SimpleExpressionOwner;

class SymbolTable : boost::noncopyable {
 public:
  // Symbol represents a symbol string, which is hashed for
  // faster linear search.
  class Symbol : boost::equality_comparable<Symbol> {
   public:
    Symbol(SIZED_STRING p_str);
    Symbol(SIZED_STRING p_str, SIZED_STRING p_parameter);

    // Comparison operator.
    bool operator==(const Symbol &p_other) const;

    // Get the name of this symbol.
    SIZED_STRING GetSymbolName() const;

    // Whether this feature is parameterized.
    bool IsParameterized() const;

    // Get the parametrization string of this symbol.
    // This function may only be called if the symbol is parameterized.
    SIZED_STRING GetSymbolParameter() const;

    // Get a std::string representation of this symbol.
    std::string ToString() const;

   private:
    // Symbol string.
    SIZED_STRING m_str;

    // Hash of symbol string.
    UInt64 m_hash;

    // Whether this feature is parameterized.
    bool m_isParameterized;

    // Parameter string.
    SIZED_STRING m_param;

    // Hash of symbol string.
    UInt64 m_paramHash;
  };

  // Create a symbol table, that can (optionally) refer to an
  // underlying feature map to supply
  SymbolTable(SimpleExpressionOwner &p_owner,
              DynamicRank::IFeatureMap *p_featureMap);

  // Bind a symbol to an expression.
  void Bind(const Symbol &p_symbol, const Expression *p_expr);

  // Unbind the top symbol.
  std::pair<Symbol, const Expression *> Unbind();

  // Unbind the top symbol. It must match the p_symbol
  // parameter.
  std::pair<Symbol, const Expression *> Unbind(const Symbol &p_symbol);

  // Look up a string in the symbol table.
  const Expression &Lookup(const Symbol &p_symbol) const;

  // Test if a name has been bound.
  bool IsBound(const Symbol &p_symbol) const;

  // Get the number of local symbols currently bound.
  size_t GetNumLocal() const;

  // Turn on or off feature symbols.
  void SetAllowFeatures(bool p_allowFeatures);
  bool GetAllowFeatures() const;

 private:
  // Find a name in the local stack. Return NULL if p_str does not name
  // a local.
  const Expression *FindLocal(const Symbol &p_str) const;

  // Find a feature index by name. Returns true if p_str names a feature
  // and features are enabled; otherwise, returns false.
  bool FindFeatureIndex(SIZED_STRING p_str, UInt32 &p_index) const;

  // Stack of local variables, in order of creation.  Since this
  // number is expected to be fairly low, we linearly search
  // these variables.
  typedef std::vector<std::pair<Symbol, const Expression *> > LocalStack;
  LocalStack m_localStack;

  // Offset at which bound symbols start in m_localStack.
  size_t m_localStackStart;

  // Owner of expressions produced by this table.
  SimpleExpressionOwner &m_owner;

  // Feature map which is searched after local variables.
  DynamicRank::IFeatureMap *m_featureMap;

  // Flag to indicate whether features should be looked up in the
  // feature map.
  bool m_allowFeatures;
};

std::ostream &operator<<(std::ostream &p_out,
                         const SymbolTable::Symbol &p_symbol);
};  // namespace FreeForm2

#endif
