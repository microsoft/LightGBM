/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include "SymbolTable.h"

#include <IFeatureMap.h>

#include <sstream>

#include "CsHash.h"
#include "FreeForm2Assert.h"
#include "FreeForm2Utils.h"
#include "LiteralExpression.h"
#include "RefExpression.h"
#include "SimpleExpressionOwner.h"

namespace {
static const FreeForm2::LiteralBoolExpression c_falseExpression(
    FreeForm2::Annotations(), false);
static const FreeForm2::LiteralBoolExpression c_trueExpression(
    FreeForm2::Annotations(), true);
}  // namespace

FreeForm2::SymbolTable::SymbolTable(SimpleExpressionOwner &p_owner,
                                    DynamicRank::IFeatureMap *p_featureMap)
    : m_owner(p_owner),
      m_featureMap(p_featureMap),
      m_localStackStart(0),
      m_allowFeatures(true) {
  // Push true and false symbols onto stack.
  static const char *c_false = "false";
  Bind(FreeForm2::SymbolTable::Symbol(CStackSizedString(c_false)),
       &c_falseExpression);

  static const char *c_true = "true";
  Bind(FreeForm2::SymbolTable::Symbol(CStackSizedString(c_true)),
       &c_trueExpression);

  // Measure the size of the local stack, to ignore effects of pushing on
  // 'system' symbols.
  m_localStackStart = m_localStack.size();
}

void FreeForm2::SymbolTable::Bind(const Symbol &p_symbol,
                                  const Expression *p_expr) {
  if (IsSimpleName(p_symbol.GetSymbolName())) {
    m_localStack.push_back(std::make_pair(p_symbol, p_expr));
  } else {
    // Prevent binding of odd names, consisting of punctuation.  This allows
    // us to change tokenisation of punctuation in the future without
    // breaking backward compatibility (because we don't have to worry about
    // how the tokenisation of '$%)*%$[]' will occur, and we know the pool
    // of names bound by primitive operations).
    std::ostringstream err;
    err << "Failed to bind name '" << p_symbol
        << "'.  Bound names can only contain alphanumeric characters, hyphens, "
           "and underscores.";
    throw ParseError(err.str(), p_expr->GetSourceLocation());
  }
}

std::pair<FreeForm2::SymbolTable::Symbol, const FreeForm2::Expression *>
FreeForm2::SymbolTable::Unbind() {
  FF2_ASSERT(!m_localStack.empty());

  SIZED_STRING name = m_localStack.back().first.GetSymbolName();
  const Expression *expr = m_localStack.back().second;

  m_localStack.pop_back();
  return std::make_pair(name, expr);
}

std::pair<FreeForm2::SymbolTable::Symbol, const FreeForm2::Expression *>
FreeForm2::SymbolTable::Unbind(const Symbol &p_symbol) {
  FF2_ASSERT(!m_localStack.empty());
  FF2_ASSERT(m_localStack.back().first == p_symbol);

  return Unbind();
}

const FreeForm2::Expression *FreeForm2::SymbolTable::FindLocal(
    const Symbol &p_symbol) const {
  // Search through the stack of local variables, from most recent
  // to least recent.
  for (LocalStack::const_reverse_iterator iter = m_localStack.rbegin();
       iter != m_localStack.rend(); ++iter) {
    if (iter->first == p_symbol) {
      return iter->second;
    }
  }
  return NULL;
}

bool FreeForm2::SymbolTable::FindFeatureIndex(SIZED_STRING p_str,
                                              UInt32 &p_index) const {
  if (m_featureMap != NULL && m_allowFeatures) {
    if (m_featureMap->ObtainFeatureIndex(p_str, p_index)) {
      return true;
    }
  }
  return false;
}

const FreeForm2::Expression &FreeForm2::SymbolTable::Lookup(
    const Symbol &p_symbol) const {
  const Expression *expr = FindLocal(p_symbol);
  if (expr != NULL) {
    return *expr;
  }

  // Failed to find symbol in local stack, check feature map.
  UInt32 index = 0;
  if (FindFeatureIndex(p_symbol.GetSymbolName(), index)) {
    // Found it in feature map, create an expression for it.
    boost::shared_ptr<Expression> exp(
        new FeatureRefExpression(Annotations(), index));
    m_owner.AddExpression(exp);
    return *exp.get();
  }

  // Failed to resolve symbol.
  std::ostringstream err;
  err << "Failed to find '" << p_symbol << "' in local variables and features.";
  throw std::runtime_error(err.str());
}

bool FreeForm2::SymbolTable::IsBound(const Symbol &p_symbol) const {
  UInt32 index;
  return (FindLocal(p_symbol) != NULL ||
          FindFeatureIndex(p_symbol.GetSymbolName(), index));
}

size_t FreeForm2::SymbolTable::GetNumLocal() const {
  return m_localStack.size() - m_localStackStart;
}

void FreeForm2::SymbolTable::SetAllowFeatures(bool p_allowFeatures) {
  m_allowFeatures = p_allowFeatures;
}

bool FreeForm2::SymbolTable::GetAllowFeatures() const {
  return m_allowFeatures;
}

FreeForm2::SymbolTable::Symbol::Symbol(SIZED_STRING p_str)
    : m_str(p_str),
      m_hash(CsHash64::Compute(p_str.pcData, p_str.cbData)),
      m_paramHash(0),
      m_isParameterized(false) {}

FreeForm2::SymbolTable::Symbol::Symbol(SIZED_STRING p_str, SIZED_STRING p_param)
    : m_str(p_str),
      m_param(p_param),
      m_isParameterized(true),
      m_hash(CsHash64::Compute(p_str.pcData, p_str.cbData)),
      m_paramHash(CsHash64::Compute(p_param.pcData, p_param.cbData)) {}

bool FreeForm2::SymbolTable::Symbol::operator==(const Symbol &p_other) const {
  return m_isParameterized == p_other.m_isParameterized &&
         m_hash == p_other.m_hash && m_paramHash == p_other.m_paramHash &&
         m_str.cbData == p_other.m_str.cbData &&
         memcmp(m_str.pcData, p_other.m_str.pcData, m_str.cbData) == 0 &&
         (m_isParameterized ? m_param.cbData == p_other.m_param.cbData &&
                                  memcmp(m_param.pcData, p_other.m_param.pcData,
                                         m_param.cbData) == 0
                            : true);
}

SIZED_STRING
FreeForm2::SymbolTable::Symbol::GetSymbolName() const { return m_str; }

bool FreeForm2::SymbolTable::Symbol::IsParameterized() const {
  return m_isParameterized;
}

SIZED_STRING
FreeForm2::SymbolTable::Symbol::GetSymbolParameter() const {
  FF2_ASSERT(m_isParameterized);
  return m_param;
}

std::string FreeForm2::SymbolTable::Symbol::ToString() const {
  std::ostringstream buffer;
  buffer << *this;
  return buffer.str();
}

std::ostream &FreeForm2::operator<<(
    std::ostream &p_out, const FreeForm2::SymbolTable::Symbol &p_symbol) {
  p_out << p_symbol.GetSymbolName();
  if (p_symbol.IsParameterized()) {
    p_out << "<" << p_symbol.GetSymbolParameter() << ">";
  }
  return p_out;
}
