/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include "ProgramParseState.h"

#include <sstream>

#include "ConvertExpression.h"
#include "ExpressionFactory.h"
#include "FreeForm2Assert.h"
#include "FreeForm2Utils.h"
#include "FunctionInlineVisitor.h"
#include "MiscFactory.h"
#include "TypeManager.h"

FreeForm2::ProgramParseState::ExpressionParseState::ExpressionParseState(
    const ExpressionFactory &p_factory, SIZED_STRING p_atom,
    unsigned int p_offset)
    : m_factory(&p_factory), m_atom(p_atom), m_offset(p_offset) {}

void FreeForm2::ProgramParseState::ExpressionParseState::Add(
    const Expression &p_expr) {
  m_children.push_back(&p_expr);
}

const FreeForm2::Expression &
FreeForm2::ProgramParseState::ExpressionParseState::Finish(
    SimpleExpressionOwner &p_owner, TypeManager &p_typeManager) const {
  const FreeForm2::Expression &ret =
      m_factory->Create(*this, p_owner, p_typeManager);

  return ret;
}

FreeForm2::ProgramParseState::ProgramParseState(
    SIZED_STRING p_input, DynamicRank::IFeatureMap &p_map,
    const OpMap &p_operators, bool p_mustProduceFloat,
    bool p_parsingAggregatedExpression)
    : m_owner(new SimpleExpressionOwner()),
      m_typeManager(TypeManager::CreateTypeManager().release()),
      m_tokenizer(p_input),
      m_symbols(*m_owner, &p_map),
      m_operators(p_operators),
      m_parsingLambdaBody(false),
      m_parsingAggregatedExpression(p_parsingAggregatedExpression) {
  // If the expression must be a float, then we create a base factory
  // to handle that.
  const ExpressionFactory *rootFactory =
      &GetFeatureSpecInstance(p_mustProduceFloat);
  ExpressionParseState initialState(*rootFactory,
                                    CStackSizedString("<initial>"), 0);
  m_parseStack.push_back(initialState);

  m_variableIdCounter.m_value = 0;
}

FreeForm2::SExpressionParse::ParserResults
FreeForm2::ProgramParseState::Finish() {
  // An expression has now been parsed.  We need to check a few things
  // to ensure that it's valid.

  if (m_parseStack.size() > 1) {
    // Left expression(s) open.
    std::ostringstream err;
    err << "After parsing expression, " << m_parseStack.size() - 1
        << " expressions remain open.";
    throw std::runtime_error(err.str());
  }

  Token tok = m_tokenizer.Advance();
  if (tok != TOKEN_END) {
    // Trailing junk after expression.
    std::ostringstream err;
    err << "Trailing " << Tokenizer::TokenName(tok)
        << " token found after expression.";
    throw std::runtime_error(err.str());
  }

  // Use the FunctionInlineVisitor to replace FunctionCallExpressions with
  // LetExpressions. This needs to be done now since the parsed expression is
  // then wrapped with a FeatureSpecExpression that uses the type of the parsed
  // expression.
  boost::shared_ptr<SimpleExpressionOwner> functionVisitorOwner(
      new SimpleExpressionOwner());
  boost::shared_ptr<TypeManager> functionVisitorTypeManager(
      TypeManager::CreateTypeManager().release());
  FunctionInlineVisitor functionInlineVisitor(
      functionVisitorOwner, functionVisitorTypeManager, GetNextVariableId());

  FF2_ASSERT(m_parseStack.back().m_children.size() == 1);
  const Expression *syntaxTree = m_parseStack.back().m_children.back();
  m_parseStack.back().m_children.pop_back();
  syntaxTree->Accept(functionInlineVisitor);
  syntaxTree = functionInlineVisitor.GetSyntaxTree();
  m_parseStack.back().m_children.push_back(syntaxTree);

  m_owner.swap(functionVisitorOwner);
  m_typeManager.swap(functionVisitorTypeManager);

  // The FunctionInlineVisitor may assign variable ids to ensure unique variable
  // ids for each function call. Update the id of the ProgramParseState to
  // ensure unique variable ids.
  m_variableIdCounter = functionInlineVisitor.GetVariableId();

  FF2_ASSERT(m_parseStack.size() == 1);
  const Expression &root = m_parseStack.back().Finish(*m_owner, *m_typeManager);

  m_parseStack.pop_back();
  FF2_ASSERT(m_parseStack.empty());
  return boost::tuples::make_tuple(
      &root, boost::shared_ptr<SimpleExpressionOwner>(m_owner),
      boost::shared_ptr<TypeManager>(m_typeManager));
}

FreeForm2::VariableID FreeForm2::ProgramParseState::GetNextVariableId() {
  VariableID id = m_variableIdCounter;
  ++m_variableIdCounter.m_value;
  return id;
}

const FreeForm2::Expression &FreeForm2::ProgramParseState::GetLastParsed()
    const {
  return *m_parseStack.back().m_children.back();
}

void FreeForm2::ProgramParseState::CloseExpression() {
  const Expression &finished =
      m_parseStack.back().Finish(*m_owner, *m_typeManager);
  m_parseStack.pop_back();

  if (m_parseStack.empty()) {
    // Too many close parens.
    std::ostringstream err;
    err << "Mismatched close parenthesis";
    throw std::runtime_error(err.str());
  }

  m_parseStack.back().Add(finished);
}
