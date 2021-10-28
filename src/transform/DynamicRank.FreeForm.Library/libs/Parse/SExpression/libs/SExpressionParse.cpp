/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include "SExpressionParse.h"

#include "Arithmetic.h"
#include "ArrayLength.h"
#include "ArrayLiteralExpression.h"
#include "ArrayDereferenceExpression.h"
#include "Bitwise.h"
#include "Conditional.h"
#include "ConvertExpression.h"
#include "ExpressionFactory.h"
#include "FreeForm2Assert.h"
#include "FreeForm2Tokenizer.h"
#include "FreeForm2Utils.h"
#include "Function.h"
#include "FunctionType.h"
#include "LetExpression.h"
#include "LiteralExpression.h"
#include "Logic.h"
#include "MiscFactory.h"
#include "ProgramParseState.h"
#include "RangeReduceExpression.h"
#include "RefExpression.h"
#include "SelectNth.h"
#include "SimpleExpressionOwner.h"
#include "SymbolTable.h"
#include "TypeManager.h"
#include "TypeUtil.h"

#include <IFeatureMap.h>

#include <cmath>
#include <boost/cast.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/make_shared.hpp>
#include <boost/variant.hpp>
#include <list>
#include <map>
#include <sstream>
#include <stdexcept>
#include <functional>

using namespace FreeForm2;

namespace
{
    boost::shared_ptr<Expression> ParseLiteralInt(const Annotations &p_annotations,
                                                  SIZED_STRING p_value)
    {
        std::string source(SIZED_STR(p_value));
        boost::shared_ptr<Expression> expr(
            boost::make_shared<LiteralIntExpression>(p_annotations,
                                                     boost::lexical_cast<Result::IntType>(source)));
        return expr;
    }

    boost::shared_ptr<Expression> ParseLiteralFloat(const Annotations &p_annotations,
                                                    SIZED_STRING p_value)
    {
        std::string source(SIZED_STR(p_value));
        boost::shared_ptr<Expression> expr(
            boost::make_shared<LiteralFloatExpression>(p_annotations,
                                                       boost::lexical_cast<Result::FloatType>(source)));
        return expr;
    }

    class StaticOperatorMap : public ProgramParseState::OpMap
    {
    public:
        StaticOperatorMap()
        {
            (*this)["+"] = ProgramParseState::OperatorInfo(&Arithmetic::GetPlusInstance());
            (*this)["-"] = ProgramParseState::OperatorInfo(&Arithmetic::GetMinusInstance());
            (*this)["*"] = ProgramParseState::OperatorInfo(&Arithmetic::GetMultiplyInstance());
            (*this)["/"] = ProgramParseState::OperatorInfo(&Arithmetic::GetDividesInstance());
            (*this)["trunc-div"] = ProgramParseState::OperatorInfo(&Arithmetic::GetIntegerDivInstance());
            (*this)["trunc-mod"] = ProgramParseState::OperatorInfo(&Arithmetic::GetIntegerModInstance());
            (*this)["mod"] = ProgramParseState::OperatorInfo(&Arithmetic::GetModInstance());
            (*this)["max"] = ProgramParseState::OperatorInfo(&Arithmetic::GetMaxInstance());
            (*this)["min"] = ProgramParseState::OperatorInfo(&Arithmetic::GetMinInstance());
            (*this)["**"] = ProgramParseState::OperatorInfo(&Arithmetic::GetPowInstance());
            (*this)["^"] = ProgramParseState::OperatorInfo(&Arithmetic::GetPowInstance());
            (*this)["ln"] = ProgramParseState::OperatorInfo(&Arithmetic::GetUnaryLogInstance());
            (*this)["log"] = ProgramParseState::OperatorInfo(&Arithmetic::GetBinaryLogInstance());
            (*this)["ln1"] = ProgramParseState::OperatorInfo(&Arithmetic::GetLog1Instance());
            (*this)["abs"] = ProgramParseState::OperatorInfo(&Arithmetic::GetAbsInstance());
            (*this)["truncate"] = ProgramParseState::OperatorInfo(&Arithmetic::GetTruncInstance());
            (*this)["round"] = ProgramParseState::OperatorInfo(&Arithmetic::GetRoundInstance());
            (*this)["float"] = ProgramParseState::OperatorInfo(&Convert::GetFloatConvertFactory());
            (*this)["int"] = ProgramParseState::OperatorInfo(&Convert::GetIntConvertFactory());
            (*this)["bool"] = ProgramParseState::OperatorInfo(&Convert::GetBoolConversionFactory());

            (*this)["=="] = ProgramParseState::OperatorInfo(&Logic::GetCmpEqInstance());
            (*this)["!="] = ProgramParseState::OperatorInfo(&Logic::GetCmpNotEqInstance());
            (*this)["<"] = ProgramParseState::OperatorInfo(&Logic::GetCmpLTInstance());
            (*this)["<="] = ProgramParseState::OperatorInfo(&Logic::GetCmpLTEInstance());
            (*this)[">"] = ProgramParseState::OperatorInfo(&Logic::GetCmpGTInstance());
            (*this)[">="] = ProgramParseState::OperatorInfo(&Logic::GetCmpGTEInstance());

            (*this)["and"] = ProgramParseState::OperatorInfo(&Logic::GetAndInstance());
            (*this)["&&"] = ProgramParseState::OperatorInfo(&Logic::GetAndInstance());
            (*this)["or"] = ProgramParseState::OperatorInfo(&Logic::GetOrInstance());
            (*this)["||"] = ProgramParseState::OperatorInfo(&Logic::GetOrInstance());
            (*this)["not"] = ProgramParseState::OperatorInfo(&Logic::GetNotInstance());

            (*this)["bitand"] = ProgramParseState::OperatorInfo(&Bitwise::GetAndInstance());
            (*this)["bitor"] = ProgramParseState::OperatorInfo(&Bitwise::GetOrInstance());
            (*this)["bitnot"] = ProgramParseState::OperatorInfo(&Bitwise::GetNotInstance());

            (*this)["if"] = ProgramParseState::OperatorInfo(&Conditional::GetIfInstance());
            (*this)["select-nth"] = ProgramParseState::OperatorInfo(&Select::GetSelectNthInstance());
            (*this)["select-range"] = ProgramParseState::OperatorInfo(&Select::GetSelectRangeInstance());

            (*this)["let"] = ProgramParseState::OperatorInfo(&SExpressionParse::ParseLet);
            (*this)["macro-let"] = ProgramParseState::OperatorInfo(&SExpressionParse::ParseMacroLet);
            (*this)["range-reduce"] = ProgramParseState::OperatorInfo(&SExpressionParse::ParseRangeReduce);
            (*this)["array-literal"] = ProgramParseState::OperatorInfo(&SExpressionParse::ParseArrayLiteral);
            (*this)["array-length"] = ProgramParseState::OperatorInfo(&GetArrayLengthInstance());
            (*this)["random-float"] = ProgramParseState::OperatorInfo(&Random::GetRandomFloatInstance());
            (*this)["random-int"] = ProgramParseState::OperatorInfo(&Random::GetRandomIntInstance());
            (*this)["lambda"] = ProgramParseState::OperatorInfo(&SExpressionParse::ParseLambda);
            (*this)["invoke"] = ProgramParseState::OperatorInfo(&SExpressionParse::ParseInvoke);
        }
    };

    static const StaticOperatorMap c_operators;

    // Indicates whether the current parse stack has hit the expression limit,
    // (in which case parsing should return to caller for
    bool HitExpressionLimit(const std::list<ProgramParseState::ExpressionParseState> &p_stack,
                            size_t p_depth,
                            size_t p_limit)
    {
        return p_stack.size() == p_depth && (p_stack.back().m_children.size() >= p_limit);
    }

    class ArrayDereferenceFactory : public ExpressionFactory
    {
    private:
        virtual const Expression &
        CreateExpression(const ProgramParseState::ExpressionParseState &p_state,
                         SimpleExpressionOwner &p_owner,
                         TypeManager &p_typeManager) const override
        {
            boost::shared_ptr<Expression> ptr(boost::make_shared<ArrayDereferenceExpression>(
                Annotations(SourceLocation(1, p_state.m_offset)),
                *p_state.m_children[0],
                *p_state.m_children[1],
                0));
            p_owner.AddExpression(ptr);
            return *ptr;
        }

        virtual std::pair<unsigned int, unsigned int> Arity() const override
        {
            return std::make_pair(2, 2);
        }
    };

    // Global instance of the array dereference expression factory.
    const static ArrayDereferenceFactory c_arrayDereferenceFactory;

    // Factory to create recursively nested array literal expressions.
    class ArrayLiteralExpressionFactory : public ExpressionFactory
    {
    private:
        virtual const Expression &
        CreateExpression(const ProgramParseState::ExpressionParseState &p_state,
                         SimpleExpressionOwner &p_owner,
                         TypeManager &p_typeManager) const override
        {
            FF2_ASSERT(p_state.m_variableIds.size() == 1);
            boost::shared_ptr<Expression> ptr(ArrayLiteralExpression::Alloc(Annotations(SourceLocation(1, p_state.m_offset)),
                                                                            TypeImpl::GetUnknownType(),
                                                                            p_state.m_children,
                                                                            p_state.m_variableIds[0],
                                                                            p_typeManager));
            p_owner.AddExpression(ptr);
            return *ptr;
        }

        virtual std::pair<unsigned int, unsigned int> Arity() const override
        {
            return std::make_pair(0, UINT_MAX);
        }
    };

    const static ArrayLiteralExpressionFactory c_arrayLiteralFactory;

    class LetExpressionFactory : public ExpressionFactory
    {
    private:
        virtual const Expression &
        CreateExpression(const ProgramParseState::ExpressionParseState &p_state,
                         SimpleExpressionOwner &p_owner,
                         TypeManager &p_typeManager) const override
        {
            // Count the number of non-function parameters that there are in the let statement.
            int numNonFunctionParameters = 0;

            std::vector<LetExpression::IdExpressionPair> children;
            for (size_t i = 0; i < p_state.m_variableIds.size(); ++i)
            {
                // Do not create a binding for lambdas.
                if (p_state.m_children[i]->GetType().Primitive() != Type::Function)
                {
                    children.push_back(std::make_pair(p_state.m_variableIds[i], p_state.m_children[i]));
                    ++numNonFunctionParameters;
                }
            }

            FF2_ASSERT(p_state.m_variableIds.size() == numNonFunctionParameters);

            if (children.size() > 0)
            {
                boost::shared_ptr<Expression> expr = LetExpression::Alloc(Annotations(SourceLocation(1, p_state.m_offset)),
                                                                          children,
                                                                          p_state.m_children.back());
                p_owner.AddExpression(expr);
                return *expr;
            }
            else
            {
                // If all let bindings were lambdas, do not produce a let
                // expression; no variable bindings are needed.
                return *p_state.m_children.back();
            }
        }

        virtual std::pair<unsigned int, unsigned int> Arity() const override
        {
            return std::make_pair(1, UINT_MAX);
        }
    };

    // Global instance of the let expression factory.
    const static LetExpressionFactory c_letFactory;

    class RangeReduceExpressionFactory : public ExpressionFactory
    {
    private:
        virtual const Expression &
        CreateExpression(const ProgramParseState::ExpressionParseState &p_state,
                         SimpleExpressionOwner &p_owner,
                         TypeManager &p_typeManager) const override
        {
            boost::shared_ptr<Expression> one(new LiteralIntExpression(Annotations(SourceLocation(1, p_state.m_offset)),
                                                                       1));
            p_owner.AddExpression(one);

            FF2_ASSERT(p_state.m_variableIds.size() == 2);
            const ChildVec &children = p_state.m_children;
            boost::shared_ptr<Expression> ptr(
                boost::make_shared<RangeReduceExpression>(Annotations(SourceLocation(1, p_state.m_offset)),
                                                          *children[0],
                                                          *children[1],
                                                          *children[2],
                                                          *children[3],
                                                          p_state.m_variableIds[0],
                                                          p_state.m_variableIds[1]));

            p_owner.AddExpression(ptr);
            return *ptr;
        }

        virtual std::pair<unsigned int, unsigned int> Arity() const override
        {
            return std::make_pair(4, 4);
        }
    };

    // Global instance of the rangereduce expression factory.
    const static RangeReduceExpressionFactory c_rangeFactory;

    class LambdaExpressionFactory : public ExpressionFactory
    {
    private:
        virtual const Expression &
        CreateExpression(const ProgramParseState::ExpressionParseState &p_state,
                         SimpleExpressionOwner &p_owner,
                         TypeManager &p_typeManager) const override
        {
            // Children of lambda ExpressionParseState are:
            // All except last: Parameters of lambda expression.
            // Last: The body of the lambda expression.
            FF2_ASSERT(p_state.m_variableIds.size() == p_state.m_children.size() - 1);
            std::vector<FunctionExpression::Parameter> params(p_state.m_variableIds.size());
            std::vector<const TypeImpl *> paramTypes(p_state.m_variableIds.size());
            for (size_t i = 0; i < p_state.m_variableIds.size(); i++)
            {
                params[i].m_parameter = boost::polymorphic_downcast<const VariableRefExpression *>(p_state.m_children[i]);
                params[i].m_isFeatureParameter = false;
                paramTypes[i] = &params[i].m_parameter->GetType();
            }
            const Expression &body = *p_state.m_children.back();
            const FunctionType &type = p_typeManager.GetFunctionType(body.GetType(), paramTypes.data(), paramTypes.size());

            // The name of the lambda is unimportant to evaluation, but we will name it anyways.
            static int id = 1;
            std::ostringstream lambdaNameStream;
            lambdaNameStream << "lambda<" << id++ << ">";
            std::string lambdaName = lambdaNameStream.str();

            boost::shared_ptr<Expression> expr = boost::make_shared<FunctionExpression>(Annotations(SourceLocation(1, p_state.m_offset)),
                                                                                        type,
                                                                                        lambdaName,
                                                                                        params,
                                                                                        body);
            p_owner.AddExpression(expr);
            return *expr;
        }

        virtual std::pair<unsigned int, unsigned int> Arity() const override
        {
            return std::make_pair(1, UINT_MAX);
        }
    };

    // Global instance of the lambda expression factory.
    const static LambdaExpressionFactory c_lambdaFactory;

    class InvokeFactory : public ExpressionFactory
    {
    public:
        InvokeFactory()
        {
        }

    private:
        virtual const Expression &
        CreateExpression(const ProgramParseState::ExpressionParseState &p_state,
                         SimpleExpressionOwner &p_owner,
                         TypeManager &p_typeManager) const override
        {
            // The children of the inoke ExpressionParseState should be:
            // 1st child: Function expression
            // All others: Parameters in the invoke expression. (Must have at least one)
            FF2_ASSERT(p_state.m_children.size() >= 2);

            if (p_state.m_children[0]->GetType().Primitive() != Type::Function)
            {
                std::ostringstream err;
                err << "Invoke may only be called on a lambda or "
                    << "an expression bound to a lambda "
                    << "(called on expression of type "
                    << p_state.m_children[0]->GetType()
                    << ")";
                throw std::runtime_error(err.str());
            }

            const FunctionType &type = static_cast<const FunctionType &>(p_state.m_children[0]->GetType());
            if (type.GetParameterCount() != p_state.m_children.size() - 1)
            {
                std::ostringstream err;
                err << "Parameter count mismatch: expected "
                    << type.GetParameterCount() << " parameters, "
                    << "got " << p_state.m_children.size() - 1;
                throw std::runtime_error(err.str());
            }

            std::vector<const Expression *> params(p_state.m_children.size() - 1);
            for (size_t i = 1; i < p_state.m_children.size(); i++)
            {
                const Expression &param = *p_state.m_children[i];

                // Since the lambda is the first child in the Invoke expression,
                // subtract 1 from the child index to get the correct parameter.
                const TypeImpl &expected = *type.BeginParameters()[i - 1];

                // Check that parameters with explicit types in the lambda
                // definition are correctly specified.
                if (!TypeUtil::IsAssignable(expected, param.GetType()))
                {
                    // Unknown types should always be valid assignment destinations.
                    FF2_ASSERT(expected.Primitive() != Type::Unknown);
                    std::ostringstream err;
                    err << "Parameter type mismatch: expected type "
                        << expected << ", "
                        << "got " << param.GetType();
                    throw std::runtime_error(err.str());
                }
                params[i - 1] = &param;
            }

            boost::shared_ptr<Expression> expr(
                FunctionCallExpression::Alloc(Annotations(SourceLocation(1, p_state.m_offset)),
                                              *p_state.m_children[0],
                                              params));
            p_owner.AddExpression(expr);
            return *expr;
        }

        virtual std::pair<unsigned int, unsigned int> Arity() const override
        {
            // Invoke must have a function and at least one parameter.
            return std::make_pair(2, MAX_UINT32);
        }
    };

    // Global instance of the invoke factory.
    static const InvokeFactory c_invokeFactory;

    // Continue advancing the tokenizer until a matched closing state is found.
    // This method matches nested open-close pairs correctly.
    FreeForm2::Token ParseUntilClosed(FreeForm2::ProgramParseState &p_state,
                                      FreeForm2::Token p_open,
                                      FreeForm2::Token p_close)
    {
        // Read tokens until we have a matched open- and close-parentheses.
        size_t depth = 1;
        FreeForm2::Token tok = p_open;
        while (depth > 0 && tok != TOKEN_END)
        {
            tok = p_state.m_tokenizer.Advance();
            if (tok == p_open)
            {
                ++depth;
            }
            else if (tok == p_close)
            {
                --depth;
            }
        }

        return tok;
    }
};

FreeForm2::SExpressionParse::ParserResults
FreeForm2::SExpressionParse::Parse(SIZED_STRING p_input,
                                   DynamicRank::IFeatureMap &p_map,
                                   bool p_mustProduceFloat,
                                   bool p_parsingAggregatedExpression)
{
    ProgramParseState parseState(p_input, p_map, c_operators, p_mustProduceFloat, p_parsingAggregatedExpression);
    SExpressionParse::ParseTokens(parseState, 0);

    return parseState.Finish();
}

FreeForm2::Token
FreeForm2::SExpressionParse::ParseTokens(ProgramParseState &p_state,
                                         unsigned int p_expressionLimit)
{
    // Keep track of the depth of the parse stack at entry.  This allows us
    // to use this function recursively in a sensible fashion, by having it
    // exit once we close back to this depth.
    const size_t parseDepth = p_state.m_parseStack.size();
    const size_t limit = p_expressionLimit == 0
                             ? static_cast<size_t>(-1)
                             : p_state.m_parseStack.back().m_children.size() + p_expressionLimit;

    Token tok = p_state.m_tokenizer.GetToken();
    while (tok != TOKEN_END)
    {
        FF2_ASSERT(!p_state.m_parseStack.empty());

        switch (tok)
        {
        case TOKEN_OPEN:
        {
            tok = p_state.m_tokenizer.Advance();

            if (tok != TOKEN_ATOM)
            {
                std::ostringstream err;
                err << "Expected atom (a name) after open parenthesis, "
                    << "got something else ("
                    << Tokenizer::TokenName(tok) << ")";
                throw std::runtime_error(err.str());
            }

            std::string atom(SIZED_STR(p_state.m_tokenizer.GetValue()));
            ProgramParseState::OpMap::const_iterator iter = p_state.m_operators.find(atom);

            if (iter == p_state.m_operators.end())
            {
                // Couldn't find operator.
                std::ostringstream err;
                err << "Failed to find operator '" << atom << "'.";
                throw std::runtime_error(err.str());
            }

            if (boost::get<ProgramParseState::ParsingFunction>(&iter->second) == NULL)
            {
                // Continue parsing this operator.
                const ExpressionFactory *const *factory = boost::get<const ExpressionFactory *>(&iter->second);
                FF2_ASSERT(factory != NULL && *factory != NULL);
                ProgramParseState::ExpressionParseState
                    state(**factory, p_state.m_tokenizer.GetValue(), p_state.m_tokenizer.GetPosition());
                p_state.m_parseStack.push_back(state);

                // Advance to next token.
                tok = p_state.m_tokenizer.Advance();
            }
            else
            {
                // Handle special form.
                ProgramParseState::ParsingFunction const *specialForm = boost::get<ProgramParseState::ParsingFunction>(&iter->second);
                FF2_ASSERT(specialForm != NULL && *specialForm != NULL);
                tok = (*specialForm)(p_state);
                FF2_ASSERT(tok == TOKEN_CLOSE || tok == TOKEN_END);
            }

            // Continue back up to the top of the loop, rather than going
            // through the processing that follows other tokens.
            continue;
        }

        case TOKEN_CLOSE:
        {
            p_state.CloseExpression();

            if (p_state.m_parseStack.size() < parseDepth)
            {
                // Have closed the expression we began parsing, consume
                // token and return.
                return p_state.m_tokenizer.Advance();
            }
            break;
        }

        case TOKEN_ATOM:
        {
            // Look up name in symbol table.
            p_state.m_parseStack.back().Add(p_state.m_symbols.Lookup(
                SymbolTable::Symbol(p_state.m_tokenizer.GetValue())));
            break;
        }

        case TOKEN_INT:
        {
            boost::shared_ptr<Expression> expr = ParseLiteralInt(Annotations(SourceLocation(1, p_state.m_tokenizer.GetPosition())),
                                                                 p_state.m_tokenizer.GetValue());
            p_state.m_owner->AddExpression(expr);
            p_state.m_parseStack.back().Add(*expr);
            break;
        }

        case TOKEN_FLOAT:
        {
            boost::shared_ptr<Expression> expr = ParseLiteralFloat(Annotations(SourceLocation(1, p_state.m_tokenizer.GetPosition())),
                                                                   p_state.m_tokenizer.GetValue());
            p_state.m_owner->AddExpression(expr);
            p_state.m_parseStack.back().Add(*expr);
            break;
        }

        case TOKEN_OPEN_ARRAY:
        {
            // Parse the array dereference expression.  Note that we can't
            // handle this through the generic special form method, because
            // they all start with '(', and this doesn't.
            tok = ParseArrayDereference(p_state);
            break;
        }

        default:
        {
            std::ostringstream err;
            err << "Unexpected token type '" << tok << "'";
            throw std::runtime_error(err.str());
            break;
        }
        };

        // Advance to next token.
        tok = p_state.m_tokenizer.Advance();

        // We check to see if the next token is an open array, because it adds
        // to the last expression (no other construct currently does), instead
        // of creating more expressions.  Thus, the expression limit hasn't
        // really been hit if the next token is an open array.
        if (tok != TOKEN_OPEN_ARRAY && HitExpressionLimit(p_state.m_parseStack, parseDepth, limit))
        {
            // Have closed the expression we began parsing, or ran into
            // a limit on the number of sub-expressions, end now.
            return tok;
        }
    }

    return tok;
}

FreeForm2::Token
FreeForm2::SExpressionParse::ParseArrayDereference(ProgramParseState &p_state)
{
    if (p_state.m_parseStack.back().m_children.empty())
    {
        // User tried to dereference without providing an expression
        // to dereference.
        std::ostringstream err;
        err << "Received " << Tokenizer::TokenName(p_state.m_tokenizer.GetToken())
            << " token, which starts an "
            << "array dereference, but there is nothing to dereference";
        throw std::runtime_error(err.str());
    }

    // Pop the last expression off of the stack.
    const Expression &last = p_state.GetLastParsed();
    p_state.m_parseStack.back().m_children.pop_back();

    // Create an array dereference factory, and push the last
    // expression parsed into it as the first expression.
    ProgramParseState::ExpressionParseState
        state(c_arrayDereferenceFactory, p_state.m_tokenizer.GetValue(), p_state.m_tokenizer.GetPosition());
    p_state.m_parseStack.push_back(state);
    p_state.m_parseStack.back().m_children.push_back(&last);

    // Parse index expression.
    p_state.m_tokenizer.Advance();
    Token tok = SExpressionParse::ParseTokens(p_state, 1);

    if (tok == TOKEN_END)
    {
        return tok;
    }
    else if (tok != TOKEN_CLOSE_ARRAY)
    {
        // User tried to dereference without providing an expression
        // to dereference.
        std::ostringstream err;
        err << "Expected a " << Tokenizer::TokenName(TOKEN_CLOSE_ARRAY)
            << " token after array dereference index, but got a "
            << Tokenizer::TokenName(tok) << " token instead.";
        throw std::runtime_error(err.str());
    }

    // Finish off the expression.
    p_state.CloseExpression();

    return tok;
}

const FreeForm2::ArrayLiteralExpression &
FreeForm2::SExpressionParse::ParseArrayLiteralRecurse(ProgramParseState &p_state)
{
    Token tok = p_state.m_tokenizer.GetToken();
    if (tok != TOKEN_OPEN_ARRAY)
    {
        std::ostringstream err;
        err << "Expected open array token, "
            << "got something else (" << Tokenizer::TokenName(tok) << " token).";
        throw std::runtime_error(err.str());
    }

    const size_t parseDepth = p_state.m_parseStack.size();
    ProgramParseState::ExpressionParseState
        arrayState(c_arrayLiteralFactory, p_state.m_tokenizer.GetValue(), p_state.m_tokenizer.GetPosition());
    arrayState.m_variableIds.push_back(p_state.GetNextVariableId());
    p_state.m_parseStack.push_back(arrayState);

    tok = p_state.m_tokenizer.Advance();
    while (tok != TOKEN_CLOSE_ARRAY)
    {
        if (tok == TOKEN_OPEN_ARRAY)
        {
            p_state.m_parseStack.back().Add(ParseArrayLiteralRecurse(p_state));
            tok = p_state.m_tokenizer.Advance();
        }
        else if (tok == TOKEN_END)
        {
            std::ostringstream err;
            err << "Unexpected end to program with array literal still open";
            throw std::runtime_error(err.str());
        }
        else
        {
            tok = SExpressionParse::ParseTokens(p_state, 1);
        }
    }

    FF2_ASSERT(p_state.m_tokenizer.GetToken() == TOKEN_CLOSE_ARRAY);
    const ArrayLiteralExpression *result = boost::polymorphic_downcast<const ArrayLiteralExpression *>(
        &p_state.m_parseStack.back().Finish(*p_state.m_owner, *p_state.m_typeManager));
    p_state.m_parseStack.pop_back();
    FF2_ASSERT(p_state.m_parseStack.size() == parseDepth);
    return *result;
}

FreeForm2::Token
FreeForm2::SExpressionParse::ParseArrayLiteral(ProgramParseState &p_state)
{
    // Advance past token that started this special form.
    p_state.m_tokenizer.Advance();

    const size_t parseDepth = p_state.m_parseStack.size();
    const ArrayLiteralExpression &array = ParseArrayLiteralRecurse(p_state);
    FF2_ASSERT(p_state.m_tokenizer.GetToken() == TOKEN_CLOSE_ARRAY);
    Token tok = p_state.m_tokenizer.Advance();
    const ArrayLiteralExpression *flat = NULL;
    if (tok == TOKEN_ATOM)
    {
        Type::TypePrimitive primitive = Type::ParsePrimitive(p_state.m_tokenizer.GetValue());
        if (primitive == Type::Invalid)
        {
            std::ostringstream err;
            err << "Couldn't parse name of array element type from '"
                << p_state.m_tokenizer.GetValue()
                << "'";
            throw std::runtime_error(err.str());
        }
        else if (!TypeImpl::IsLeafType(primitive))
        {
            std::ostringstream err;
            err << "Array elements must be of fixed size (such as int, float), "
                << "not " << Type::Name(primitive);
            throw std::runtime_error(err.str());
        }

        const TypeImpl &child = TypeImpl::GetCommonType(primitive, true);
        flat = &array.Flatten(*p_state.m_owner, &child, p_state.m_typeManager.get());

        tok = p_state.m_tokenizer.Advance();
    }
    else
    {
        flat = &array.Flatten(*p_state.m_owner);
    }
    FF2_ASSERT(flat != NULL);

    if (tok != TOKEN_CLOSE && tok != TOKEN_END)
    {
        std::ostringstream err;
        err << "Trailing junk (" << Tokenizer::TokenName(tok) << ") after array literal.";
        throw std::runtime_error(err.str());
    }

    // Arrange for flattened array to be popped off the stack once the
    // top-level parser receives the TOKEN_CLOSE.
    ProgramParseState::ExpressionParseState
        flatState(Convert::GetIdentityFactory(), p_state.m_tokenizer.GetValue(), p_state.m_tokenizer.GetPosition());
    p_state.m_parseStack.push_back(flatState);
    p_state.m_parseStack.back().Add(*flat);
    FF2_ASSERT(p_state.m_parseStack.size() == parseDepth + 1);
    FF2_ASSERT(tok == TOKEN_CLOSE || tok == TOKEN_END);
    return tok;
}

FreeForm2::Token
FreeForm2::SExpressionParse::ParseLet(ProgramParseState &p_state)
{
    if (p_state.m_parsingLambdaBody)
    {
        std::ostringstream err;
        err << "Cannot include a let expression within a lambda function body.";
        throw std::runtime_error(err.str());
    }

    size_t parseDepth = p_state.m_parseStack.size();
    std::vector<SIZED_STRING> bindings;

    p_state.m_parseStack.push_back(
        ProgramParseState::ExpressionParseState(c_letFactory,
                                                p_state.m_tokenizer.GetValue(),
                                                p_state.m_tokenizer.GetPosition()));

    ProgramParseState::ExpressionParseState &letState = p_state.m_parseStack.back();

    Token tok = p_state.m_tokenizer.Advance();
    if (tok != TOKEN_OPEN)
    {
        std::ostringstream err;
        err << "Expected open parenthesis after 'let', "
            << "got something else (" << Tokenizer::TokenName(tok) << " token).";
        throw std::runtime_error(err.str());
    }

    // Iterate through binding pairs.
    tok = p_state.m_tokenizer.Advance();
    do
    {
        if (tok != TOKEN_OPEN)
        {
            std::ostringstream err;
            err << "Expected parenthesised pairs of bindings after "
                << "let open parentheses, "
                << "got something else (" << Tokenizer::TokenName(tok) << " token).";
            throw std::runtime_error(err.str());
        }

        tok = p_state.m_tokenizer.Advance();
        bindings.push_back(p_state.m_tokenizer.GetValue());
        if (tok != TOKEN_ATOM)
        {
            std::ostringstream err;
            err << "Expected atom to be bound in let binding pair, "
                << "got something else (" << Tokenizer::TokenName(tok) << " token).";
            throw std::runtime_error(err.str());
        }

        // Now parse the remaining expression using the main parser.
        p_state.m_tokenizer.Advance();
        tok = SExpressionParse::ParseTokens(p_state, 1);

        if (tok != TOKEN_CLOSE)
        {
            std::ostringstream err;
            err << "Expected binding of name to a single expression: "
                << "got trailing junk (" << Tokenizer::TokenName(tok) << ").";
            throw std::runtime_error(err.str());
        }

        FF2_ASSERT(bindings.size() == p_state.m_parseStack.back().m_children.size());
        FF2_ASSERT(p_state.m_parseStack.size() == parseDepth + 1);
        const SymbolTable::Symbol symbol(bindings.back());
        // If the variable is a function, create a binding from it directly to the function expression
        // so the invoke ExpressionParseState can access the FunctionExpression.
        if (p_state.GetLastParsed().GetType().Primitive() == Type::Function)
        {
            p_state.m_symbols.Bind(symbol, p_state.m_parseStack.back().m_children.back());
        }
        else
        {
            // Bind variable (we do this incrementally, so each successive
            // binding can refer to the previously bound expressions as well).
            const VariableID id = p_state.GetNextVariableId();
            boost::shared_ptr<Expression> exp(
                new VariableRefExpression(Annotations(SourceLocation(1, p_state.m_tokenizer.GetPosition())),
                                          id,
                                          0,
                                          p_state.GetLastParsed().GetType()));
            p_state.m_owner->AddExpression(exp);
            p_state.m_symbols.Bind(symbol, exp.get());
            letState.m_variableIds.push_back(id);
        }

        tok = p_state.m_tokenizer.Advance();
    } while (tok != TOKEN_END && tok != TOKEN_CLOSE);

    // Make sure they closed with a close parens.
    if (tok != TOKEN_CLOSE)
    {
        if (tok == TOKEN_END)
        {
            return tok;
        }
        else
        {
            std::ostringstream err;
            err << "Expected parenthesised pairs of bindings to finish "
                << "with a close parenthesis, "
                << "got something else (" << Tokenizer::TokenName(tok) << " token).";
            throw std::runtime_error(err.str());
        }
    }

    // Make sure they provided at least one binding (there's nothing explicitly
    // wrong with not providing any bindings, but it seems weird, so i'm
    // banning it for now).
    if (bindings.size() == 0)
    {
        std::ostringstream err;
        err << "Let expressions that bind no variables are not currently allowed.";
        throw std::runtime_error(err.str());
    }

    // Parse the bound expression.
    tok = p_state.m_tokenizer.Advance();
    if (tok == TOKEN_CLOSE || tok == TOKEN_END)
    {
        std::ostringstream err;
        err << "Expected an expression after bindings in let "
            << "expression, but let has no additional arguments.";
        throw std::runtime_error(err.str());
    }
    tok = SExpressionParse::ParseTokens(p_state, 1);

    // Make sure they closed with a close parens.
    if (tok != TOKEN_CLOSE)
    {
        if (tok == TOKEN_END)
        {
            return tok;
        }
        else
        {
            std::ostringstream err;
            err << "Expected a single expression after bindings in let "
                << "expression, but let has additional arguments ("
                << Tokenizer::TokenName(tok) << ")";
            throw std::runtime_error(err.str());
        }
    }

    // Remove bindings (note reverse iteration order to pop things off local
    // variable stack in the order they were pushed on).
    for (std::vector<SIZED_STRING>::const_reverse_iterator rnameIter = bindings.rbegin();
         rnameIter != bindings.rend();
         ++rnameIter)
    {
        p_state.m_symbols.Unbind(SymbolTable::Symbol(*rnameIter));
    }

    FF2_ASSERT(p_state.m_parseStack.size() == parseDepth + 1);
    return tok;
}

FreeForm2::Token
FreeForm2::SExpressionParse::ParseMacroLet(ProgramParseState &p_state)
{
    // Ideally, disallowing macro-let while recording a macro should preclude
    // macro-let expressions during macro playback, but if string concatenation
    // or other complex macro expansion mechanisms are ever added, macro
    // expansion could potentially introduce new operator calls which were not
    // present in the original recording.
    if (p_state.m_tokenizer.IsRecordingMacro() || p_state.m_tokenizer.IsExpandingMacro())
    {
        throw std::runtime_error("Macro definitions may not contain macro-let expressions.");
    }

    const size_t parseDepth = p_state.m_parseStack.size();

    // Push an identity factory to allow this ExpressionParseState to evaluate
    // to the macro-let body expression.
    p_state.m_parseStack.push_back(
        ProgramParseState::ExpressionParseState(Convert::GetIdentityFactory(),
                                                p_state.m_tokenizer.GetValue(),
                                                p_state.m_tokenizer.GetPosition()));

    ProgramParseState::ExpressionParseState &state = p_state.m_parseStack.back();

    Token tok = p_state.m_tokenizer.Advance();
    if (tok != TOKEN_OPEN)
    {
        std::ostringstream err;
        err << "Expected open parenthesis after 'macro-let', "
            << "got something else (" << Tokenizer::TokenName(tok) << " token).";
        throw std::runtime_error(err.str());
    }

    std::vector<SIZED_STRING> macros;

    // Iterate through binding pairs.
    tok = p_state.m_tokenizer.Advance();
    do
    {
        if (tok != TOKEN_OPEN)
        {
            std::ostringstream err;
            err << "Expected parenthesised pairs of bindings after "
                << "macro-let open parentheses, "
                << "got something else (" << Tokenizer::TokenName(tok) << " token).";
            throw std::runtime_error(err.str());
        }

        tok = p_state.m_tokenizer.Advance();
        SIZED_STRING name = p_state.m_tokenizer.GetValue();
        macros.push_back(name);

        if (tok != TOKEN_ATOM)
        {
            std::ostringstream err;
            err << "Expected atom to be bound in macro-let binding pair, "
                << "got something else (" << Tokenizer::TokenName(tok) << " token).";
            throw std::runtime_error(err.str());
        }

        // Now parse the remaining expression using the main parser.
        p_state.m_tokenizer.StartMacro(name);
        tok = p_state.m_tokenizer.Advance();

        if (tok == TOKEN_OPEN)
        {
            tok = ParseUntilClosed(p_state, TOKEN_OPEN, TOKEN_CLOSE);
            if (tok == TOKEN_END)
            {
                p_state.m_tokenizer.EndMacro();
                return tok;
            }
        }
        else if (tok == TOKEN_OPEN_ARRAY)
        {
            tok = ParseUntilClosed(p_state, TOKEN_OPEN_ARRAY, TOKEN_CLOSE_ARRAY);
            if (tok == TOKEN_END)
            {
                p_state.m_tokenizer.EndMacro();
                return tok;
            }
        }

        // Always consume at least one token.
        tok = p_state.m_tokenizer.Advance();
        p_state.m_tokenizer.EndMacro();

        if (tok != TOKEN_CLOSE)
        {
            if (tok == TOKEN_END)
            {
                return tok;
            }
            else
            {
                std::ostringstream err;
                err << "Expected binding of name to a single expression: "
                    << "got trailing junk ("
                    << Tokenizer::TokenName(tok) << ").";
                throw std::runtime_error(err.str());
            }
        }

        FF2_ASSERT(p_state.m_parseStack.size() == parseDepth + 1);
        state.m_children.clear();

        tok = p_state.m_tokenizer.Advance();
    } while (tok != TOKEN_END && tok != TOKEN_CLOSE);

    // Make sure they closed with a close parens.
    if (tok != TOKEN_CLOSE)
    {
        if (tok == TOKEN_END)
        {
            return tok;
        }
        else
        {
            std::ostringstream err;
            err << "Expected parenthesised pairs of bindings to finish "
                << "with a close parenthesis, "
                << "got something else (" << Tokenizer::TokenName(tok) << " token).";
            throw std::runtime_error(err.str());
        }
    }

    // Make sure they provided at least one binding (there's nothing explicitly
    // wrong with not providing any bindings, but it seems weird, so it's
    // banned it for now).
    if (macros.size() == 0)
    {
        std::ostringstream err;
        err << "Macro-let expressions that bind no macros are not currently allowed.";
        throw std::runtime_error(err.str());
    }

    // Parse the bound expression. Since this will be added to an
    // IdentityExpressionFactory, the ExpressionParseState which will be
    // finished by the caller will produce the body of the macro-let.
    p_state.m_tokenizer.Advance();
    tok = SExpressionParse::ParseTokens(p_state, 1);

    // Make sure they closed with a close parens.
    if (tok != TOKEN_CLOSE)
    {
        if (tok == TOKEN_END)
        {
            return tok;
        }
        else
        {
            std::ostringstream err;
            err << "Expected a single expression after bindings in macro-let "
                << "expression, but macro-let has additional arguments ("
                << Tokenizer::TokenName(tok) << ")";
            throw std::runtime_error(err.str());
        }
    }

    // Remove bindings (note reverse iteration order to pop things off local
    // variable stack in the order they were pushed on).
    for (auto iter = macros.crbegin(); iter != macros.crend(); ++iter)
    {
        p_state.m_tokenizer.DeleteMacro(*iter);
    }

    FF2_ASSERT(p_state.m_parseStack.size() == parseDepth + 1);
    return tok;
}

FreeForm2::Token
FreeForm2::SExpressionParse::ParseRangeReduce(ProgramParseState &p_state)
{
    const size_t parseDepth = p_state.m_parseStack.size();
    const VariableID stepId = p_state.GetNextVariableId();
    const VariableID reduceId = p_state.GetNextVariableId();

    ProgramParseState::ExpressionParseState
        rangeState(c_rangeFactory, p_state.m_tokenizer.GetValue(), p_state.m_tokenizer.GetPosition());
    rangeState.m_variableIds.push_back(stepId);
    rangeState.m_variableIds.push_back(reduceId);

    p_state.m_parseStack.push_back(rangeState);

    // Get next token (should be an atom).
    Token tok = p_state.m_tokenizer.Advance();
    if (tok != TOKEN_ATOM)
    {
        std::ostringstream err;
        err << "Expected variable name 'range-reduce', "
            << "got something else (" << Tokenizer::TokenName(tok) << " token).";
        throw std::runtime_error(err.str());
    }
    const SymbolTable::Symbol rangeSymbol(p_state.m_tokenizer.GetValue());

    // Parse range limits.
    tok = p_state.m_tokenizer.Advance();
    tok = SExpressionParse::ParseTokens(p_state, 2);
    if (tok == TOKEN_END)
    {
        return tok;
    }

    // Next token should be an atom.
    if (tok != TOKEN_ATOM)
    {
        std::ostringstream err;
        err << "Expected variable name 'range-reduce', "
            << "got something else (" << Tokenizer::TokenName(tok) << " token).";
        throw std::runtime_error(err.str());
    }
    const SymbolTable::Symbol accSymbol(p_state.m_tokenizer.GetValue());

    // Parse initial value expression.
    tok = p_state.m_tokenizer.Advance();
    tok = SExpressionParse::ParseTokens(p_state, 1);
    if (tok == TOKEN_END)
    {
        return tok;
    }

    // Bind previous and current value variables.
    boost::shared_ptr<Expression>
        boundRange(new VariableRefExpression(Annotations(SourceLocation(1, p_state.m_tokenizer.GetPosition())),
                                             stepId,
                                             0,
                                             TypeImpl::GetIntInstance(true)));
    p_state.m_owner->AddExpression(boundRange);
    boost::shared_ptr<Expression> accExp(
        new VariableRefExpression(Annotations(SourceLocation(1, p_state.m_tokenizer.GetPosition())),
                                  reduceId,
                                  0,
                                  p_state.GetLastParsed().GetType()));
    p_state.m_owner->AddExpression(accExp);
    p_state.m_symbols.Bind(accSymbol, accExp.get());
    p_state.m_symbols.Bind(rangeSymbol, boundRange.get());

    // Parse reduction expression.
    tok = SExpressionParse::ParseTokens(p_state, 1);
    if (tok == TOKEN_END)
    {
        return tok;
    }

    // Unbind the variables, so they can't be used anymore.
    p_state.m_symbols.Unbind(rangeSymbol);
    p_state.m_symbols.Unbind(accSymbol);

    // Next token should be a close.
    if (tok != TOKEN_CLOSE)
    {
        std::ostringstream err;
        err << "Expected close of 'range-reduce', "
            << "got something else (" << Tokenizer::TokenName(tok) << " token).";
        throw std::runtime_error(err.str());
    }

    // Return close token to main parser, it will take care of the rest.
    FF2_ASSERT(p_state.m_parseStack.size() == parseDepth + 1);
    return tok;
}

FreeForm2::Token
FreeForm2::SExpressionParse::ParseLambda(ProgramParseState &p_state)
{
    if (p_state.m_parsingAggregatedExpression)
    {
        std::ostringstream err;
        err << "Cannot include a lambda expression within an aggregated expression.";
        throw std::runtime_error(err.str());
    }

    if (p_state.m_parsingLambdaBody)
    {
        std::ostringstream err;
        err << "Cannot include a lambda expression within a lambda function body.";
        throw std::runtime_error(err.str());
    }
    p_state.m_parsingLambdaBody = true;

    const size_t parseDepth = p_state.m_parseStack.size();
    std::vector<SIZED_STRING> bindings;

    // Push a factory that will produce the function expression.
    p_state.m_parseStack.push_back(
        ProgramParseState::ExpressionParseState(c_lambdaFactory,
                                                p_state.m_tokenizer.GetValue(),
                                                p_state.m_tokenizer.GetPosition()));

    ProgramParseState::ExpressionParseState &lambdaState = p_state.m_parseStack.back();

    Token tok = p_state.m_tokenizer.Advance();
    if (tok != TOKEN_OPEN)
    {
        std::ostringstream err;
        err << "Expected open parenthesis after 'lambda', "
            << "got something else (" << Tokenizer::TokenName(tok) << " token).";
        throw std::runtime_error(err.str());
    }

    // Iterate through lambda parameters
    tok = p_state.m_tokenizer.Advance();
    while (tok != TOKEN_CLOSE && tok != TOKEN_END)
    {
        if (tok != TOKEN_ATOM && tok != TOKEN_OPEN)
        {
            std::ostringstream err;
            err << "Expected formal declaration, got something else ("
                << Tokenizer::TokenName(tok) << " token).";
            throw std::runtime_error(err.str());
        }

        const TypeImpl *type = &TypeImpl::GetUnknownType().AsConstType();
        bool matchOpenParen = false;
        if (tok == TOKEN_OPEN)
        {
            matchOpenParen = true;
            tok = p_state.m_tokenizer.Advance();
            if (tok != TOKEN_ATOM)
            {
                std::ostringstream err;
                err << "Expected atom declaration, got something else ("
                    << Tokenizer::TokenName(tok) << " token).";
                throw std::runtime_error(err.str());
            }
            const Type::TypePrimitive prim = Type::ParsePrimitive(p_state.m_tokenizer.GetValue());
            if (prim != Type::Float && prim != Type::Int && prim != Type::Bool)
            {
                std::ostringstream err;
                err << "Expected type in lambda formals, got something else ("
                    << p_state.m_tokenizer.GetValue() << ").";
                throw std::runtime_error(err.str());
            }

            type = &TypeImpl::GetCommonType(prim, true);
            tok = p_state.m_tokenizer.Advance();
        }

        bindings.push_back(p_state.m_tokenizer.GetValue());
        const VariableID id = p_state.GetNextVariableId();
        boost::shared_ptr<Expression> exp(
            new VariableRefExpression(Annotations(SourceLocation(1, p_state.m_tokenizer.GetPosition())),
                                      id,
                                      0,
                                      *type));
        p_state.m_owner->AddExpression(exp);
        p_state.m_symbols.Bind(SymbolTable::Symbol(bindings.back()), exp.get());
        lambdaState.m_variableIds.push_back(id);
        lambdaState.m_children.push_back(exp.get());

        tok = p_state.m_tokenizer.Advance();

        if (matchOpenParen)
        {
            if (tok != TOKEN_CLOSE)
            {
                std::ostringstream err;
                err << "Expected close parenthesis, got something else ("
                    << Tokenizer::TokenName(tok) << " token).";
                throw std::runtime_error(err.str());
            }

            tok = p_state.m_tokenizer.Advance();
        }
    }
    FF2_ASSERT(lambdaState.m_variableIds.size() == lambdaState.m_children.size());

    if (bindings.size() == 0)
    {
        std::ostringstream err;
        err << "lambdas must have formals (use a regular let binding "
            << "for computations without parameters)";
        throw std::runtime_error(err.str());
    }

    // Parse the bound expression.
    p_state.m_tokenizer.Advance();
    tok = SExpressionParse::ParseTokens(p_state, 1);

    // Make sure they closed with a close parens.
    if (tok != TOKEN_CLOSE)
    {
        if (tok == TOKEN_END)
        {
            return tok;
        }
        else
        {
            std::ostringstream err;
            err << "Expected a single expression after formals in lambda "
                << "expression, but lambda has additional arguments ("
                << Tokenizer::TokenName(tok) << ")";
            throw std::runtime_error(err.str());
        }
    }

    // Remove bindings (note reverse iteration order to pop things off local
    // variable stack in the order they were pushed on).
    for (auto rnameIter = bindings.crbegin(); rnameIter != bindings.crend(); ++rnameIter)
    {
        p_state.m_symbols.Unbind(SymbolTable::Symbol(*rnameIter));
    }

    p_state.m_parsingLambdaBody = false;

    FF2_ASSERT(p_state.m_parseStack.size() == parseDepth + 1);
    return tok;
}

FreeForm2::Token
FreeForm2::SExpressionParse::ParseInvoke(ProgramParseState &p_state)
{
    if (p_state.m_parsingAggregatedExpression)
    {
        std::ostringstream err;
        err << "Cannot include an invoke expression within an aggregated expression.";
        throw std::runtime_error(err.str());
    }

    if (p_state.m_parsingLambdaBody)
    {
        std::ostringstream err;
        err << "Cannot include a invoke expression within a lambda function body.";
        throw std::runtime_error(err.str());
    }

    const size_t parseDepth = p_state.m_parseStack.size();

    // Push a factory that will produce the function expression.
    p_state.m_parseStack.push_back(
        ProgramParseState::ExpressionParseState(c_invokeFactory,
                                                p_state.m_tokenizer.GetValue(),
                                                p_state.m_tokenizer.GetPosition()));

    // Process the parameters within the Invoke expression.
    Token tok = p_state.m_tokenizer.Advance();
    while (tok != TOKEN_CLOSE)
    {
        tok = SExpressionParse::ParseTokens(p_state, 1);
    }

    FF2_ASSERT(p_state.m_parseStack.size() == parseDepth + 1);
    return tok;
}
