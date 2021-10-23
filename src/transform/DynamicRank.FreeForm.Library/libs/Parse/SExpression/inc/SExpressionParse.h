#pragma once

#ifndef FREEFORM2_SEXPRESSIONPARSE_H
#define FREEFORM2_SEXPRESSIONPARSE_H

#include <boost/tuple/tuple.hpp>
#include <boost/shared_ptr.hpp>
#include "FreeForm2.h"
#include "FreeForm2Tokenizer.h"

namespace FreeForm2
{
    class ProgramParseState;
    class Expression;
    class ExpressionOwner;
    class ArrayLiteralExpression;
    class TypeManager;

    class SExpressionParse
    {
    public:
        // Main driver function for parsing.  p_expressionLimit can be used to limit
        // the number of expressions parsed into the current expression, with zero
        // reserved to indicate no limit.
        static Token ParseTokens(ProgramParseState& p_state, 
                                 unsigned int p_expressionLimit);

        typedef boost::tuples::tuple<const Expression*,
                                     boost::shared_ptr<ExpressionOwner>,
                                     boost::shared_ptr<TypeManager>> ParserResults;

        // Parse an expression from a string.
        static ParserResults
        Parse(SIZED_STRING p_input, 
              DynamicRank::IFeatureMap& p_map, 
              bool p_mustProduceFloat, 
              bool p_parsingAggregatedExpression);

        // Parse an array dereference expression.
        static Token ParseArrayDereference(ProgramParseState& p_state);

        // Function to parse a let expression.
        static Token ParseLet(ProgramParseState& p_state);

        // Function to parse a macro-let expression.
        static Token ParseMacroLet(ProgramParseState& p_state);

        // Parse a RangeReduceExpression.
        static Token ParseRangeReduce(ProgramParseState& p_state);

        // Parse an ArrayLiteralExpression.
        static Token ParseArrayLiteral(ProgramParseState& p_state);

        // Parse a lambda expression.
        static Token ParseLambda(ProgramParseState& p_state);

        // Parse an invoke expression.
        static Token ParseInvoke(ProgramParseState& p_state);

    private:
        // Recursively parse array literal expressions.
        static const ArrayLiteralExpression& ParseArrayLiteralRecurse(ProgramParseState& p_state);
    };
}

#endif

