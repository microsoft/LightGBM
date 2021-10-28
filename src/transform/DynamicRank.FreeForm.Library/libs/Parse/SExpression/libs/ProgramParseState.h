/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#pragma once

#ifndef FREEFORM2_PROGRAM_PARSE_STATE_H
#define FREEFORM2_PROGRAM_PARSE_STATE_H

#include <basic_types.h>
#include <boost/shared_ptr.hpp>
#include <boost/variant.hpp>
#include "Expression.h"
#include "FreeForm2Tokenizer.h"
#include <list>
#include <map>
#include "SExpressionParse.h"
#include "SimpleExpressionOwner.h"
#include "SymbolTable.h"

namespace FreeForm2
{
    class ExpressionFactory;
    class TypeManager;

    // ProgramParseState represents the parsing state of a program.
    class ProgramParseState
    {
    public:
        // Class that represents an s-expression as it's being parsed.  We
        // accumulate children, and then create a root expression
        class ExpressionParseState
        {
        public:
            // Construct an ExpressionParseState object, from the
            // expression factory used for parsing, and the atom that
            // caused this object to be constructed (only used in error messages).
            ExpressionParseState(const ExpressionFactory &p_factory,
                                 SIZED_STRING p_atom,
                                 unsigned int p_offset);

            // Add a subexpression to this expression.
            void Add(const Expression &p_expr);

            // Finish parsing and return this expression.
            const Expression &Finish(SimpleExpressionOwner &p_owner,
                                     TypeManager &p_typeManager) const;

            // Accumulated children.
            std::vector<const Expression *> m_children;

            // Variable IDs allocated by the parser for this expression.
            std::vector<VariableID> m_variableIds;

            // Expression factory that will produce the finished expression.
            const ExpressionFactory *m_factory;

            // Atom that caused this parsestate to be created.
            SIZED_STRING m_atom;

            // The offset of the current token with respect to the beginning of the expression.
            unsigned int m_offset;
        };

        // Function that parses a special form (such as 'let').
        typedef Token (*ParsingFunction)(ProgramParseState &p_state);

        // Variant that provides either an expression factory for standard
        // parsing, or a ParsingFunction for special form parsing, depending on
        // the operator.
        typedef boost::variant<const ExpressionFactory *, ParsingFunction> OperatorInfo;

        // Map of operator names to parsing method.
        typedef std::map<std::string, OperatorInfo> OpMap;

        // Construct a parse state object from the input program, the feature
        // map used for this program, the available set of operators, and an
        // indication as to whether this program must produce a float (by
        // conversion, if necessary - this also means things that can't be
        // converted to a float produce errors during parsing).
        ProgramParseState(SIZED_STRING p_input,
                          DynamicRank::IFeatureMap &p_map,
                          const OpMap &p_operators,
                          bool p_mustProduceFloat,
                          bool p_parsingAggregatedExpression);

        // Get the next variable ID.
        VariableID GetNextVariableId();

        // Return the last expression parsed.
        const Expression &GetLastParsed() const;

        // Finish parsing, producing the final expression and owner.
        SExpressionParse::ParserResults Finish();

        // Close the expression currently being parsed.
        void CloseExpression();

        // Object to own produced expressions.
        boost::shared_ptr<SimpleExpressionOwner> m_owner;

        // Object to own/manage the types.
        boost::shared_ptr<TypeManager> m_typeManager;

        // Stack of expressions being parsed.
        std::list<ExpressionParseState> m_parseStack;

        // List of available operators.
        const OpMap &m_operators;

        // Tokenizer, to turn textual input into tokens of different types.
        Tokenizer m_tokenizer;

        // Mapping from strings to bound values.
        SymbolTable m_symbols;

        // Whether a lambda expression is currently being parsed.
        bool m_parsingLambdaBody;

        // Whether an aggregated expression is being parsed.
        const bool m_parsingAggregatedExpression;

    private:
        // Counter to keep track of the next variable ID.  This tracks the
        // number of values that have been allocated for a let statement, or
        // other special forms, and allows us to assign IDs to each value for
        // later use.
        VariableID m_variableIdCounter;
    };
}

#endif
