#pragma once

#ifndef FREEFORM2_MATCH_SUB_H
#define FREEFORM2_MATCH_SUB_H

#include "Expression.h"
#include "FreeForm2Assert.h"

namespace FreeForm2
{
    // A match sub- expression is a base class representing match
    // operators (that is, repetition, concatenation, etc).
    class MatchSubExpression : public Expression
    {
    public:
        struct Info
        {
            Info(unsigned int p_minLength, unsigned int p_maxLength)
                : m_minLength(p_minLength), m_maxLength(p_maxLength)
            {
                FF2_ASSERT(m_minLength <= m_maxLength);
                FF2_ASSERT(m_minLength != c_indeterminate);
            }

            // Calculated minimum limit on the length of matches from this FSM.
            // Must be less than or equal to m_maxLength, and not c_indeterminate.
            unsigned int m_minLength;

            // Calculated maximum limit on the length of matches from this FSM.
            // Will be c_indeterminate if there's no easily calculable limit on the
            // length of match using this FSM.
            unsigned int m_maxLength;

            // Constant indicating that a pattern matches arbitrarily long input.
            static const unsigned int c_indeterminate = UINT_MAX;
        };


        MatchSubExpression(const Annotations& p_annotations)
            : Expression(p_annotations)
        {
        }

        virtual ~MatchSubExpression()
        {
        }

        // Calculate information for this sub-expression.
        virtual Info GetInfo() const = 0;
    };
};

#endif

