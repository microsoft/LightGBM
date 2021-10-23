#pragma once

#ifndef FREEFORM2_LOGIC_H
#define FREEFORM2_LOGIC_H

namespace FreeForm2
{
    class ExpressionFactory;

    namespace Logic
    {
        const ExpressionFactory& GetCmpEqInstance();
        const ExpressionFactory& GetCmpNotEqInstance();
        const ExpressionFactory& GetCmpLTInstance();
        const ExpressionFactory& GetCmpLTEInstance();
        const ExpressionFactory& GetCmpGTInstance();
        const ExpressionFactory& GetCmpGTEInstance();

        const ExpressionFactory& GetAndInstance();
        const ExpressionFactory& GetOrInstance();
        const ExpressionFactory& GetNotInstance();
    }
};


#endif

