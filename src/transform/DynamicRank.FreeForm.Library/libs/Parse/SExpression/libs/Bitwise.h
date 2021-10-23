#pragma once

#ifndef FREEFORM2_BITWISE_H
#define FREEFORM2_BITWISE_H

namespace FreeForm2
{
    class ExpressionFactory;

    namespace Bitwise
    {
        const ExpressionFactory& GetAndInstance();
        const ExpressionFactory& GetOrInstance();
        const ExpressionFactory& GetNotInstance();
    }
};


#endif

