/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#pragma once

#ifndef FREEFORM2_MISC_FACTORY_H
#define FREEFORM2_MISC_FACTORY_H

namespace FreeForm2
{
    class ExpressionFactory;

    namespace Conditional
    {
        const ExpressionFactory &GetIfInstance();
    }

    namespace Random
    {
        const ExpressionFactory &GetRandomFloatInstance();
        const ExpressionFactory &GetRandomIntInstance();
    }

    // Returns an expression factory for the array-length primitive.
    const ExpressionFactory &GetArrayLengthInstance();

    namespace Convert
    {
        const ExpressionFactory &GetFloatConvertFactory();
        const ExpressionFactory &GetIntConvertFactory();
        const ExpressionFactory &GetBoolConversionFactory();
        const ExpressionFactory &GetIdentityFactory();
    }

    namespace Select
    {
        const ExpressionFactory &GetSelectNthInstance();
        const ExpressionFactory &GetSelectRangeInstance();
    }

    const ExpressionFactory &GetFeatureSpecInstance(bool p_mustConvertToFloat);
};

#endif
