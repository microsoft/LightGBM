/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#pragma once

#ifndef FREEFORM2_ARITHMETIC_H
#define FREEFORM2_ARITHMETIC_H

namespace FreeForm2 {
class ExpressionFactory;

namespace Arithmetic {
const ExpressionFactory &GetPlusInstance();
const ExpressionFactory &GetMinusInstance();
const ExpressionFactory &GetMultiplyInstance();
const ExpressionFactory &GetDividesInstance();
const ExpressionFactory &GetIntegerDivInstance();
const ExpressionFactory &GetIntegerModInstance();
const ExpressionFactory &GetModInstance();
const ExpressionFactory &GetMaxInstance();
const ExpressionFactory &GetMinInstance();
const ExpressionFactory &GetPowInstance();
const ExpressionFactory &GetUnaryLogInstance();
const ExpressionFactory &GetBinaryLogInstance();
const ExpressionFactory &GetLog1Instance();
const ExpressionFactory &GetAbsInstance();
const ExpressionFactory &GetRoundInstance();
const ExpressionFactory &GetTruncInstance();
}  // namespace Arithmetic
};  // namespace FreeForm2

#endif
