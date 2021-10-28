/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#pragma once

#ifndef FREEFORM2_LOGIC_H
#define FREEFORM2_LOGIC_H

namespace FreeForm2 {
class ExpressionFactory;

namespace Logic {
const ExpressionFactory &GetCmpEqInstance();
const ExpressionFactory &GetCmpNotEqInstance();
const ExpressionFactory &GetCmpLTInstance();
const ExpressionFactory &GetCmpLTEInstance();
const ExpressionFactory &GetCmpGTInstance();
const ExpressionFactory &GetCmpGTEInstance();

const ExpressionFactory &GetAndInstance();
const ExpressionFactory &GetOrInstance();
const ExpressionFactory &GetNotInstance();
}  // namespace Logic
};  // namespace FreeForm2

#endif
