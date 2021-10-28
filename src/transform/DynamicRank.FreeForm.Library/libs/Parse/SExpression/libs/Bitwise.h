/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#pragma once

#ifndef FREEFORM2_BITWISE_H
#define FREEFORM2_BITWISE_H

namespace FreeForm2 {
class ExpressionFactory;

namespace Bitwise {
const ExpressionFactory &GetAndInstance();
const ExpressionFactory &GetOrInstance();
const ExpressionFactory &GetNotInstance();
}  // namespace Bitwise
};  // namespace FreeForm2

#endif
