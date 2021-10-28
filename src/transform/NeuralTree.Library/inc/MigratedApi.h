/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#pragma once
#include <stdlib.h>

template <typename T, std::size_t N>
constexpr std::size_t countof(T const (&)[N]) noexcept {
  return N;
}

static inline unsigned int rotl(const unsigned int x, int bits) {
  const unsigned int n = ((bits % 32) + 32) % 32;
  return (x << n) | (x >> (32 - n));
}

#ifdef _snprintf_s
#undef _snprintf_s
#endif
#define _snprintf_s(a, b, c, ...) snprintf(a, b, __VA_ARGS__)
