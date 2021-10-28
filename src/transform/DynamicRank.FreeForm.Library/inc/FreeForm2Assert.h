/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#pragma once

#ifndef FREEFORM2_ASSERT_H
#define FREEFORM2_ASSERT_H

namespace FreeForm2 {
// Assert a condition, providing file and line of the assertion call.
// Note that this function throws on failure, rather than aborting (like
// standard c assert).
void ThrowAssert(bool p_condition, const char *p_file, unsigned int p_line);

// Assert a condition, providing the expression, file and line of the
// assertion call.
// Note that this function throws on failure, rather than aborting (like
// standard c assert).
void ThrowAssert(bool p_condition, const char *p_expression, const char *p_file,
                 unsigned int p_line);

// Assert that this function call should not be reached during normal
// program execution.  Note that this function throws on failure, rather
// than aborting.
// __declspec(noreturn)
void Unreachable(const char *p_file, unsigned int p_line);
};  // namespace FreeForm2

// Macros for asserting.
// Call the regular FreeForm2::ThrowAssert function with macro-generated
// parameters
#define FF2_ASSERT(cond) \
  FreeForm2::ThrowAssert((cond), #cond, __FILE__, __LINE__)

// Call the regular FreeForm2::Unreachable function with macro-generated
// parameters
#define FF2_UNREACHABLE() FreeForm2::Unreachable(__FILE__, __LINE__)

#endif
