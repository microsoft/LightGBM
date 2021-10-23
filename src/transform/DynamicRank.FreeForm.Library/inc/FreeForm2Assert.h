#pragma once

#ifndef FREEFORM2_ASSERT_H
#define FREEFORM2_ASSERT_H

namespace FreeForm2
{
    // Assert a condition, providing file and line of the assertion call.
    // Note that this function throws on failure, rather than aborting (like
    // standard c assert).
    void ThrowAssert(bool p_condition, const char* p_file, unsigned int p_line);

    // Assert a condition, providing the expression, file and line of the
    // assertion call.
    // Note that this function throws on failure, rather than aborting (like
    // standard c assert).
    void ThrowAssert(bool p_condition, const char* p_expression, const char* p_file, unsigned int p_line);

    // Assert that this function call should not be reached during normal
    // program execution.  Note that this function throws on failure, rather 
    // than aborting.
    // __declspec(noreturn) 
    void Unreachable(const char* p_file, unsigned int p_line);
};

// Macros for asserting.
#define FF2_ASSERT(cond)                                                      \
    /* Call the regular FreeForm2::ThrowAssert function with macro-generated  \
     * parameters */                                                          \
    FreeForm2::ThrowAssert((cond),                                            \
                           #cond,                                             \
                           __FILE__,                                          \
                           __LINE__)                                           

#define FF2_UNREACHABLE()                                                     \
    /* Call the regular FreeForm2::Unreachable function with macro-generated  \
     * parameters */                                                          \
    FreeForm2::Unreachable(__FILE__,                                          \
                           __LINE__)                                           

#endif
