/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include "FreeForm2Assert.h"

#include <sstream>
#include <stdexcept>

void FreeForm2::ThrowAssert(bool p_condition, const char *p_file, unsigned int p_line)
{
    if (!p_condition)
    {
        std::ostringstream err;
        err << "Assertion error at " << p_file << ":" << p_line;
        throw std::runtime_error(err.str());
    }
}

void FreeForm2::ThrowAssert(bool p_condition, const char *p_expression, const char *p_file, unsigned int p_line)
{
    if (!p_condition)
    {
        std::ostringstream err;
        err << "Assertion error: \"" << p_expression << "\" failed at " << p_file << ":" << p_line;
        throw std::runtime_error(err.str());
    }
}

void FreeForm2::Unreachable(const char *p_file, unsigned int p_line)
{
    std::ostringstream err;
    err << "Unreachable code reached at " << p_file << ":" << p_line;
    throw std::runtime_error(err.str());
}
