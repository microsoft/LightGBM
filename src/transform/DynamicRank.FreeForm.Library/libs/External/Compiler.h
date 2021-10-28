/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#pragma once

#ifndef FREEFORM2_COMPILER
#define FREEFORM2_COMPILER

#include <boost/noncopyable.hpp>
#include <memory>

namespace FreeForm2
{
    class CompilerResults;
    class ProgramImpl;

    class CompilerImpl : boost::noncopyable
    {
    public:
        virtual ~CompilerImpl();

        // Compile the given program.
        virtual std::unique_ptr<CompilerResults> Compile(const ProgramImpl &p_program,
                                                         bool p_debugOutput) = 0;
    };
}

#endif
