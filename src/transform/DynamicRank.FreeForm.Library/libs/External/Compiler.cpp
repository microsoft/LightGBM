/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include "Compiler.h"

#include <memory>

#include "FreeForm2Compiler.h"
#include "FreeForm2Program.h"

FreeForm2::CompilerImpl::~CompilerImpl() {}

FreeForm2::CompilerResults::~CompilerResults() {}

FreeForm2::Compiler::Compiler(std::auto_ptr<CompilerImpl> p_impl)
    : m_impl(p_impl.release()) {}

FreeForm2::Compiler::~Compiler() {}

std::unique_ptr<FreeForm2::CompilerResults> FreeForm2::Compiler::Compile(
    const Program &p_program, bool p_debugOutput) {
  return m_impl->Compile(p_program.GetImplementation(), p_debugOutput);
}
