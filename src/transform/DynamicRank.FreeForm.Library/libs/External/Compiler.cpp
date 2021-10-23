#include "Compiler.h"
#include "FreeForm2Compiler.h"

#include "FreeForm2Program.h"
#include <memory>


FreeForm2::CompilerImpl::~CompilerImpl()
{
}


FreeForm2::CompilerResults::~CompilerResults()
{
}


FreeForm2::Compiler::Compiler(std::auto_ptr<CompilerImpl> p_impl)
    : m_impl(p_impl.release())
{
}


FreeForm2::Compiler::~Compiler()
{
}


std::unique_ptr<FreeForm2::CompilerResults>
FreeForm2::Compiler::Compile(const Program& p_program, bool p_debugOutput)
{
    return m_impl->Compile(p_program.GetImplementation(), p_debugOutput);
}
