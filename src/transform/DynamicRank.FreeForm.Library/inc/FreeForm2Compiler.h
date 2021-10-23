#pragma once

#ifndef FREEFORM2_INC_COMPILER_H
#define FREEFORM2_INC_COMPILER_H

#include <boost/noncopyable.hpp>
#include <boost/scoped_ptr.hpp>
#include <memory>
#include <string>

namespace DynamicRank
{
    class IFeatureMap;
}

namespace FreeForm2
{
    class CompilerImpl;
    class ExternalDataManager;
    class Program;

    // This class contains the results of the compilation process. Since the
    // result of the compilation process is backend-dependent, this class 
    // should be downcast to the backend result type according to the compiler
    // used.
    class CompilerResults : boost::noncopyable
    {
    public:
        virtual ~CompilerResults();
    };

    // A compiler compiles multiple programs using the same set of compiler
    // resources, which amortises costs across different programs.
    class Compiler : boost::noncopyable
    {
    public:
        Compiler(std::auto_ptr<CompilerImpl> p_impl);

        ~Compiler();

        // Compile the program producing a backend-dependent results obejct. 
        // This method, optionally producing debug output on stderr.  
        std::unique_ptr<CompilerResults>
        Compile(const Program& p_program, bool p_debugOutput);

        // Default optimization level, used whenever the level is not explicitly specified.
        static const unsigned int c_defaultOptimizationLevel = 0;

    private:
        // Pointer to implementation class (pimpl idiom).  
        boost::scoped_ptr<CompilerImpl> m_impl;
    };
}

// TODO: Remove the following include (TFS# 453473).
// This is used to prevent a breaking API change (moving compiler factory into
// a separate header).
#include "FreeForm2CompilerFactory.h"

#endif
