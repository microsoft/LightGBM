/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#pragma once

#ifndef FREEFORM2_COMPILER_FACTORY_H
#define FREEFORM2_COMPILER_FACTORY_H

#include <memory>
#include <string>

namespace FreeForm2 {
namespace Cpp {
class ExternResolver;
}

class Compiler;

// This class contains methods to instantiate the various types of
// compilers.
class CompilerFactory {
 public:
  enum DestinationFunctionType {
    // typedef float (*DirectEvalFun)(StreamFeatureInput*, const FeatureType[],
    // OutputType[]).
    SingleDocumentEvaluation,

    // typedef float (*AggregatedEvalFun)(StreamFeatureInput*, const
    // FeatureType[][], UInt32, UInt32, OutputType[]).
    DocumentSetEvaluation,
  };

  // Create a compiler which takes a program and compiles it into an
  // ExecutableCompilerResults object. p_optimizationLevel describes the
  // degree of optimization to perform on the code with an integer,
  // analagous to 'gcc -O p_optimizationLevel'.
  static std::unique_ptr<Compiler> CreateExecutableCompiler(
      unsigned int p_optimizationLevel,
      DestinationFunctionType p_destinationFunctionType =
          SingleDocumentEvaluation);

  // This method creates a Compiler object to compile a program to a
  // C++/IFM target. The results of this compiler are defined in
  // FreeForm2CppCompiler.h.
  static std::unique_ptr<Compiler> CreateCppIFMCompiler(
      const Cpp::ExternResolver &p_resolver);

  // This method creates a Compiler object to compile a program to a
  // C++/Barramundi target. The results of this compiler are defined
  // in FreeForm2CppCompiler.h.
  static std::unique_ptr<Compiler> CreateCppBarramundiCompiler(
      const Cpp::ExternResolver &p_resolver, const std::string &p_metadataPath);

  // This method creates a Compiler object to compile a program to a
  // C++/Barramundi target with debugging instrumentation present in the
  // program. The printf command is the name of a printf-style function
  // for use with debugging statements. The results of this compiler are
  // defined in FreeForm2CppCompiler.h.
  static std::unique_ptr<Compiler> CreateDebuggingCppBarramundiCompiler(
      const Cpp::ExternResolver &p_resolver, const std::string &p_metadataPath,
      const std::string &p_printfCommand);

  // This method creates a Compiler object to compile a program to a
  // C++/FPGA target. The results of this compiler are defined
  // in FreeForm2CppCompiler.h.
  static std::unique_ptr<Compiler> CreateCppFpgaCompiler(
      const Cpp::ExternResolver &p_resolver,
      const std::string &p_outputMappingPath, const std::string &p_msdlPath);
};
}  // namespace FreeForm2

#endif
