/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#pragma once

#include <basic_types.h>

#include <boost/noncopyable.hpp>
#include <memory>

namespace llvm {
class LLVMContext;
class ExecutionEngine;
class Module;
class GlobalValue;
class Function;
}  // namespace llvm

namespace FreeForm2 {
class LlvmRuntimeLibrary : boost::noncopyable {
 public:
  // Create a LLVM Runtime Library using the specified LLVMContext.
  LlvmRuntimeLibrary(llvm::LLVMContext &p_context);

  // Destroy the implementation.
  ~LlvmRuntimeLibrary();

  // Add all runtime symbols to the specified module. This is similar to
  // adding forward declarations to a .cpp file.
  void AddLibraryToModule(llvm::Module &p_module) const;

  // Add global value mappings to an exeuction engine, which is necessary
  // when linking a module which uses the runtime library.
  void AddExecutionMappings(llvm::ExecutionEngine &p_engine) const;

  // Look up a GlobalValue by name. GlobalValues generally include
  // external variables and functions; see the LLVM documentation for
  // more information. If the value is not found, this method returns
  // null.
  llvm::GlobalValue *FindValue(SIZED_STRING p_name) const;

  // Find a runtime function with the specified name. This is a
  // specialization of FindValue for Functions. If the function is not
  // found, or if the GlobalValue is not a function, this method returns
  // null.
  llvm::Function *FindFunction(SIZED_STRING p_name) const;

 private:
  // The implementation of this class is hidden.
  class Impl;

  // Pointer to the implementation.
  std::unique_ptr<Impl> m_impl;
};
}  // namespace FreeForm2

extern "C" double FreeForm2GetRandomValue();

inline unsigned long GetTickCount() {
  struct timespec ts;

  clock_gettime(CLOCK_MONOTONIC, &ts);

  return (ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}
