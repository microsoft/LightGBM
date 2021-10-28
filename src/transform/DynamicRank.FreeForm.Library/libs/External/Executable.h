/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#pragma once

#ifndef FREEFORM2_EXECUTABLE_H
#define FREEFORM2_EXECUTABLE_H

#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>

#include "FreeForm2CompilerFactory.h"
#include "FreeForm2Executable.h"

class StreamFeatureInput;

namespace FreeForm2 {
class Result;
class Type;

// An ExectuableImpl is the implementation class for exectuables, which
// currently compiles and runs via one of the backends.
class ExecutableImpl : boost::noncopyable {
 public:
  virtual ~ExecutableImpl();

  virtual boost::shared_ptr<Result> Evaluate(
      StreamFeatureInput *p_input,
      const Executable::FeatureType p_features[]) const = 0;

  // List based evaluation.
  virtual boost::shared_ptr<FreeForm2::Result> Evaluate(
      const Executable::FeatureType *const *p_features,
      UInt32 p_currentDocument, UInt32 p_documentCount,
      Int64 *p_cache) const = 0;

  virtual Executable::DirectEvalFun EvaluationFunction() const = 0;

  // Get list based evaluation function.
  virtual Executable::AggregatedEvalFun AggregatedEvaluationFunction()
      const = 0;

  virtual const Type &GetType() const = 0;

  // Get the size of external memory.
  virtual size_t GetExternalSize() const = 0;
};
}  // namespace FreeForm2

#endif
