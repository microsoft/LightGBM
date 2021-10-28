/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#pragma once

#ifndef FREEFORM2_ARRAYCODEGEN_H
#define FREEFORM2_ARRAYCODEGEN_H

#include <basic_types.h>

#include <boost/shared_array.hpp>
#include <boost/static_assert.hpp>
#include <utility>
#include <vector>

#include "ArrayType.h"
#include "LlvmCodeGenerator.h"

namespace FreeForm2 {
class CompilationState;

class ArrayCodeGen {
 public:
  // Number of bits used to count members per flattened dimension.
  // This limits the number of elements in an array dimension.
  static const unsigned int c_bitsPerFlatDimension = 8;

  // Type used to represent encoded array bounds.
  typedef UInt64 ArrayBoundsType;

  // Ensure that our representation can handle all arrays.
  static_assert(
      sizeof(ArrayBoundsType) * 8 >=
          c_bitsPerFlatDimension * ArrayType::c_maxDimensions,
      "Please update ArrayBoundsType to reflect the max number of dimensions.");

  // Type used to calculate the total number of elements in an array.
  // Note that we assume that counting the number of elements in an array
  // will inherently occupy equal of fewer bits than the encoded bounds,
  // which is safe unless we start playing tricks with the bounds
  // representation.
  typedef ArrayBoundsType ArrayCountType;

  // Enumeration that declares where the encoded bounds and space pointer
  // are in the LLVM structure used to represent an array.
  enum ArrayStructPosition {
    // Position of encoded array bounds.
    boundsPosition,

    // Position of total array element count (in flattened elements).
    countPosition,

    // Position of pointer to array space.
    pointerPosition
  };

  // Encode array dimensions in an unsigned integer, returning
  // the integer and the total number of elements in the array.
  static std::pair<ArrayBoundsType, ArrayCountType> EncodeDimensions(
      const unsigned int *p_dimensions, const unsigned int p_dimensionCount);

  // Encode array dimensions in an unsigned integer, returning the
  // integer and the total number of elements in the array.
  static std::pair<ArrayBoundsType, ArrayCountType> EncodeDimensions(
      const ArrayType &p_type);

  // Decode given number of array dimensions (p_numDimensions) from an
  // unsigned integer (p_bounds), returning the the total array element count,
  // and populating p_dimensions (which will be cleared first) with the arrays
  // dimensions.
  static ArrayCountType DecodeDimensions(
      ArrayBoundsType p_bounds, unsigned int p_numDimensions,
      std::vector<unsigned int> &p_dimensions);

  // Issue LLVM code to return an array from a function.
  static LlvmCodeGenerator::CompiledValue &IssueReturn(
      CompilationState &p_state, LlvmCodeGenerator::CompiledValue &p_array,
      const ArrayType &p_arrayType,
      LlvmCodeGenerator::CompiledValue &p_arraySpace);

  // Create an array result from a flattened array and calculated bounds.
  template <typename T>
  static boost::shared_ptr<Result> CreateArrayResult(
      const ArrayType &p_arrayType, ArrayCodeGen::ArrayBoundsType p_bounds,
      const boost::shared_array<T> &p_space);

  // Create an empty array of the given type.
  static LlvmCodeGenerator::CompiledValue &CreateArray(
      CompilationState &p_state, const ArrayType &p_type,
      LlvmCodeGenerator::CompiledValue &p_bounds,
      LlvmCodeGenerator::CompiledValue &p_count,
      LlvmCodeGenerator::CompiledValue &p_pointer);

  // Create an empty array of the given type.
  static LlvmCodeGenerator::CompiledValue &CreateEmptyArray(
      CompilationState &p_state, const ArrayType &p_arrayType);

  // Mask the given bounds to extract the top dimension.
  static LlvmCodeGenerator::CompiledValue &MaskBounds(
      CompilationState &p_state, LlvmCodeGenerator::CompiledValue &p_bounds);

  // Shift the given bounds down to remove one dimension.
  static LlvmCodeGenerator::CompiledValue &ShiftBounds(
      CompilationState &p_state, LlvmCodeGenerator::CompiledValue &p_bounds,
      unsigned int p_dimensions);

  // Push a dimension to the beginning of the bit vector. This is
  // effectively the opposite of ShiftBounds.
  static LlvmCodeGenerator::CompiledValue &UnshiftBound(
      CompilationState &p_state, LlvmCodeGenerator::CompiledValue &p_bounds,
      LlvmCodeGenerator::CompiledValue &p_newBound);
};
};  // namespace FreeForm2

#endif
