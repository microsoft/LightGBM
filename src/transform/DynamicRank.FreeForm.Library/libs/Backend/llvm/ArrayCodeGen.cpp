/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include "ArrayCodeGen.h"

#include <sstream>

#include "ArrayResult.h"
#include "CompilationState.h"
#include "FreeForm2Assert.h"
#include "LlvmCodeGenUtils.h"
#include "ResultIteratorImpl.h"
#include "ValueResult.h"

using namespace FreeForm2;

std::pair<ArrayCodeGen::ArrayBoundsType, ArrayCodeGen::ArrayCountType>
FreeForm2::ArrayCodeGen::EncodeDimensions(const unsigned int *p_dimensions,
                                          const unsigned int p_dimensionCount) {
  ArrayBoundsType encoded = 0;
  ArrayCountType count = 1;
  size_t bitsNeeded = c_bitsPerFlatDimension * p_dimensionCount;
  FF2_ASSERT(bitsNeeded <= sizeof(encoded) * 8);

  // Encode array dimensions into an integer.  Note that we proceed from
  // highest dimension to lowest, so that the first dimension is in the least
  // significant bits.  As we always index into that dimension first, we can
  // then operate by a simple bitshift down on dereference.
  for (unsigned int i = p_dimensionCount; i > 0; --i) {
    if (p_dimensions[i - 1] >= (1 << c_bitsPerFlatDimension)) {
      std::ostringstream err;
      err << "Array (dimensions ";
      for (unsigned int j = p_dimensionCount; j > 0; --j) {
        err << (j != p_dimensionCount ? ", " : "") << p_dimensions[j - 1];
      }
      err << ") exceed the maximum single dimension size of "
          << (1 << c_bitsPerFlatDimension);
      throw std::runtime_error(err.str());
    }

    encoded = encoded << c_bitsPerFlatDimension;
    encoded |= p_dimensions[i - 1];
    count *= p_dimensions[i - 1];
  }

  return std::make_pair(encoded, count);
}

std::pair<ArrayCodeGen::ArrayBoundsType, ArrayCodeGen::ArrayCountType>
FreeForm2::ArrayCodeGen::EncodeDimensions(const ArrayType &p_type) {
  return EncodeDimensions(p_type.GetDimensions(), p_type.GetDimensionCount());
}

ArrayCodeGen::ArrayCountType FreeForm2::ArrayCodeGen::DecodeDimensions(
    ArrayBoundsType p_bounds, unsigned int p_numDimensions,
    std::vector<unsigned int> &p_dimensions) {
  p_dimensions.clear();
  ArrayCountType numElements = 1;
  p_dimensions.assign(p_numDimensions, 0);
  const unsigned int mask = (1 << c_bitsPerFlatDimension) - 1;

  // Decode bounds.
  for (unsigned int i = 0; i < p_numDimensions;
       ++i, p_bounds >>= c_bitsPerFlatDimension) {
    p_dimensions[i] = p_bounds & mask;
    numElements *= p_dimensions[i];
  }

  return numElements;
}

LlvmCodeGenerator::CompiledValue &FreeForm2::ArrayCodeGen::IssueReturn(
    CompilationState &p_state, LlvmCodeGenerator::CompiledValue &p_array,
    const ArrayType &p_arrayType,
    LlvmCodeGenerator::CompiledValue &p_arraySpace) {
  LlvmCodeGenerator::CompiledValue *bounds =
      p_state.GetBuilder().CreateExtractValue(&p_array, boundsPosition);
  CHECK_LLVM_RET(bounds);
  LlvmCodeGenerator::CompiledValue *count =
      p_state.GetBuilder().CreateExtractValue(&p_array, countPosition);
  CHECK_LLVM_RET(count);
  LlvmCodeGenerator::CompiledValue *pointer =
      p_state.GetBuilder().CreateExtractValue(&p_array, pointerPosition);
  CHECK_LLVM_RET(pointer);

  // Calculate number of bytes per element.
  const TypeImpl &childType = p_arrayType.GetChildType().AsConstType();
  unsigned int bytes = p_state.GetSizeInBytes(&p_state.GetType(childType));

  LlvmCodeGenerator::CompiledValue *elementSize = llvm::ConstantInt::get(
      p_state.GetContext(), llvm::APInt(sizeof(ArrayBoundsType) * 8, bytes));
  CHECK_LLVM_RET(elementSize);

  LlvmCodeGenerator::CompiledValue *copyBytes =
      p_state.GetBuilder().CreateMul(elementSize, count);
  CHECK_LLVM_RET(copyBytes);

  // Copy array into provided space.  Note that first arg is
  // destination, second is source, third is number of bytes to copy,
  // final is guaranteed alignment (which we could optimise by
  // guaranteeing and providing a higher value).
  p_state.GetBuilder().CreateMemCpy(&p_arraySpace, pointer, copyBytes, 0);

  return *bounds;
}

template <typename T>
boost::shared_ptr<FreeForm2::Result> FreeForm2::ArrayCodeGen::CreateArrayResult(
    const ArrayType &p_arrayType, ArrayCodeGen::ArrayBoundsType p_bounds,
    const boost::shared_array<T> &p_space) {
  // Decode bounds.
  SharedDimensions dimensions(new std::vector<unsigned int>());
  ArrayCodeGen::DecodeDimensions(p_bounds, p_arrayType.GetDimensionCount(),
                                 *dimensions);
  return boost::shared_ptr<Result>(
      new ArrayResult<T>(p_arrayType, 0, dimensions, p_space.get(), p_space));
}

// Instantiate CreateArrayResult for needed types.
template boost::shared_ptr<FreeForm2::Result>
FreeForm2::ArrayCodeGen::CreateArrayResult<bool>(
    const ArrayType &p_arrayType, ArrayCodeGen::ArrayBoundsType p_bounds,
    const boost::shared_array<bool> &p_space);

template boost::shared_ptr<FreeForm2::Result>
FreeForm2::ArrayCodeGen::CreateArrayResult<Result::IntType>(
    const ArrayType &p_arrayType, ArrayCodeGen::ArrayBoundsType p_bounds,
    const boost::shared_array<Result::IntType> &p_space);

template boost::shared_ptr<FreeForm2::Result>
FreeForm2::ArrayCodeGen::CreateArrayResult<Result::FloatType>(
    const ArrayType &p_arrayType, ArrayCodeGen::ArrayBoundsType p_bounds,
    const boost::shared_array<Result::FloatType> &p_space);

LlvmCodeGenerator::CompiledValue &FreeForm2::ArrayCodeGen::CreateArray(
    CompilationState &p_state, const ArrayType &p_arrayType,
    LlvmCodeGenerator::CompiledValue &p_bounds,
    LlvmCodeGenerator::CompiledValue &p_count,
    LlvmCodeGenerator::CompiledValue &p_pointer) {
  // Calculate array bounds.
  llvm::Type &arrayType = p_state.GetType(p_arrayType);

  // Create structure with calculated bounds.
  LlvmCodeGenerator::CompiledValue *undef = llvm::UndefValue::get(&arrayType);
  CHECK_LLVM_RET(undef);
  LlvmCodeGenerator::CompiledValue *structure =
      p_state.GetBuilder().CreateInsertValue(undef, &p_bounds, boundsPosition);
  CHECK_LLVM_RET(structure);

  // Add calculated element count.
  structure = p_state.GetBuilder().CreateInsertValue(structure, &p_count,
                                                     countPosition);
  CHECK_LLVM_RET(structure);

  structure = p_state.GetBuilder().CreateInsertValue(structure, &p_pointer,
                                                     pointerPosition);
  CHECK_LLVM_RET(structure);

  return *structure;
}

LlvmCodeGenerator::CompiledValue &FreeForm2::ArrayCodeGen::CreateEmptyArray(
    CompilationState &p_state, const ArrayType &p_arrayType) {
  LlvmCodeGenerator::CompiledValue *boundsZero = llvm::ConstantInt::get(
      p_state.GetContext(), llvm::APInt(sizeof(ArrayBoundsType) * 8, 0));
  CHECK_LLVM_RET(boundsZero);

  LlvmCodeGenerator::CompiledValue *countZero = llvm::ConstantInt::get(
      p_state.GetContext(), llvm::APInt(sizeof(ArrayCountType) * 8, 0));
  CHECK_LLVM_RET(countZero);

  llvm::Type &childType =
      p_state.GetType(p_arrayType.GetChildType().AsConstType());
  llvm::Type *nullType = llvm::PointerType::get(&childType, 0);
  CHECK_LLVM_RET(nullType);
  LlvmCodeGenerator::CompiledValue *pointer =
      llvm::Constant::getNullValue(nullType);
  CHECK_LLVM_RET(pointer);

  return CreateArray(p_state, p_arrayType, *boundsZero, *countZero, *pointer);
}

LlvmCodeGenerator::CompiledValue &FreeForm2::ArrayCodeGen::MaskBounds(
    CompilationState &p_state, LlvmCodeGenerator::CompiledValue &p_bounds) {
  LlvmCodeGenerator::CompiledValue *mask = llvm::ConstantInt::get(
      p_state.GetContext(), llvm::APInt(sizeof(ArrayBoundsType) * 8,
                                        (1 << c_bitsPerFlatDimension) - 1));
  CHECK_LLVM_RET(mask);
  LlvmCodeGenerator::CompiledValue *masked =
      p_state.GetBuilder().CreateAnd(&p_bounds, mask);
  CHECK_LLVM_RET(masked);
  return *masked;
}

LlvmCodeGenerator::CompiledValue &FreeForm2::ArrayCodeGen::ShiftBounds(
    CompilationState &p_state, LlvmCodeGenerator::CompiledValue &p_bounds,
    unsigned int p_dimensions) {
  LlvmCodeGenerator::CompiledValue *shift = llvm::ConstantInt::get(
      p_state.GetContext(), llvm::APInt(sizeof(ArrayBoundsType) * 8,
                                        c_bitsPerFlatDimension * p_dimensions));
  CHECK_LLVM_RET(shift);
  LlvmCodeGenerator::CompiledValue *shifted =
      p_state.GetBuilder().CreateAShr(&p_bounds, shift);
  CHECK_LLVM_RET(shifted);
  return *shifted;
}

LlvmCodeGenerator::CompiledValue &FreeForm2::ArrayCodeGen::UnshiftBound(
    CompilationState &p_state, LlvmCodeGenerator::CompiledValue &p_bounds,
    LlvmCodeGenerator::CompiledValue &p_newBound) {
  llvm::Value *bitCount =
      llvm::ConstantInt::get(p_bounds.getType(), c_bitsPerFlatDimension);
  CHECK_LLVM_RET(bitCount);

  llvm::Value *leftShift = p_state.GetBuilder().CreateShl(&p_bounds, bitCount);
  CHECK_LLVM_RET(leftShift);

  llvm::Value *maxBound = llvm::ConstantInt::get(
      p_newBound.getType(), (1 << c_bitsPerFlatDimension) - 1);
  CHECK_LLVM_RET(maxBound);

  llvm::Value *checkBound =
      p_state.GetBuilder().CreateICmpUGT(&p_newBound, maxBound);
  CHECK_LLVM_RET(checkBound);

  llvm::Value *realBound =
      p_state.GetBuilder().CreateSelect(checkBound, maxBound, &p_newBound);
  CHECK_LLVM_RET(realBound);

  if (realBound->getType()->getPrimitiveSizeInBits() <
      p_bounds.getType()->getPrimitiveSizeInBits()) {
    realBound = p_state.GetBuilder().CreateZExt(realBound, p_bounds.getType());
    CHECK_LLVM_RET(realBound);
  }

  llvm::Value *finalBounds =
      p_state.GetBuilder().CreateOr(leftShift, realBound);
  CHECK_LLVM_RET(finalBounds);
  return *finalBounds;
}
