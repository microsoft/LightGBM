/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include "LlvmCodeGenerator.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

#include <queue>
#include <sstream>

#include "Allocation.h"
#include "ArrayCodeGen.h"
#include "ArrayDereferenceExpression.h"
#include "ArrayLength.h"
#include "ArrayLiteralExpression.h"
#include "BinaryOperator.h"
#include "BlockExpression.h"
#include "Conditional.h"
#include "ConvertExpression.h"
#include "Declaration.h"
#include "FeatureSpec.h"
#include "FreeForm2.h"
#include "FreeForm2Assert.h"
#include "LetExpression.h"
#include "LiteralExpression.h"
#include "LlvmCodeGenUtils.h"
#include "OperatorExpression.h"
#include "RandExpression.h"
#include "RangeReduceExpression.h"
#include "RefExpression.h"
#include "SelectNth.h"
#include "UnaryOperator.h"

using namespace FreeForm2;

namespace {
LlvmCodeGenerator::CompiledValue &CompileFloatEquality(
    CompilationState &m_state, LlvmCodeGenerator::CompiledValue &p_left,
    LlvmCodeGenerator::CompiledValue &p_right) {
  llvm::IRBuilder<> &builder = m_state.GetBuilder();

  llvm::Type *type = p_left.getType();
  CHECK_LLVM_RET(type);
  FF2_ASSERT(p_right.getType() &&
             type->getTypeID() == p_right.getType()->getTypeID());

  LlvmCodeGenerator::CompiledValue *small = llvm::ConstantFP::get(type, 10E-9);
  CHECK_LLVM_RET(small);
  LlvmCodeGenerator::CompiledValue *negSmall =
      llvm::ConstantFP::get(type, -10E-9);
  CHECK_LLVM_RET(negSmall);

  // Determine whether the right expression is small.
  LlvmCodeGenerator::CompiledValue *rightSmallCmp =
      builder.CreateFCmp(llvm::CmpInst::FCMP_OLT, &p_right, small);
  CHECK_LLVM_RET(rightSmallCmp);
  LlvmCodeGenerator::CompiledValue *rightNegSmallCmp =
      builder.CreateFCmp(llvm::CmpInst::FCMP_OGT, &p_right, negSmall);
  CHECK_LLVM_RET(rightNegSmallCmp);
  LlvmCodeGenerator::CompiledValue *rightIsSmall =
      builder.CreateAnd(rightSmallCmp, rightNegSmallCmp);
  CHECK_LLVM_RET(rightIsSmall);

  GenerateConditional cond(m_state, *rightIsSmall,
                           "Approximate fp cmp: right small?");

  // Determine whether the left expression is small.
  LlvmCodeGenerator::CompiledValue *leftSmallCmp =
      builder.CreateFCmp(llvm::CmpInst::FCMP_OLT, &p_left, small);
  CHECK_LLVM_RET(leftSmallCmp);
  LlvmCodeGenerator::CompiledValue *leftNegSmallCmp =
      builder.CreateFCmp(llvm::CmpInst::FCMP_OGT, &p_left, negSmall);
  CHECK_LLVM_RET(leftNegSmallCmp);
  LlvmCodeGenerator::CompiledValue *leftIsSmall =
      builder.CreateAnd(leftSmallCmp, leftNegSmallCmp);
  CHECK_LLVM_RET(leftIsSmall);
  cond.FinishThen(leftIsSmall);

  // Determine whether the difference between left and right is small.
  LlvmCodeGenerator::CompiledValue *diff =
      builder.CreateFSub(&p_left, &p_right);
  CHECK_LLVM_RET(diff);
  LlvmCodeGenerator::CompiledValue *normDiff =
      builder.CreateFDiv(diff, &p_right);
  CHECK_LLVM_RET(normDiff);
  LlvmCodeGenerator::CompiledValue *diffSmallCmp =
      builder.CreateFCmp(llvm::CmpInst::FCMP_OLT, normDiff, small);
  CHECK_LLVM_RET(diffSmallCmp);
  LlvmCodeGenerator::CompiledValue *diffNegSmallCmp =
      builder.CreateFCmp(llvm::CmpInst::FCMP_OGT, normDiff, negSmall);
  CHECK_LLVM_RET(diffNegSmallCmp);
  LlvmCodeGenerator::CompiledValue *diffIsSmall =
      builder.CreateAnd(diffSmallCmp, diffNegSmallCmp);
  CHECK_LLVM_RET(diffIsSmall);
  cond.FinishElse(diffIsSmall);

  return cond.Finish(m_state.GetBoolType());
}

// Create a value which will evaluate to a random number in the range [0, 1.0].
llvm::Value &CreateRandomFloat(CompilationState &p_state,
                               llvm::Type *p_floatType) {
  llvm::IRBuilder<> &builder = p_state.GetBuilder();

  llvm::Function *rand =
      p_state.GetRuntimeLibrary().FindFunction(CStackSizedString("rand"));
  CHECK_LLVM_RET(rand);

  llvm::Value *ret = builder.CreateCall(rand);
  CHECK_LLVM_RET(ret);

  if (p_floatType && p_floatType->getPrimitiveSizeInBits() <
                         rand->getReturnType()->getPrimitiveSizeInBits()) {
    ret = builder.CreateFPTrunc(ret, p_floatType);
    CHECK_LLVM_RET(ret);
  } else if (p_floatType &&
             p_floatType->getPrimitiveSizeInBits() >
                 rand->getReturnType()->getPrimitiveSizeInBits()) {
    ret = builder.CreateFPExt(ret, p_floatType);
    CHECK_LLVM_RET(ret);
  }

  return *ret;
}

}  // namespace

FreeForm2::LlvmCodeGenerator::LlvmCodeGenerator(
    CompilationState &p_state, const AllocationVector &p_allocations,
    CompilerFactory::DestinationFunctionType p_destinationFunctionType)
    : m_state(p_state),
      m_allocations(p_allocations),
      m_destinationFunctionType(p_destinationFunctionType),
      m_returnType(nullptr),
      m_returnValue(nullptr),
      m_function(nullptr) {}

LlvmCodeGenerator::CompiledValue *FreeForm2::LlvmCodeGenerator::GetResult() {
  return m_stack.top();
}

llvm::Function *FreeForm2::LlvmCodeGenerator::GetFuction() const {
  return m_function;
}

void FreeForm2::LlvmCodeGenerator::Visit(const SelectNthExpression &p_expr) {
  // Ensure that the number of selection elements can be represented in an
  // Int type.
  const llvm::APInt maxInt =
      llvm::APInt::getSignedMaxValue(m_state.GetIntBits());
  const size_t sizetBitSize = sizeof(size_t) * 8;
  llvm::APInt numChildren(sizetBitSize, p_expr.GetNumChildren() - 1);
  if (m_state.GetIntBits() < sizetBitSize) {
    llvm::APInt trunc = numChildren.trunc(m_state.GetIntBits());
    FF2_ASSERT(trunc.getActiveBits() == numChildren.getActiveBits());
    numChildren = std::move(trunc);
  }

  // Check that the unsigned Int version of numChildren is no larger than the
  // max signed Int.
  FF2_ASSERT(maxInt.uge(numChildren));

  CompiledValue *index = m_stack.top();
  m_stack.pop();

  FF2_ASSERT(p_expr.GetIndex().GetType().IsIntegerType());

  // Promote index to an Int type.
  index = &ValueConversion::Do(*index, p_expr.GetIndex().GetType(),
                               TypeImpl::GetIntInstance(true), m_state);

  CompiledValue *high = m_stack.top();
  m_stack.pop();

  CompiledValue *select = high;

  // Loop backward through all children that aren't the
  // last, creating a linear chained select.  Note that this
  // handles the single-child case by simply returning the
  // value of that child.
  for (llvm::APInt i(m_state.GetIntBits(), 1); i.slt(numChildren); ++i) {
    CompiledValue *current = m_stack.top();
    m_stack.pop();

    const llvm::APInt child = numChildren - i - 1;
    CompiledValue *childValue =
        llvm::ConstantInt::get(&m_state.GetIntType(), child);
    CHECK_LLVM_RET(childValue);
    CompiledValue *cond = m_state.GetBuilder().CreateICmpSLE(index, childValue);
    CHECK_LLVM_RET(cond);
    select = m_state.GetBuilder().CreateSelect(cond, current, select);
    CHECK_LLVM_RET(select);
  }

  // Check whether index is out-of-bounds.  We use two's-complement-based
  // cleverness to do this, by doing an unsigned comparison to the length of
  // the array, and relying on two's complement representation to make
  // negative numbers larger (in unsigned values) than the maximum index.
  CompiledValue *bounds =
      llvm::ConstantInt::get(&m_state.GetIntType(), numChildren);
  CHECK_LLVM_RET(bounds);
  CompiledValue *inBounds = m_state.GetBuilder().CreateICmpULT(index, bounds);
  CHECK_LLVM_RET(inBounds);

  // Select between value created before (in-bounds) and zero value
  // (out-of-bounds).
  CompiledValue *boundsSelect = m_state.GetBuilder().CreateSelect(
      inBounds, select, &m_state.CreateZeroValue(p_expr.GetType()));
  CHECK_LLVM_RET(boundsSelect);

  m_stack.push(boundsSelect);
}

void FreeForm2::LlvmCodeGenerator::Visit(const SelectRangeExpression &p_expr) {
  llvm::IRBuilder<> &builder = m_state.GetBuilder();
  CompiledValue *array = m_stack.top();
  m_stack.pop();

  CompiledValue *count = m_stack.top();
  m_stack.pop();

  CompiledValue *start = m_stack.top();
  m_stack.pop();

  FF2_ASSERT(p_expr.GetType().Primitive() == Type::Array);
  const ArrayType &destType = static_cast<const ArrayType &>(p_expr.GetType());

  CompiledValue *bounds =
      builder.CreateExtractValue(array, ArrayCodeGen::boundsPosition);
  CHECK_LLVM_RET(bounds);

  CompiledValue &dimensionBound = ArrayCodeGen::MaskBounds(m_state, *bounds);

  // Guard against a start value that is past the end of the array or less
  // than zero. The less-than-zero condition is checked implicitly by doing
  // unsigned comparison (start would become larger than dimensionBound).
  CompiledValue *validateStart = builder.CreateICmpULT(start, &dimensionBound);
  CHECK_LLVM_RET(validateStart);

  CompiledValue *countZero = llvm::ConstantInt::get(count->getType(), 0);
  CHECK_LLVM_RET(countZero);

  // Guard against a count value that is less than or equal to zero. If the
  // count falls into this range, the returned array should be empty.
  CompiledValue *validateCount = builder.CreateICmpSGT(count, countZero);
  CHECK_LLVM_RET(validateCount);

  CompiledValue *guardCondition =
      builder.CreateAnd(validateStart, validateCount);
  CHECK_LLVM_RET(guardCondition);

  GenerateConditional startGuard(m_state, *guardCondition,
                                 "SelectRange start guard");

  CompiledValue *srcCount =
      builder.CreateExtractValue(array, ArrayCodeGen::countPosition);
  CHECK_LLVM_RET(srcCount);

  CompiledValue *subArrayCount = builder.CreateUDiv(srcCount, &dimensionBound);
  CHECK_LLVM_RET(subArrayCount);

  // Compute the new pointer to the first element.
  CompiledValue *oldPtr =
      builder.CreateExtractValue(array, ArrayCodeGen::pointerPosition);
  CHECK_LLVM_RET(oldPtr);

  CompiledValue *startElemIndex = builder.CreateMul(subArrayCount, start);
  CHECK_LLVM_RET(startElemIndex);

  CompiledValue *newPtr = builder.CreateInBoundsGEP(oldPtr, startElemIndex);
  CHECK_LLVM_RET(newPtr);

  // Correct the count if necessary.
  CompiledValue *maxCount = builder.CreateSub(&dimensionBound, start);
  CHECK_LLVM_RET(maxCount);

  CompiledValue *checkCount = builder.CreateICmpSGT(count, maxCount);
  CHECK_LLVM_RET(checkCount);

  CompiledValue *correctCount =
      builder.CreateSelect(checkCount, maxCount, count);
  CHECK_LLVM_RET(correctCount);

  // Get the new bounds bit vector.
  CompiledValue &subArrayBounds =
      ArrayCodeGen::ShiftBounds(m_state, *bounds, 1);

  CompiledValue &newBounds =
      ArrayCodeGen::UnshiftBound(m_state, subArrayBounds, *correctCount);

  // Get the new array count.
  CompiledValue *newCount = builder.CreateMul(subArrayCount, correctCount);
  CHECK_LLVM_RET(newCount);

  CompiledValue &newArray = ArrayCodeGen::CreateArray(
      m_state, destType, newBounds, *newCount, *newPtr);

  startGuard.FinishThen(&newArray);

  startGuard.FinishElse(&ArrayCodeGen::CreateEmptyArray(m_state, destType));

  CompiledValue &returnVal = startGuard.Finish(*newArray.getType());
  m_stack.push(&returnVal);
}

void FreeForm2::LlvmCodeGenerator::Visit(const ArrayLengthExpression &p_expr) {
  CompiledValue *array = m_stack.top();
  m_stack.pop();

  CompiledValue *bounds = m_state.GetBuilder().CreateExtractValue(
      array, ArrayCodeGen::boundsPosition);
  CHECK_LLVM_RET(bounds);

  // Length of the array we're dereferencing is in the least significant bits.
  CompiledValue &length = ArrayCodeGen::MaskBounds(m_state, *bounds);
  FF2_ASSERT(length.getType() && length.getType()->isIntegerTy());

  llvm::Type &retType = m_state.GetType(p_expr.GetType());
  FF2_ASSERT(retType.isIntegerTy());

  // Truncate the 64-bit return value to the return size, which should be 32
  // bits.
  FF2_ASSERT(retType.getPrimitiveSizeInBits() <
             length.getType()->getPrimitiveSizeInBits());
  CompiledValue *ret = m_state.GetBuilder().CreateTrunc(&length, &retType);
  CHECK_LLVM_RET(ret);

  m_stack.push(ret);
}

void FreeForm2::LlvmCodeGenerator::Visit(
    const ArrayDereferenceExpression &p_expr) {
  LlvmCodeGenerator::CompiledValue *index = m_stack.top();
  m_stack.pop();

  LlvmCodeGenerator::CompiledValue *array = m_stack.top();
  m_stack.pop();

  LlvmCodeGenerator::CompiledValue *bounds =
      m_state.GetBuilder().CreateExtractValue(array,
                                              ArrayCodeGen::boundsPosition);
  CHECK_LLVM_RET(bounds);

  // Length of the array we're dereferencing is in the least significant bits.
  LlvmCodeGenerator::CompiledValue &arrayLength =
      ArrayCodeGen::MaskBounds(m_state, *bounds);

  // Check whether index is out-of-bounds.  We use two's-complement-based
  // cleverness to do this, by doing an unsigned comparison to the length of
  // the array, and relying on two's complement representation to make
  // negative numbers larger (in unsigned values) than the maximum index.

  // Compare index to length.
  LlvmCodeGenerator::CompiledValue *inBounds =
      m_state.GetBuilder().CreateICmpULT(index, &arrayLength);
  CHECK_LLVM_RET(inBounds);

  GenerateConditional inBoundsCase(m_state, *inBounds,
                                   "array dereference guard");

  LlvmCodeGenerator::CompiledValue *val = NULL;
  LlvmCodeGenerator::CompiledValue *pointer =
      m_state.GetBuilder().CreateExtractValue(array,
                                              ArrayCodeGen::pointerPosition);
  CHECK_LLVM_RET(pointer);
  if (p_expr.GetType().Primitive() == Type::Array) {
    const ArrayType &arrayType =
        static_cast<const ArrayType &>(p_expr.GetType());

    LlvmCodeGenerator::CompiledValue &resultBounds =
        ArrayCodeGen::ShiftBounds(m_state, *bounds, 1);

    // The count of the sub-array being dereferenced is the count of the
    // current array divided by the bounds of the current array. This is
    // used to compute the offset of the base of the sub-array.
    CompiledValue *arrayCount = m_state.GetBuilder().CreateExtractValue(
        array, ArrayCodeGen::countPosition);
    CHECK_LLVM_RET(arrayCount);

    CompiledValue *subArrayCount =
        m_state.GetBuilder().CreateUDiv(arrayCount, &arrayLength);
    CHECK_LLVM_RET(subArrayCount);

    CompiledValue *indexSelect =
        m_state.GetBuilder().CreateMul(subArrayCount, index);
    CHECK_LLVM_RET(indexSelect);

    LlvmCodeGenerator::CompiledValue *elementPtr =
        m_state.GetBuilder().CreateInBoundsGEP(pointer, indexSelect);
    CHECK_LLVM_RET(elementPtr);

    val = &ArrayCodeGen::CreateArray(m_state, arrayType, resultBounds,
                                     *subArrayCount, *elementPtr);
  } else {
    // Lookup index into pointer.
    LlvmCodeGenerator::CompiledValue *elementPtr =
        m_state.GetBuilder().CreateInBoundsGEP(pointer, index);
    CHECK_LLVM_RET(elementPtr);

    val = m_state.GetBuilder().CreateLoad(elementPtr);
  }
  CHECK_LLVM_RET(val);

  inBoundsCase.FinishThen(val);

  // If the index wasn't in bounds, give back zero.
  inBoundsCase.FinishElse(&m_state.CreateZeroValue(p_expr.GetType()));
  llvm::Value *ret = &inBoundsCase.Finish(m_state.GetType(p_expr.GetType()));

  m_stack.push(ret);
}

void FreeForm2::LlvmCodeGenerator::VisitReference(
    const ArrayDereferenceExpression &p_expr) {
  // References are not supported in FF2.
  FF2_UNREACHABLE();
}

void FreeForm2::LlvmCodeGenerator::Allocate(const Allocation &p_allocation) {
  switch (p_allocation.GetAllocationType()) {
    case Allocation::ArrayLiteral: {
      FF2_ASSERT(p_allocation.GetType().Primitive() == Type::Array);
      const ArrayType &arrayInfo =
          static_cast<const ArrayType &>(p_allocation.GetType());

      PreAllocatedArray preAllocatedArray;

      // Calculate array bounds.
      std::pair<ArrayCodeGen::ArrayBoundsType, ArrayCodeGen::ArrayCountType>
          encoded = ArrayCodeGen::EncodeDimensions(arrayInfo);
      preAllocatedArray.m_bounds = llvm::ConstantInt::get(
          m_state.GetContext(),
          llvm::APInt(sizeof(encoded.first) * 8, encoded.first));
      CHECK_LLVM_RET(preAllocatedArray.m_bounds);

      // Add calculated element count.
      FF2_ASSERT(encoded.second == arrayInfo.GetMaxElements());
      FF2_ASSERT(encoded.second == p_allocation.GetNumChildren());
      preAllocatedArray.m_count = llvm::ConstantInt::get(
          m_state.GetContext(),
          llvm::APInt(sizeof(encoded.second) * 8, encoded.second));
      CHECK_LLVM_RET(preAllocatedArray.m_count);

      preAllocatedArray.m_array = NULL;
      llvm::Type &elementType = m_state.GetType(arrayInfo.GetChildType());
      if (arrayInfo.GetMaxElements() > 0) {
        // Add array pointer to structure.
        LlvmCodeGenerator::CompiledValue *arraySize = llvm::ConstantInt::get(
            m_state.GetContext(), llvm::APInt(sizeof(encoded.second) * 8,
                                              arrayInfo.GetMaxElements()));
        CHECK_LLVM_RET(arraySize);
        preAllocatedArray.m_array =
            m_state.GetBuilder().CreateAlloca(&elementType, arraySize);
      } else {
        // Handle zero-length case by assigning NULL, as alloca with zero bytes
        // invokes undefined LLVM behaviour.
        llvm::Type *nullType = llvm::PointerType::get(&elementType, 0);
        CHECK_LLVM_RET(nullType);
        preAllocatedArray.m_array = llvm::Constant::getNullValue(nullType);
      }
      CHECK_LLVM_RET(preAllocatedArray.m_array);

      m_allocatedArrays.insert(
          std::make_pair(p_allocation.GetAllocationId(), preAllocatedArray));

      break;
    }

    // Declarations not implemented in FF2.
    case Allocation::Declaration:
      __attribute__((__fallthrough__));

    // Literal streams not implemented in FF2
    case Allocation::LiteralStream:
      __attribute__((__fallthrough__));

    default: {
      FF2_UNREACHABLE();
    }
  }
}

void FreeForm2::LlvmCodeGenerator::Visit(const ArrayLiteralExpression &p_expr) {
  FF2_ASSERT(m_allocatedArrays.find(p_expr.GetId()) != m_allocatedArrays.end());
  FF2_ASSERT(p_expr.GetType().Primitive() == Type::Array);

  PreAllocatedArray preAllocatedArray = m_allocatedArrays[p_expr.GetId()];

  // Populate structure with values.
  for (unsigned int i = 0; i < p_expr.GetNumChildren(); i++) {
    LlvmCodeGenerator::CompiledValue *element =
        m_state.GetBuilder().CreateConstInBoundsGEP1_32(
            preAllocatedArray.m_array, i);
    CHECK_LLVM_RET(element);
    m_state.GetBuilder().CreateStore(m_stack.top(), element);
    m_stack.pop();
  }

  m_stack.push(&ArrayCodeGen::CreateArray(
      m_state, static_cast<const ArrayType &>(p_expr.GetType()),
      *preAllocatedArray.m_bounds, *preAllocatedArray.m_count,
      *preAllocatedArray.m_array));
}

bool FreeForm2::LlvmCodeGenerator::AlternativeVisit(
    const ConditionalExpression &p_expr) {
  p_expr.GetCondition().Accept(*this);
  GenerateConditional cond(m_state, *m_stack.top(), "if");
  m_stack.pop();

  p_expr.GetThen().Accept(*this);
  cond.FinishThen(m_stack.top());
  m_stack.pop();

  p_expr.GetElse().Accept(*this);
  cond.FinishElse(m_stack.top());
  m_stack.pop();

  m_stack.push(&cond.Finish(m_state.GetType(p_expr.GetType())));

  return true;
}

void FreeForm2::LlvmCodeGenerator::Visit(const ConditionalExpression &p_expr) {
  // This should have been managed by AlternativeVisit.
  Unreachable(__FILE__, __LINE__);
}

void FreeForm2::LlvmCodeGenerator::Visit(
    const ConvertToFloatExpression &p_expr) {
  CompiledValue *child = m_stack.top();
  m_stack.pop();

  m_stack.push(&ValueConversion::Do(*child, p_expr.GetChildType(),
                                    p_expr.GetType(), m_state));
}

void FreeForm2::LlvmCodeGenerator::Visit(const ConvertToIntExpression &p_expr) {
  CompiledValue *child = m_stack.top();
  m_stack.pop();

  m_stack.push(&ValueConversion::Do(*child, p_expr.GetChildType(),
                                    p_expr.GetType(), m_state));
}

void FreeForm2::LlvmCodeGenerator::Visit(
    const ConvertToUInt64Expression &p_expr) {
  // Not supported in FF2.
  FF2_UNREACHABLE();
}

void FreeForm2::LlvmCodeGenerator::Visit(
    const ConvertToInt32Expression &p_expr) {
  CompiledValue *child = m_stack.top();
  m_stack.pop();

  m_stack.push(&ValueConversion::Do(*child, p_expr.GetChildType(),
                                    p_expr.GetType(), m_state));
}

void FreeForm2::LlvmCodeGenerator::Visit(
    const ConvertToUInt32Expression &p_expr) {
  CompiledValue *child = m_stack.top();
  m_stack.pop();

  m_stack.push(&ValueConversion::Do(*child, p_expr.GetChildType(),
                                    p_expr.GetType(), m_state));
}

void FreeForm2::LlvmCodeGenerator::Visit(
    const ConvertToBoolExpression &p_expr) {
  LlvmCodeGenerator::CompiledValue *child = m_stack.top();
  m_stack.pop();

  m_stack.push(&ValueConversion::Do(*child, p_expr.GetChildType(),
                                    p_expr.GetType(), m_state));
}

void FreeForm2::LlvmCodeGenerator::Visit(
    const ConvertToImperativeExpression &) {
  // Not supported in FF2.
  FF2_UNREACHABLE();
}

void FreeForm2::LlvmCodeGenerator::Visit(const DeclarationExpression &) {
  // Not supported in FF2.
  FF2_UNREACHABLE();
}

void FreeForm2::LlvmCodeGenerator::Visit(const DirectPublishExpression &) {
  // Not supported in FF2.
  FF2_UNREACHABLE();
}

void FreeForm2::LlvmCodeGenerator::Visit(const ExternExpression &) {
  // Not supported in FF2.
  FF2_UNREACHABLE();
}

void FreeForm2::LlvmCodeGenerator::Visit(const LiteralIntExpression &p_expr) {
  const llvm::Type &returnType = m_state.GetType(p_expr.GetType());
  FF2_ASSERT(returnType.isIntegerTy());
  const llvm::APInt value(returnType.getPrimitiveSizeInBits(),
                          static_cast<UInt64>(p_expr.GetConstantValue().m_int),
                          true);
  CompiledValue *ret = llvm::ConstantInt::get(m_state.GetContext(), value);
  CHECK_LLVM_RET(ret);
  m_stack.push(ret);
}

void FreeForm2::LlvmCodeGenerator::Visit(const LiteralUInt64Expression &) {
  // Not supported in FF2.
  FF2_UNREACHABLE();
}

void FreeForm2::LlvmCodeGenerator::Visit(const LiteralInt32Expression &p_expr) {
  const llvm::Type &returnType = m_state.GetType(p_expr.GetType());
  FF2_ASSERT(returnType.isIntegerTy());

  // Sign-extend the value to 64-bits; direct cast to UInt64 would lose data.
  const Int64 intValue = p_expr.GetConstantValue().m_int32;
  const llvm::APInt value(returnType.getPrimitiveSizeInBits(),
                          static_cast<UInt64>(intValue), true);
  CompiledValue *ret = llvm::ConstantInt::get(m_state.GetContext(), value);
  CHECK_LLVM_RET(ret);
  m_stack.push(ret);
}

void FreeForm2::LlvmCodeGenerator::Visit(
    const LiteralUInt32Expression &p_expr) {
  const llvm::Type &returnType = m_state.GetType(p_expr.GetType());
  FF2_ASSERT(returnType.isIntegerTy());
  const llvm::APInt value(returnType.getPrimitiveSizeInBits(),
                          p_expr.GetConstantValue().m_int, false);
  CompiledValue *ret = llvm::ConstantInt::get(m_state.GetContext(), value);
  CHECK_LLVM_RET(ret);
  m_stack.push(ret);
}

void FreeForm2::LlvmCodeGenerator::Visit(const LiteralFloatExpression &p_expr) {
  CompiledValue *ret = llvm::ConstantFP::get(
      m_state.GetContext(), llvm::APFloat(p_expr.GetConstantValue().m_float));
  CHECK_LLVM_RET(ret);
  m_stack.push(ret);
}

void FreeForm2::LlvmCodeGenerator::Visit(const LiteralBoolExpression &p_expr) {
  CompiledValue *ret = llvm::ConstantInt::get(
      m_state.GetContext(),
      llvm::APInt(1, p_expr.GetConstantValue().m_bool ? 1 : 0));
  CHECK_LLVM_RET(ret);
  m_stack.push(ret);
}

void FreeForm2::LlvmCodeGenerator::Visit(const LiteralVoidExpression &) {
  // Not supported in FF2.
  FF2_UNREACHABLE();
}

void FreeForm2::LlvmCodeGenerator::Visit(const LiteralStreamExpression &) {
  // Not supported in FF2.
  FF2_UNREACHABLE();
}

void FreeForm2::LlvmCodeGenerator::Visit(const LiteralWordExpression &) {
  // Not supported in FF2.
  FF2_UNREACHABLE();
}

void FreeForm2::LlvmCodeGenerator::Visit(
    const LiteralInstanceHeaderExpression &) {
  // Not supported in FF2.
  FF2_UNREACHABLE();
}

bool FreeForm2::LlvmCodeGenerator::AlternativeVisit(
    const LetExpression &p_expr) {
  for (size_t i = 0; i + 1 < p_expr.GetNumChildren(); i++) {
    p_expr.GetBound()[i].second->Accept(*this);

    LlvmCodeGenerator::CompiledValue *value = m_stack.top();
    m_stack.pop();

    // Store value for later use.
    m_state.SetVariableValue(p_expr.GetBound()[i].first, *value);
  }

  p_expr.GetValue().Accept(*this);

  return true;
}

void FreeForm2::LlvmCodeGenerator::Visit(const LetExpression &) {
  // Handled by AlternativeVisit.
  Unreachable(__FILE__, __LINE__);
}

bool FreeForm2::LlvmCodeGenerator::AlternativeVisit(
    const BlockExpression &p_expr) {
  const auto children = static_cast<unsigned int>(p_expr.GetNumChildren());
  FF2_ASSERT(children == p_expr.GetNumChildren());

  for (unsigned int i = 0; i < children; i++) {
    p_expr.GetChild(i).Accept(*this);
  }

  // Block expressions visit children from top-to-bottom.  This means that
  // the last child is on top of the stack.
  LlvmCodeGenerator::CompiledValue &ret = *m_stack.top();

  // Remove all other values from the stack.
  for (unsigned int i = 0; i < children; i++) {
    m_stack.pop();
  }

  // Return result value to the stack.
  m_stack.push(&ret);

  return true;
}

void FreeForm2::LlvmCodeGenerator::Visit(const BlockExpression &) {
  // This should have been managed by AlternativeVisit.
  FF2_UNREACHABLE();
}

void FreeForm2::LlvmCodeGenerator::Visit(const MutationExpression &) {
  // Not supported in FF2.
  FF2_UNREACHABLE();
}

void FreeForm2::LlvmCodeGenerator::Visit(const MatchExpression &) {
  // Not supported in FF2.
  FF2_UNREACHABLE();
}

// void
// FreeForm2::LlvmCodeGenerator::Visit(const MatchWordExpression&)
// {
//     // Not supported in FF2.
//     FF2_UNREACHABLE();
// }

// void
// FreeForm2::LlvmCodeGenerator::Visit(const MatchLiteralExpression&)
// {
//     // Not supported in FF2.
//     FF2_UNREACHABLE();
// }

// void
// FreeForm2::LlvmCodeGenerator::Visit(const MatchCurrentWordExpression&)
// {
//     // Not supported in FF2.
//     FF2_UNREACHABLE();
// }

void FreeForm2::LlvmCodeGenerator::Visit(const MatchOperatorExpression &) {
  // Not supported in FF2.
  FF2_UNREACHABLE();
}

void FreeForm2::LlvmCodeGenerator::Visit(const MatchGuardExpression &) {
  // Not supported in FF2.
  FF2_UNREACHABLE();
}

void FreeForm2::LlvmCodeGenerator::Visit(const MatchBindExpression &) {
  // Not supported in FF2.
  FF2_UNREACHABLE();
}

void FreeForm2::LlvmCodeGenerator::Visit(const MemberAccessExpression &) {
  // Not yet supported.
  Unreachable(__FILE__, __LINE__);
}

void FreeForm2::LlvmCodeGenerator::VisitReference(
    const MemberAccessExpression &) {
  // Not yet supported.
  Unreachable(__FILE__, __LINE__);
}

// void
// FreeForm2::LlvmCodeGenerator::Visit(const ObjectMethodExpression& p_expr)
//{
//     // Not supported in FF2.
//     FF2_UNREACHABLE();
// }

// void
// FreeForm2::LlvmCodeGenerator::Visit(const NeuralInputResultExpression&
// p_expr)
//{
//     // Only outputting to a float array is allowed at this time.
//     FF2_ASSERT(p_expr.m_child.GetType().Primitive() == Type::Float);
//
//     // Treat output as an array of float
//     auto cast
//         =
//         m_state.GetBuilder().CreatePointerCast(&m_state.GetArrayReturnSpace(),
//                                                  &m_state.GetFloatPtrType());
//     CHECK_LLVM_RET(cast);
//
//     // Get pointer to the target element in the output array.
//     auto element
//         = m_state.GetBuilder().CreateConstInBoundsGEP1_32(cast,
//                                                           p_expr.m_index);
//     CHECK_LLVM_RET(element);
//
//     // Store value in the target.
//     m_state.GetBuilder().CreateStore(m_stack.top(), element);
// }

void FreeForm2::LlvmCodeGenerator::Visit(const PhiNodeExpression &p_expr) {
  // Not supported in FF2.
  FF2_UNREACHABLE();
}

void FreeForm2::LlvmCodeGenerator::Visit(const PublishExpression &p_expr) {
  // Not supported in FF2.
  FF2_UNREACHABLE();
}

void FreeForm2::LlvmCodeGenerator::Visit(const FeatureRefExpression &p_expr) {
  FF2_ASSERT(m_state.GetFeatureArgument() != NULL);

  // Call below encodes size of m_index, so we check it's what we expect.
  llvm::Value *address = m_state.GetBuilder().CreateConstInBoundsGEP1_32(
      m_state.GetFeatureArgument(), p_expr.m_index,
      llvm::Twine("feature array access"));
  CHECK_LLVM_RET(address);
  llvm::Value *val = m_state.GetBuilder().CreateLoad(address);
  CHECK_LLVM_RET(val);

  // We ensure that there's enough space in the integer type to take the
  // features, plus at least a sign bit.  Zero-extend the feature value into a
  // full integer.
  BOOST_STATIC_ASSERT(sizeof(Result::IntType) >
                      sizeof(Expression::FeatureType));
  FF2_ASSERT(p_expr.GetType().Primitive() == Type::Int);
  val = m_state.GetBuilder().CreateZExt(val, &m_state.GetIntType());
  CHECK_LLVM_RET(val);
  m_stack.push(val);
}

// void
// FreeForm2::LlvmCodeGenerator::Visit(const FSMExpression&)
//{
//     // Not supported in FF2.
//     Unreachable(__FILE__, __LINE__);
// }

void FreeForm2::LlvmCodeGenerator::Visit(const FunctionExpression &) {
  // Not supported in FF2.
  Unreachable(__FILE__, __LINE__);
}

void FreeForm2::LlvmCodeGenerator::Visit(const FunctionCallExpression &) {
  // Not supported in FF2.
  Unreachable(__FILE__, __LINE__);
}

bool FreeForm2::LlvmCodeGenerator::AlternativeVisit(
    const RangeReduceExpression &p_expr) {
  llvm::IRBuilder<> &builder = m_state.GetBuilder();
  FF2_ASSERT(p_expr.GetReduceId() != VariableID::c_invalidID);

  p_expr.GetLow().Accept(*this);
  CompiledValue &index = *m_stack.top();
  m_stack.pop();

  p_expr.GetHigh().Accept(*this);
  CompiledValue &limit = *m_stack.top();
  m_stack.pop();

  CompiledValue *step = llvm::ConstantInt::get(
      m_state.GetContext(), llvm::APInt(m_state.GetIntBits(), 1));
  CHECK_LLVM_RET(step);

  p_expr.GetInitial().Accept(*this);
  CompiledValue &initial = *m_stack.top();
  m_stack.pop();

  // Save current state.
  llvm::BasicBlock *originalBlock = builder.GetInsertBlock();
  llvm::Function *currentFun = originalBlock->getParent();

  // Create a block for the accumulator and loop-var phi nodes, as well as
  // the condition.
  llvm::BasicBlock *condBlock = llvm::BasicBlock::Create(
      m_state.GetContext(), llvm::Twine("range-reduce-condition"), currentFun);
  CHECK_LLVM_RET(condBlock);

  // Create and insert a basic block that encapsulates the loop, and a block
  // for the code after the loop.
  llvm::BasicBlock *loopBlock = llvm::BasicBlock::Create(
      m_state.GetContext(), llvm::Twine("range-reduce-loop"), currentFun);
  CHECK_LLVM_RET(loopBlock);
  llvm::BasicBlock *afterLoopBlock = llvm::BasicBlock::Create(
      m_state.GetContext(), llvm::Twine("range-reduce-after"), currentFun);
  CHECK_LLVM_RET(afterLoopBlock);

  const CompiledValue *startBranch = builder.CreateBr(condBlock);
  CHECK_LLVM_RET(startBranch);

  builder.SetInsertPoint(condBlock);

  // Create a PHI node to unify the values of the loop index, and one to unify
  // values of the accumulator variable.
  llvm::PHINode *loopVar = builder.CreatePHI(
      index.getType(), 2, llvm::Twine("range-reduce-loop-var"));
  CHECK_LLVM_RET(loopVar);
  loopVar->addIncoming(&index, originalBlock);
  m_state.SetVariableValue(p_expr.GetStepId(), *loopVar);

  FF2_ASSERT(p_expr.GetType().Primitive() != Type::Void);
  llvm::PHINode *accVar = builder.CreatePHI(
      initial.getType(), 2, llvm::Twine("range-reduce-acc-var"));
  CHECK_LLVM_RET(accVar);
  accVar->addIncoming(&initial, originalBlock);
  m_state.SetVariableValue(p_expr.GetReduceId(), *accVar);

  // Visit the initial condition (that the low limit is less than the high
  // limit). This guards against iterating through the loop once under these
  // conditions.
  CompiledValue *initialCond = builder.CreateICmpSLT(loopVar, &limit);
  CHECK_LLVM_RET(initialCond);

  builder.CreateCondBr(initialCond, loopBlock, afterLoopBlock);
  builder.SetInsertPoint(loopBlock);

  p_expr.GetReduceExpression().Accept(*this);

  LlvmCodeGenerator::CompiledValue &loopValue = *m_stack.top();
  m_stack.pop();

  // Add step to the loop variable.
  llvm::Type *types[] = {&m_state.GetIntType()};
  llvm::Function *add = llvm::Intrinsic::getDeclaration(
      &m_state.GetModule(), llvm::Intrinsic::sadd_with_overflow, types);
  CHECK_LLVM_RET(add);
  CompiledValue *addCall = builder.CreateCall2(add, loopVar, step);
  CHECK_LLVM_RET(addCall);
  CompiledValue *inc = builder.CreateExtractValue(addCall, 0);
  CHECK_LLVM_RET(inc);

  CompiledValue *jumpCond = builder.CreateExtractValue(addCall, 1);
  CHECK_LLVM_RET(jumpCond);

  // Create a conditional branch to the condition block; overflow causes
  // a loop to break.
  const CompiledValue *jump =
      builder.CreateCondBr(jumpCond, afterLoopBlock, condBlock);
  CHECK_LLVM_RET(jump);

  // Add new values to PHI nodes.
  llvm::BasicBlock *loopEndBlock = builder.GetInsertBlock();
  loopVar->addIncoming(inc, loopEndBlock);

  builder.SetInsertPoint(afterLoopBlock);

  accVar->addIncoming(&loopValue, loopEndBlock);

  // Create a PHI node in case the loop was skipped entirely
  llvm::PHINode *end = builder.CreatePHI(initial.getType(), 2,
                                         llvm::Twine("range-reduce-skip-acc"));
  end->addIncoming(accVar, condBlock);
  end->addIncoming(&loopValue, loopEndBlock);

  m_stack.push(end);
  return true;
}

void FreeForm2::LlvmCodeGenerator::Visit(const ForEachLoopExpression &) {
  // TODO: Implement (TFS #461742)
  Unreachable(__FILE__, __LINE__);
}

void FreeForm2::LlvmCodeGenerator::Visit(const ComplexRangeLoopExpression &) {
  // TODO: Implement (TFS #461742)
  Unreachable(__FILE__, __LINE__);
}

void FreeForm2::LlvmCodeGenerator::Visit(const RangeReduceExpression &p_expr) {
  // Handled by AlternativeVisit.
  Unreachable(__FILE__, __LINE__);
}

void FreeForm2::LlvmCodeGenerator::Visit(
    const UnaryOperatorExpression &p_expr) {
  switch (p_expr.m_op) {
    case UnaryOperator::minus: {
      VisitUnaryMinus(p_expr);
      break;
    }

    case UnaryOperator::log: {
      VisitUnaryLog(p_expr, false);
      break;
    }

    case UnaryOperator::log1: {
      VisitUnaryLog(p_expr, true);
      break;
    }

    case UnaryOperator::abs: {
      VisitUnaryAbs(p_expr);
      break;
    }

    case UnaryOperator::_not:
    case UnaryOperator::bitnot: {
      VisitUnaryNot(p_expr);
      break;
    }

    case UnaryOperator::round: {
      VisitUnaryRound(p_expr);
      break;
    }

    case UnaryOperator::trunc: {
      VisitUnaryTrunc(p_expr);
      break;
    }

    default: {
      Unreachable(__FILE__, __LINE__);
      break;
    }
  };
}

void FreeForm2::LlvmCodeGenerator::VisitUnaryMinus(
    const UnaryOperatorExpression &p_expr) {
  LlvmCodeGenerator::CompiledValue *child = m_stack.top();
  m_stack.pop();

  LlvmCodeGenerator::CompiledValue *neg;

  if (p_expr.GetType().IsIntegerType()) {
    neg = m_state.GetBuilder().CreateNeg(child);
  } else if (p_expr.GetType().IsFloatingPointType()) {
    neg = m_state.GetBuilder().CreateFNeg(child);
  } else {
    FF2_UNREACHABLE();
  }
  CHECK_LLVM_RET(neg);
  m_stack.push(neg);
}

LlvmCodeGenerator::CompiledValue &CompileLogCall(
    CompilationState &p_state, llvm::ArrayRef<llvm::Type *> p_type,
    LlvmCodeGenerator::CompiledValue &p_child) {
  llvm::Function *fun = llvm::Intrinsic::getDeclaration(
      &p_state.GetModule(), llvm::Intrinsic::log, p_type);
  CHECK_LLVM_RET(fun);
  LlvmCodeGenerator::CompiledValue *log =
      p_state.GetBuilder().CreateCall(fun, &p_child);
  CHECK_LLVM_RET(log);
  return *log;
}

void FreeForm2::LlvmCodeGenerator::VisitUnaryLog(
    const UnaryOperatorExpression &p_expr, bool p_addOne) {
  FF2_ASSERT(p_expr.GetType().IsFloatingPointType());
  FF2_ASSERT(p_expr.m_child.GetType().IsFloatingPointType() ||
             p_expr.m_child.GetType().IsIntegerType());

  LlvmCodeGenerator::CompiledValue *arg = m_stack.top();
  m_stack.pop();

  // Ensure the argument is a float.
  arg = &ValueConversion::Do(*arg, p_expr.m_child.GetType(), p_expr.GetType(),
                             m_state);

  llvm::Type &floatType = m_state.GetType(p_expr.GetType());
  if (p_addOne) {
    LlvmCodeGenerator::CompiledValue *one =
        llvm::ConstantFP::get(&floatType, 1);
    CHECK_LLVM_RET(one);
    arg = m_state.GetBuilder().CreateFAdd(arg, one);
    CHECK_LLVM_RET(arg);
  }

  // Guard against values zero or less, for which we always return
  // negative infinity.
  LlvmCodeGenerator::CompiledValue *zero = llvm::ConstantFP::get(&floatType, 0);
  CHECK_LLVM_RET(zero);
  LlvmCodeGenerator::CompiledValue *cmp =
      m_state.GetBuilder().CreateFCmpOGT(arg, zero);
  CHECK_LLVM_RET(cmp);

  LlvmCodeGenerator::CompiledValue *log =
      &CompileLogCall(m_state, &floatType, *arg);

  // Select the value to return.
  LlvmCodeGenerator::CompiledValue *negInfinity =
      llvm::ConstantFP::getInfinity(&floatType, true);
  CHECK_LLVM_RET(negInfinity);
  LlvmCodeGenerator::CompiledValue *select =
      m_state.GetBuilder().CreateSelect(cmp, log, negInfinity);
  CHECK_LLVM_RET(select);

  m_stack.push(select);
}

void FreeForm2::LlvmCodeGenerator::VisitUnaryNot(
    const UnaryOperatorExpression &p_expr) {
  FF2_ASSERT(p_expr.GetType().Primitive() == Type::Bool ||
             p_expr.GetType().IsIntegerType());
  LlvmCodeGenerator::CompiledValue *child = m_stack.top();
  m_stack.pop();

  LlvmCodeGenerator::CompiledValue *ret = m_state.GetBuilder().CreateNot(child);
  CHECK_LLVM_RET(ret);
  m_stack.push(ret);
}

void FreeForm2::LlvmCodeGenerator::VisitUnaryAbs(
    const UnaryOperatorExpression &p_expr) {
  FF2_ASSERT(p_expr.GetType().IsFloatingPointType() ||
             p_expr.GetType().IsIntegerType());

  CompiledValue *child = m_stack.top();
  m_stack.pop();

  CompiledValue *zero = &m_state.CreateZeroValue(p_expr.GetType());
  CompiledValue *cond = nullptr;
  CompiledValue *neg = nullptr;

  if (p_expr.GetType().IsIntegerType()) {
    // Generate the condition.
    cond = m_state.GetBuilder().CreateICmpSGE(child, zero,
                                              llvm::Twine("abs test"));

    // Create negation.
    neg = m_state.GetBuilder().CreateSub(zero, child);
  } else {
    // Generate the condition.
    cond = m_state.GetBuilder().CreateFCmpOGE(child, zero,
                                              llvm::Twine("abs test"));

    // Create negation.
    neg = m_state.GetBuilder().CreateFSub(zero, child);
  }
  CHECK_LLVM_RET(cond);
  CHECK_LLVM_RET(neg);

  // Select between them.
  LlvmCodeGenerator::CompiledValue *select =
      m_state.GetBuilder().CreateSelect(cond, child, neg);
  CHECK_LLVM_RET(select);

  m_stack.push(select);
}

void FreeForm2::LlvmCodeGenerator::VisitUnaryRound(
    const UnaryOperatorExpression &p_expr) {
  FF2_ASSERT(p_expr.GetType().IsIntegerType());
  FF2_ASSERT(p_expr.m_child.GetType().IsFloatingPointType() ||
             p_expr.m_child.GetType().IsIntegerType());
  if (p_expr.m_child.GetType().IsIntegerType()) {
    FF2_ASSERT(p_expr.GetType().IsSameAs(p_expr.m_child.GetType(), true));

    // No work is necessary to round an int.
    return;
  }

  CompiledValue *child = m_stack.top();
  m_stack.pop();

  CompiledValue *ret = NULL;
  llvm::IRBuilder<> &builder = m_state.GetBuilder();

  llvm::Type &floatType = m_state.GetType(p_expr.m_child.GetType());
  llvm::Type &intType = m_state.GetType(p_expr.GetType());

  CompiledValue *zero = llvm::ConstantFP::get(&floatType, 0.0);
  CHECK_LLVM_RET(zero);
  CompiledValue *cmp =
      builder.CreateFCmpOGE(child, zero, llvm::Twine("round comparison"));
  CHECK_LLVM_RET(cmp);

  CompiledValue *half = llvm::ConstantFP::get(&floatType, 0.5);
  CHECK_LLVM_RET(half);

  GenerateConditional cond(m_state, *cmp, "round test");

  // Generate condition where value is over zero, and we add 0.5 to push
  // toward infinity.
  CompiledValue *plus = builder.CreateFAdd(child, half);
  CHECK_LLVM_RET(plus);
  CompiledValue *plusret = builder.CreateFPToSI(plus, &intType);
  CHECK_LLVM_RET(plusret);
  cond.FinishThen(plusret);

  // Generate condition where value is under zero, and we subtract 0.5
  // to push toward infinity.
  CompiledValue *minus = builder.CreateFSub(child, half);
  CHECK_LLVM_RET(minus);
  CompiledValue *minusret = builder.CreateFPToSI(minus, &intType);
  CHECK_LLVM_RET(minusret);

  cond.FinishElse(minusret);

  ret = &cond.Finish(intType);

  m_stack.push(ret);
}

void FreeForm2::LlvmCodeGenerator::VisitUnaryTrunc(
    const UnaryOperatorExpression &p_expr) {
  FF2_ASSERT(p_expr.GetType().IsIntegerType());
  if (p_expr.m_child.GetType().IsIntegerType()) {
    // This operator covers only floating-point-to-int truncation. Int-to-
    // int truncation is covert by the ConvertTo*Expressions.
    FF2_ASSERT(p_expr.m_child.GetType().IsSameAs(p_expr.GetType(), true));
    return;
  } else {
    FF2_ASSERT(p_expr.m_child.GetType().IsFloatingPointType());
  }

  LlvmCodeGenerator::CompiledValue *child = m_stack.top();
  m_stack.pop();

  CompiledValue *ret = &ValueConversion::Do(*child, p_expr.m_child.GetType(),
                                            p_expr.GetType(), m_state);

  m_stack.push(ret);
}

void FreeForm2::LlvmCodeGenerator::Visit(
    const BinaryOperatorExpression &p_expr) {
  // All the operands in the binary expression are pushed into the stack
  // left-to-right. The binary operators expect the operands to be
  // in reverse order, so a queue is used to invert that part of the stack.
  std::queue<LlvmCodeGenerator::CompiledValue *> tmpQueue;

  for (size_t i = 0; i < p_expr.GetNumChildren(); i++) {
    tmpQueue.push(m_stack.top());
    m_stack.pop();
  }

  for (size_t i = 0; i < p_expr.GetNumChildren(); i++) {
    m_stack.push(tmpQueue.front());
    tmpQueue.pop();
  }

  for (size_t i = 1; i < p_expr.GetNumChildren(); i++) {
    switch (p_expr.GetOperator()) {
      case BinaryOperator::plus: {
        VisitPlus(p_expr);
        break;
      }

      case BinaryOperator::minus: {
        VisitMinus(p_expr);
        break;
      }

      case BinaryOperator::multiply: {
        VisitMultiply(p_expr);
        break;
      }

      case BinaryOperator::divides: {
        VisitDivides(p_expr);
        break;
      }

      case BinaryOperator::mod: {
        VisitMod(p_expr);
        break;
      }

      case BinaryOperator::_and:
      case BinaryOperator::_bitand: {
        VisitAnd(p_expr);
        break;
      }

      case BinaryOperator::_or:
      case BinaryOperator::_bitor: {
        VisitOr(p_expr);
        break;
      }

      case BinaryOperator::log: {
        VisitLog(p_expr);
        break;
      }

      case BinaryOperator::pow: {
        VisitPow(p_expr);
        break;
      }

      case BinaryOperator::max: {
        VisitMaxMin(p_expr, false);
        break;
      }

      case BinaryOperator::min: {
        VisitMaxMin(p_expr, true);
        break;
      }

      case BinaryOperator::eq: {
        VisitEqual(p_expr, false);
        break;
      }

      case BinaryOperator::neq: {
        VisitEqual(p_expr, true);
        break;
      }

      case BinaryOperator::lt: {
        VisitCompare(p_expr, true, false);
        break;
      }

      case BinaryOperator::lte: {
        VisitCompare(p_expr, true, true);
        break;
      }

      case BinaryOperator::gt: {
        VisitCompare(p_expr, false, false);
        break;
      }

      case BinaryOperator::gte: {
        VisitCompare(p_expr, false, true);
        break;
      }

      default: {
        Unreachable(__FILE__, __LINE__);
        break;
      }
    };
  }
}

void FreeForm2::LlvmCodeGenerator::VisitPlus(
    const BinaryOperatorExpression &p_expr) {
  FF2_ASSERT(p_expr.GetType() == p_expr.GetChildType());

  CompiledValue *left = m_stack.top();
  m_stack.pop();

  CompiledValue *right = m_stack.top();
  m_stack.pop();

  CompiledValue *ret = NULL;
  if (p_expr.GetType().IsIntegerType()) {
    ret = m_state.GetBuilder().CreateAdd(left, right);
  } else if (p_expr.GetType().IsFloatingPointType()) {
    ret = m_state.GetBuilder().CreateFAdd(left, right);
  } else {
    // Shouldn't get here.
    FF2_UNREACHABLE();
  }
  CHECK_LLVM_RET(ret);
  m_stack.push(ret);
}

void FreeForm2::LlvmCodeGenerator::VisitMinus(
    const BinaryOperatorExpression &p_expr) {
  FF2_ASSERT(p_expr.GetType() == p_expr.GetChildType());

  CompiledValue *left = m_stack.top();
  m_stack.pop();

  CompiledValue *right = m_stack.top();
  m_stack.pop();

  CompiledValue *ret = NULL;
  if (p_expr.GetType().IsIntegerType()) {
    ret = m_state.GetBuilder().CreateSub(left, right);
  } else if (p_expr.GetType().IsFloatingPointType()) {
    ret = m_state.GetBuilder().CreateFSub(left, right);
  } else {
    // Shouldn't get here.
    FF2_UNREACHABLE();
  }
  CHECK_LLVM_RET(ret);
  m_stack.push(ret);
}

void FreeForm2::LlvmCodeGenerator::VisitMultiply(
    const BinaryOperatorExpression &p_expr) {
  FF2_ASSERT(p_expr.GetType() == p_expr.GetChildType());

  LlvmCodeGenerator::CompiledValue *left = m_stack.top();
  m_stack.pop();

  LlvmCodeGenerator::CompiledValue *right = m_stack.top();
  m_stack.pop();

  LlvmCodeGenerator::CompiledValue *ret = NULL;
  if (p_expr.GetType().IsIntegerType()) {
    ret = m_state.GetBuilder().CreateMul(left, right);
  } else if (p_expr.GetType().IsFloatingPointType()) {
    ret = m_state.GetBuilder().CreateFMul(left, right);
  } else {
    // Shouldn't get here.
    FF2_UNREACHABLE();
  }
  CHECK_LLVM_RET(ret);
  m_stack.push(ret);
}

void FreeForm2::LlvmCodeGenerator::VisitAnd(
    const BinaryOperatorExpression &p_expr) {
  FF2_ASSERT(p_expr.GetType() == p_expr.GetChildType());

  LlvmCodeGenerator::CompiledValue *left = m_stack.top();
  m_stack.pop();

  LlvmCodeGenerator::CompiledValue *right = m_stack.top();
  m_stack.pop();

  LlvmCodeGenerator::CompiledValue *ret =
      m_state.GetBuilder().CreateAnd(left, right);
  CHECK_LLVM_RET(ret);

  m_stack.push(ret);
}

void FreeForm2::LlvmCodeGenerator::VisitOr(
    const BinaryOperatorExpression &p_expr) {
  FF2_ASSERT(p_expr.GetType() == p_expr.GetChildType());

  LlvmCodeGenerator::CompiledValue *left = m_stack.top();
  m_stack.pop();

  LlvmCodeGenerator::CompiledValue *right = m_stack.top();
  m_stack.pop();

  LlvmCodeGenerator::CompiledValue *ret =
      m_state.GetBuilder().CreateOr(left, right);
  CHECK_LLVM_RET(ret);

  m_stack.push(ret);
}

void FreeForm2::LlvmCodeGenerator::VisitLog(
    const BinaryOperatorExpression &p_expr) {
  FF2_ASSERT(p_expr.GetType().IsFloatingPointType());
  FF2_ASSERT(p_expr.GetChildType().IsFloatingPointType() ||
             p_expr.GetChildType().IsIntegerType());

  CompiledValue *left = m_stack.top();
  m_stack.pop();

  CompiledValue *right = m_stack.top();
  m_stack.pop();

  if (!p_expr.GetChildType().IsFloatingPointType()) {
    left = &ValueConversion::Do(*left, p_expr.GetChildType(), p_expr.GetType(),
                                m_state);
    right = &ValueConversion::Do(*right, p_expr.GetChildType(),
                                 p_expr.GetType(), m_state);
  }

  // Guard against values zero or less, for which we always return
  // negative infinity.
  CompiledValue *zero = &m_state.CreateZeroValue(p_expr.GetType());
  CompiledValue *cmpLeft = m_state.GetBuilder().CreateFCmpOGT(left, zero);
  CHECK_LLVM_RET(cmpLeft);
  CompiledValue *cmpRight = m_state.GetBuilder().CreateFCmpOGT(right, zero);
  CHECK_LLVM_RET(cmpRight);
  CompiledValue *cmp = m_state.GetBuilder().CreateAnd(cmpLeft, cmpRight,
                                                      llvm::Twine("log guard"));

  // Create the log.
  llvm::Type &floatType = m_state.GetType(p_expr.GetType());
  CompiledValue *logLeft = &CompileLogCall(m_state, &floatType, *left);
  CompiledValue *logRight = &CompileLogCall(m_state, &floatType, *right);
  CompiledValue *log = m_state.GetBuilder().CreateFDiv(logLeft, logRight);

  // Select the value to return.
  CompiledValue *negInfinity = llvm::ConstantFP::getInfinity(&floatType, true);
  CHECK_LLVM_RET(negInfinity);
  CompiledValue *select =
      m_state.GetBuilder().CreateSelect(cmp, log, negInfinity);
  CHECK_LLVM_RET(select);
  m_stack.push(select);
}

// Typedef for a function that combines two values into a returned value.
typedef LlvmCodeGenerator::CompiledValue &(*OperatorFun)(
    CompilationState &m_state, LlvmCodeGenerator::CompiledValue &p_left,
    LlvmCodeGenerator::CompiledValue &p_right);

static LlvmCodeGenerator::CompiledValue &GenerateSDivInstruction(
    CompilationState &m_state, LlvmCodeGenerator::CompiledValue &p_left,
    LlvmCodeGenerator::CompiledValue &p_right) {
  LlvmCodeGenerator::CompiledValue *div =
      m_state.GetBuilder().CreateSDiv(&p_left, &p_right);
  CHECK_LLVM_RET(div);
  return *div;
}

static LlvmCodeGenerator::CompiledValue &GenerateUDivInstruction(
    CompilationState &m_state, LlvmCodeGenerator::CompiledValue &p_left,
    LlvmCodeGenerator::CompiledValue &p_right) {
  LlvmCodeGenerator::CompiledValue *div =
      m_state.GetBuilder().CreateUDiv(&p_left, &p_right);
  CHECK_LLVM_RET(div);
  return *div;
}

static LlvmCodeGenerator::CompiledValue &GenerateSModInstruction(
    CompilationState &m_state, LlvmCodeGenerator::CompiledValue &p_left,
    LlvmCodeGenerator::CompiledValue &p_right) {
  LlvmCodeGenerator::CompiledValue *mod =
      m_state.GetBuilder().CreateSRem(&p_left, &p_right);
  CHECK_LLVM_RET(mod);
  return *mod;
}

static LlvmCodeGenerator::CompiledValue &GenerateUModInstruction(
    CompilationState &m_state, LlvmCodeGenerator::CompiledValue &p_left,
    LlvmCodeGenerator::CompiledValue &p_right) {
  LlvmCodeGenerator::CompiledValue *mod =
      m_state.GetBuilder().CreateURem(&p_left, &p_right);
  CHECK_LLVM_RET(mod);
  return *mod;
}

LlvmCodeGenerator::CompiledValue &CompileGuardedDivMod(
    CompilationState &m_state, LlvmCodeGenerator::CompiledValue &p_left,
    LlvmCodeGenerator::CompiledValue &p_right, OperatorFun p_operator,
    LlvmCodeGenerator::CompiledValue &underflowResult) {
  // Assert that both operands are integer types of the same size.
  FF2_ASSERT(p_left.getType() && p_right.getType() &&
             p_left.getType()->isIntegerTy() &&
             p_right.getType()->isIntegerTy(
                 p_left.getType()->getPrimitiveSizeInBits()));
  llvm::IntegerType &intType = llvm::cast<llvm::IntegerType>(*p_left.getType());

  // Division/Modulus is complicated in the freeforms language by the
  // fact that we guard for division by zero using conditionals.

  // Generate the test for divide-by-zero.
  LlvmCodeGenerator::CompiledValue *zero = llvm::ConstantInt::get(&intType, 0);
  CHECK_LLVM_RET(zero);
  LlvmCodeGenerator::CompiledValue *zeroCond =
      m_state.GetBuilder().CreateICmpNE(zero, &p_right,
                                        llvm::Twine("div-by-zero guard"));
  CHECK_LLVM_RET(zeroCond);

  GenerateConditional divzero(m_state, *zeroCond, "div-by-zero guard");

  // Generate some constants.
  LlvmCodeGenerator::CompiledValue *negOne =
      llvm::ConstantInt::getSigned(&intType, -1);
  CHECK_LLVM_RET(negOne);
  LlvmCodeGenerator::CompiledValue *minInt = llvm::ConstantInt::get(
      p_left.getType(), llvm::APInt::getSignedMinValue(intType.getBitWidth()));
  CHECK_LLVM_RET(minInt);

  // Check whether we're dividing MIN_INT by -1 (which causes a
  // strange underflow condition).
  LlvmCodeGenerator::CompiledValue *minIntCond =
      m_state.GetBuilder().CreateICmpNE(&p_left, minInt);
  CHECK_LLVM_RET(minIntCond);
  LlvmCodeGenerator::CompiledValue *negOneCond =
      m_state.GetBuilder().CreateICmpNE(&p_right, negOne);
  CHECK_LLVM_RET(negOneCond);
  LlvmCodeGenerator::CompiledValue *underFlowCond =
      m_state.GetBuilder().CreateOr(minIntCond, negOneCond);
  CHECK_LLVM_RET(underFlowCond);

  // Select between underflow result and regular result.  Note that we
  // need to generate the instruction here, via a function pointer, to
  // avoid evaluating that instruction when we are dividing by zero.
  GenerateConditional underflow(m_state, *underFlowCond, "underflow guard");
  underflow.FinishThen(&p_operator(m_state, p_left, p_right));
  underflow.FinishElse(&underflowResult);

  // Select between result of underflow comparison and div-by-zero result
  // (defined to be zero).
  divzero.FinishThen(&underflow.Finish(intType));
  divzero.FinishElse(zero);
  return divzero.Finish(intType);
}

void FreeForm2::LlvmCodeGenerator::VisitDivides(
    const BinaryOperatorExpression &p_expr) {
  FF2_ASSERT(p_expr.GetType() == p_expr.GetChildType());

  CompiledValue *left = m_stack.top();
  m_stack.pop();

  CompiledValue *right = m_stack.top();
  m_stack.pop();

  CompiledValue *ret = nullptr;

  if (p_expr.GetType().IsIntegerType()) {
    llvm::Type &type = m_state.GetType(p_expr.GetType());
    FF2_ASSERT(type.isIntegerTy());

    llvm::APInt max;
    OperatorFun op;
    if (p_expr.GetType().IsSigned()) {
      max = llvm::APInt::getSignedMaxValue(type.getPrimitiveSizeInBits());
      op = GenerateSDivInstruction;
    } else {
      max = llvm::APInt::getMaxValue(type.getPrimitiveSizeInBits());
      op = GenerateUDivInstruction;
    }
    CompiledValue *maxInt = llvm::ConstantInt::get(&type, max);
    CHECK_LLVM_RET(maxInt);

    ret = &CompileGuardedDivMod(m_state, *left, *right, op, *maxInt);
  } else if (p_expr.GetType().IsFloatingPointType()) {
    ret = m_state.GetBuilder().CreateFDiv(left, right);
  } else {
    FF2_UNREACHABLE();
  }
  CHECK_LLVM_RET(ret);

  m_stack.push(ret);
}

void FreeForm2::LlvmCodeGenerator::VisitMod(
    const BinaryOperatorExpression &p_expr) {
  FF2_ASSERT(p_expr.GetType() == p_expr.GetChildType());

  CompiledValue *left = m_stack.top();
  m_stack.pop();

  CompiledValue *right = m_stack.top();
  m_stack.pop();

  CompiledValue *ret = nullptr;

  if (p_expr.GetType().IsIntegerType()) {
    CompiledValue &zero = m_state.CreateZeroValue(p_expr.GetType());
    if (p_expr.GetType().IsSigned()) {
      ret = &CompileGuardedDivMod(m_state, *left, *right,
                                  GenerateSModInstruction, zero);
    } else {
      ret = &CompileGuardedDivMod(m_state, *left, *right,
                                  GenerateUModInstruction, zero);
    }
  } else if (p_expr.GetType().IsFloatingPointType()) {
    ret = m_state.GetBuilder().CreateFRem(left, right);
    CHECK_LLVM_RET(ret);
  } else {
    FF2_UNREACHABLE();
  }

  m_stack.push(ret);
}

void FreeForm2::LlvmCodeGenerator::VisitPow(
    const BinaryOperatorExpression &p_expr) {
  FF2_ASSERT(p_expr.GetType() == p_expr.GetChildType());
  FF2_ASSERT(p_expr.GetType().IsIntegerType() ||
             p_expr.GetType().IsFloatingPointType());

  CompiledValue *left = m_stack.top();
  m_stack.pop();

  CompiledValue *right = m_stack.top();
  m_stack.pop();

  CompiledValue *ret = nullptr;

  if (p_expr.GetChildType().IsIntegerType()) {
    const TypeImpl &floatType = TypeImpl::GetFloatInstance(true);
    left =
        &ValueConversion::Do(*left, p_expr.GetChildType(), floatType, m_state);
    right =
        &ValueConversion::Do(*right, p_expr.GetChildType(), floatType, m_state);
  } else {
    FF2_ASSERT(left->getType() && right->getType() &&
               left->getType()->getTypeID() == right->getType()->getTypeID());
  }

  // Create a reference to the LLVM intrinsic 'pow', which
  // does floating point power raising (note that we supply
  // the floating point type we're interested in as a
  // parameter, to specify exactly which 'pow' we want).
  llvm::Type *floatType = left->getType();
  llvm::Function *fun = llvm::Intrinsic::getDeclaration(
      &m_state.GetModule(), llvm::Intrinsic::pow, floatType);
  CHECK_LLVM_RET(fun);
  ret = m_state.GetBuilder().CreateCall2(fun, left, right);
  CHECK_LLVM_RET(ret);

  if (p_expr.GetType().IsIntegerType()) {
    ret = &ValueConversion::Do(*ret, TypeImpl::GetFloatInstance(true),
                               p_expr.GetType(), m_state);
  }

  m_stack.push(ret);
}

void FreeForm2::LlvmCodeGenerator::VisitMaxMin(
    const BinaryOperatorExpression &p_expr, bool p_minimum) {
  FF2_ASSERT(p_expr.GetType() == p_expr.GetChildType());
  FF2_ASSERT(p_expr.GetChildType().IsIntegerType() ||
             p_expr.GetChildType().IsFloatingPointType());

  // Compare the two values.
  const char *descr = p_minimum ? "min compare" : "max compare";

  CompiledValue *left = m_stack.top();
  m_stack.pop();

  CompiledValue *right = m_stack.top();
  m_stack.pop();

  CompiledValue *cmp = nullptr;
  CompiledValue *select = nullptr;

  if (p_expr.GetType().IsIntegerType()) {
    cmp = m_state.GetBuilder().CreateICmpSGT(left, right, llvm::Twine(descr));
  } else {
    cmp = m_state.GetBuilder().CreateFCmpOGT(left, right, llvm::Twine(descr));
  }
  CHECK_LLVM_RET(cmp);
  select = m_state.GetBuilder().CreateSelect(cmp, p_minimum ? right : left,
                                             p_minimum ? left : right);
  CHECK_LLVM_RET(select);
  m_stack.push(select);
}

void FreeForm2::LlvmCodeGenerator::VisitEqual(
    const BinaryOperatorExpression &p_expr, bool p_inequality) {
  CompiledValue *left = m_stack.top();
  m_stack.pop();

  CompiledValue *right = m_stack.top();
  m_stack.pop();

  CompiledValue *ret = nullptr;
  if (p_expr.GetChildType().IsIntegerType() ||
      p_expr.GetChildType().Primitive() == Type::Bool) {
    ret = m_state.GetBuilder().CreateICmp(llvm::CmpInst::ICMP_EQ, left, right);
  } else if (p_expr.GetChildType().IsFloatingPointType()) {
    ret = &CompileFloatEquality(m_state, *left, *right);
  } else {
    // Shouldn't get here.
    FF2_UNREACHABLE();
  }
  CHECK_LLVM_RET(ret);

  if (p_inequality) {
    ret = m_state.GetBuilder().CreateNot(ret);
    CHECK_LLVM_RET(ret);
  }
  m_stack.push(ret);
}

void FreeForm2::LlvmCodeGenerator::VisitCompare(
    const BinaryOperatorExpression &p_expr, bool p_less, bool p_equal) {
  CompiledValue *left = m_stack.top();
  m_stack.pop();

  CompiledValue *right = m_stack.top();
  m_stack.pop();

  CompiledValue *ret = nullptr;
  llvm::CmpInst::Predicate predicate = llvm::CmpInst::BAD_ICMP_PREDICATE;

  if (p_expr.GetChildType().IsIntegerType() ||
      p_expr.GetChildType().Primitive() == Type::Bool) {
    if (p_less) {
      predicate = p_equal ? llvm::CmpInst::ICMP_SLE : llvm::CmpInst::ICMP_SLT;
    } else {
      predicate = p_equal ? llvm::CmpInst::ICMP_SGE : llvm::CmpInst::ICMP_SGT;
    }

    ret = m_state.GetBuilder().CreateICmp(predicate, left, right);
  } else if (p_expr.GetChildType().IsFloatingPointType()) {
    if (p_less) {
      predicate = p_equal ? llvm::CmpInst::FCMP_OLE : llvm::CmpInst::FCMP_OLT;
    } else {
      predicate = p_equal ? llvm::CmpInst::FCMP_OGE : llvm::CmpInst::FCMP_OGT;
    }

    ret = m_state.GetBuilder().CreateFCmp(predicate, left, right);
    if (p_equal) {
      CHECK_LLVM_RET(ret);

      LlvmCodeGenerator::CompiledValue &approx =
          CompileFloatEquality(m_state, *left, *right);
      ret = m_state.GetBuilder().CreateOr(ret, &approx);
    }
  } else {
    // Shouldn't get here.
    FF2_UNREACHABLE();
  }
  CHECK_LLVM_RET(ret);
  m_stack.push(ret);
}

llvm::Function *FreeForm2::LlvmCodeGenerator::CreateFeatureFunction(
    const TypeImpl &p_returnType) {
  // Create top-level function.
  llvm::Type *returnType = nullptr;
  std::vector<llvm::Type *> arguments;

  if (m_destinationFunctionType == CompilerFactory::SingleDocumentEvaluation) {
    // The streamFeatureInput argument will not be used, so use any pointer
    // for its type.
    llvm::PointerType *streamFeatureInput =
        llvm::PointerType::get(&m_state.GetIntType(), 0);
    CHECK_LLVM_RET(streamFeatureInput);
    arguments.push_back(streamFeatureInput);

    // Create feature array input type.
    llvm::PointerType *featurePointerType =
        llvm::PointerType::get(&m_state.GetFeatureType(), 0);
    CHECK_LLVM_RET(featurePointerType);
    arguments.push_back(featurePointerType);
  } else if (m_destinationFunctionType ==
             CompilerFactory::DocumentSetEvaluation) {
    // Create feature array input type.
    llvm::PointerType *featurePointerType = llvm::PointerType::get(
        llvm::PointerType::get(&m_state.GetFeatureType(), 0), 0);
    CHECK_LLVM_RET(featurePointerType);
    arguments.push_back(featurePointerType);

    // Create document index input type.
    llvm::IntegerType *docIndexType = &m_state.GetInt32Type();
    CHECK_LLVM_RET(docIndexType);
    arguments.push_back(docIndexType);

    // Create document count input type.
    llvm::IntegerType *docCountType = &m_state.GetInt32Type();
    CHECK_LLVM_RET(docCountType);
    arguments.push_back(docCountType);

    // Create cache pointer type
    llvm::PointerType *cachePointerType =
        llvm::PointerType::get(&m_state.GetIntType(), 0);
    CHECK_LLVM_RET(cachePointerType);
    arguments.push_back(cachePointerType);
  }

  if (p_returnType.Primitive() == Type::Array) {
    // If return type is an array, pass in an argument to copy resulting,
    // flattened array into.  Return value indicates dynamic bounds.
    const ArrayType &info = static_cast<const ArrayType &>(p_returnType);
    llvm::PointerType *space =
        llvm::PointerType::get(&m_state.GetType(info.GetChildType()), 0);
    CHECK_LLVM_RET(space);
    if (m_destinationFunctionType ==
        CompilerFactory::SingleDocumentEvaluation) {
      arguments.push_back(space);
    }
    returnType = &m_state.GetArrayBoundsType();
  } else {
    // Otherwise, pass in dummy arg (pointer to int type).
    llvm::PointerType *space = llvm::PointerType::get(&m_state.GetIntType(), 0);
    CHECK_LLVM_RET(space);
    if (m_destinationFunctionType ==
        CompilerFactory::SingleDocumentEvaluation) {
      arguments.push_back(space);
    }

    returnType = &m_state.GetType(p_returnType);
  }

  llvm::FunctionType *funType =
      llvm::FunctionType::get(returnType, arguments, false);
  CHECK_LLVM_RET(funType);

  llvm::Function *fun = llvm::Function::Create(
      funType, llvm::Function::ExternalLinkage, llvm::Twine("<freeformmain>"),
      &m_state.GetModule());
  CHECK_LLVM_RET(fun);

  // Create a basic block within the function, and codegen into it.
  llvm::BasicBlock *block =
      llvm::BasicBlock::Create(m_state.GetContext(), llvm::Twine("entry"), fun);
  m_state.GetBuilder().SetInsertPoint(block);

  if (m_destinationFunctionType == CompilerFactory::SingleDocumentEvaluation) {
    // Push function arguments onto the value stack.
    llvm::Function::arg_iterator iter = fun->arg_begin();
    FF2_ASSERT(iter != fun->arg_end());
    iter->setName(llvm::Twine("arg1"));
    ++iter;

    FF2_ASSERT(iter != fun->arg_end());
    m_state.SetFeatureArgument(*iter);
    iter->setName(llvm::Twine("p_features"));
    ++iter;

    // Keep array return space reference.
    FF2_ASSERT(iter != fun->arg_end());
    llvm::Value &arraySpace = *iter;
    m_state.SetArrayReturnSpace(arraySpace);
    iter->setName(llvm::Twine("p_output"));
    ++iter;
    FF2_ASSERT(iter == fun->arg_end());
  } else if (m_destinationFunctionType ==
             CompilerFactory::DocumentSetEvaluation) {
    // Push function arguments onto the value stack.
    llvm::Function::arg_iterator iter = fun->arg_begin();
    FF2_ASSERT(iter != fun->arg_end());
    m_state.SetFeatureArrayPointer(*iter);
    iter->setName(llvm::Twine("p_featureArray"));
    ++iter;

    FF2_ASSERT(iter != fun->arg_end());
    m_state.SetAggregatedDocumentIndex(*iter);
    iter->setName(llvm::Twine("p_index"));
    ++iter;

    FF2_ASSERT(iter != fun->arg_end());
    m_state.SetAggregatedDocumentCount(*iter);
    iter->setName(llvm::Twine("p_count"));
    ++iter;

    FF2_ASSERT(iter != fun->arg_end());
    m_state.SetAggregatedCache(*iter);
    iter->setName(llvm::Twine("p_cache"));
    ++iter;
    FF2_ASSERT(iter == fun->arg_end());

    llvm::Value *featureArrayPointer = m_state.GetBuilder().CreateInBoundsGEP(
        &m_state.GetFeatureArrayPointer(),
        &m_state.GetAggregatedDocumentIndex(),
        llvm::Twine("init feature array"));
    CHECK_LLVM_RET(featureArrayPointer);

    llvm::Value *featureArray =
        m_state.GetBuilder().CreateLoad(featureArrayPointer);
    CHECK_LLVM_RET(featureArray);

    m_state.SetFeatureArgument(*featureArray);

    m_documentContextStack.push(featureArray);
  }

  return fun;
}

void FreeForm2::LlvmCodeGenerator::CreateAllocations() {
  typedef std::vector<const boost::shared_ptr<Allocation> > AllocationVector;
  for (size_t i = 0; i < m_allocations.size(); i++) {
    Allocate(*m_allocations[i]);
  }
}

FreeForm2::LlvmCodeGenerator::CompiledValue &
FreeForm2::LlvmCodeGenerator::CreateReturn(const TypeImpl &p_type) {
  CompiledValue &value = *m_returnValue;

  llvm::Value *ret = NULL;
  if (p_type.Primitive() == Type::Array) {
    llvm::Value &arraySpace = m_state.GetArrayReturnSpace();
    const ArrayType &arrayType = static_cast<const ArrayType &>(p_type);

    // Need to copy array result into array result argument, return bounds.
    ret = m_state.GetBuilder().CreateRet(
        &ArrayCodeGen::IssueReturn(m_state, value, arrayType, arraySpace));
  } else {
    ret = m_state.GetBuilder().CreateRet(&value);
  }
  CHECK_LLVM_RET(ret);
  return *ret;
}

bool FreeForm2::LlvmCodeGenerator::AlternativeVisit(
    const FeatureSpecExpression &p_expr) {
  // Generate the feature body.
  p_expr.GetBody().Accept(*this);

  return true;
}

void FreeForm2::LlvmCodeGenerator::Visit(const FeatureSpecExpression &p_expr) {
  // This is handled in AlternativeVisit
  Unreachable(__FILE__, __LINE__);
}

void FreeForm2::LlvmCodeGenerator::Visit(const FeatureGroupSpecExpression &) {
  // Not supported in FF2.
  Unreachable(__FILE__, __LINE__);
}

void FreeForm2::LlvmCodeGenerator::Visit(const ReturnExpression &) {
  // Not supported by FF2.
  FF2_UNREACHABLE();
}

void FreeForm2::LlvmCodeGenerator::Visit(const StreamDataExpression &) {
  // Not supported by FF2.
  FF2_UNREACHABLE();
}

void FreeForm2::LlvmCodeGenerator::Visit(const UpdateStreamDataExpression &) {
  // Not supported by FF2.
  FF2_UNREACHABLE();
}

void FreeForm2::LlvmCodeGenerator::Visit(const VariableRefExpression &p_expr) {
  llvm::Value *value = m_state.GetVariableValue(p_expr.GetId());
  FF2_ASSERT(value != NULL);
  m_stack.push(value);
}

void FreeForm2::LlvmCodeGenerator::VisitReference(
    const VariableRefExpression &) {
  // Not supported in FF2.
  FF2_UNREACHABLE();
}

void FreeForm2::LlvmCodeGenerator::VisitReference(const ThisExpression &) {
  // Not supported in FF2.
  FF2_UNREACHABLE();
}

void FreeForm2::LlvmCodeGenerator::VisitReference(
    const UnresolvedAccessExpression &) {
  // Not supported in FF2.
  FF2_UNREACHABLE();
}

// bool
// IsFeatureSpecMissing(const FreeForm2::Expression& p_expr)
//{
//     const FreeForm2::BlockExpression* block
//         = dynamic_cast<const FreeForm2::BlockExpression*>(&p_expr);
//     if (!block || block->GetNumChildren() <= 0)
//     {
//         return false;
//     }
//
//     const FreeForm2::NeuralInputResultExpression* input
//         = dynamic_cast<const
//         FreeForm2::NeuralInputResultExpression*>(&block->GetChild(0));
//     if (!input || input->GetNumChildren() <= 0)
//     {
//         return false;
//     }
//
//
//     const FreeForm2::FeatureSpecExpression* feature
//         = dynamic_cast<const
//         FreeForm2::FeatureSpecExpression*>(&input->m_child);
//
//     return feature == nullptr;
// }

void FreeForm2::LlvmCodeGenerator::Visit(const ImportFeatureExpression &) {
  // Not supported in FF2.
  FF2_UNREACHABLE();
}

void FreeForm2::LlvmCodeGenerator::Visit(const StateExpression &) {
  // Not supported in FF2.
  FF2_UNREACHABLE();
}

void FreeForm2::LlvmCodeGenerator::Visit(const StateMachineExpression &) {
  // Not supported in FF2.
  FF2_UNREACHABLE();
}

void FreeForm2::LlvmCodeGenerator::Visit(const ExecuteMachineExpression &) {
  // Not supported in FF2.
  FF2_UNREACHABLE();
}

void FreeForm2::LlvmCodeGenerator::Visit(
    const ExecuteStreamRewritingStateMachineGroupExpression &) {
  // Not supported in FF2.
  FF2_UNREACHABLE();
}

void FreeForm2::LlvmCodeGenerator::Visit(
    const ExecuteMachineGroupExpression &) {
  // Not supported in FF2.
  FF2_UNREACHABLE();
}

void FreeForm2::LlvmCodeGenerator::Visit(const YieldExpression &) {
  // Not supported in FF2.
  FF2_UNREACHABLE();
}

void FreeForm2::LlvmCodeGenerator::Visit(const RandFloatExpression &p_expr) {
  CompiledValue &randValue =
      CreateRandomFloat(m_state, &m_state.GetType(p_expr.GetType()));
  m_stack.push(&randValue);
}

void FreeForm2::LlvmCodeGenerator::Visit(const RandIntExpression &p_expr) {
  llvm::IRBuilder<> &builder = m_state.GetBuilder();
  CompiledValue *upperBound = m_stack.top();
  m_stack.pop();

  CompiledValue *lowerBound = m_stack.top();
  m_stack.pop();

  CompiledValue &randValue = CreateRandomFloat(m_state, nullptr);

  CompiledValue *range = builder.CreateSub(upperBound, lowerBound);
  CHECK_LLVM_RET(range);

  CompiledValue *floatRange = builder.CreateSIToFP(range, randValue.getType());
  CHECK_LLVM_RET(floatRange);

  CompiledValue *valInRange = builder.CreateFMul(floatRange, &randValue);
  CHECK_LLVM_RET(valInRange);

  CompiledValue *intValInRange =
      builder.CreateFPToSI(valInRange, &m_state.GetType(p_expr.GetType()));
  CHECK_LLVM_RET(intValInRange);

  CompiledValue *finalValue = builder.CreateAdd(intValInRange, lowerBound);
  CHECK_LLVM_RET(finalValue);

  m_stack.push(finalValue);
}

void FreeForm2::LlvmCodeGenerator::Visit(const ThisExpression &) {
  // Not supported in FF2.
  FF2_UNREACHABLE();
}

void FreeForm2::LlvmCodeGenerator::Visit(const UnresolvedAccessExpression &) {
  // Not supported in FF2.
  FF2_UNREACHABLE();
}

void FreeForm2::LlvmCodeGenerator::Visit(const TypeInitializerExpression &) {
  // Not supported in FF2.
  FF2_UNREACHABLE();
}

void FreeForm2::LlvmCodeGenerator::Visit(const AggregateContextExpression &) {
  // TODO: Implement me.
  FF2_UNREACHABLE();
}

void FreeForm2::LlvmCodeGenerator::Visit(const DebugExpression &) {
  // Not supported in FreeForm2.
  FF2_UNREACHABLE();
}

llvm::Function &FreeForm2::LlvmCodeGenerator::Compile(
    const Expression &p_expr, CompilationState &p_state,
    const AllocationVector &p_allocations,
    CompilerFactory::DestinationFunctionType p_destinationFunctionType) {
  LlvmCodeGenerator visitor(p_state, p_allocations, p_destinationFunctionType);
  visitor.m_returnType = &p_expr.GetType();
  visitor.m_function = visitor.CreateFeatureFunction(*visitor.m_returnType);
  visitor.CreateAllocations();

  p_expr.Accept(visitor);

  if (!visitor.m_returnValue) {
    visitor.m_returnValue = visitor.m_stack.top();
  }

  visitor.CreateReturn(*visitor.m_returnType);
  FF2_ASSERT(llvm::Function::classof(visitor.m_function));
  return *visitor.m_function;
}
