/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include "BinaryOperator.h"

#include <vector>

#include "FreeForm2Assert.h"
#include "FreeForm2Type.h"
#include "TypeImpl.h"
#include "TypeUtil.h"

static std::vector<FreeForm2::Type::TypePrimitive> GetSupportedOperandTypes(
    FreeForm2::BinaryOperator::Operation p_op) {
  using FreeForm2::BinaryOperator;
  std::vector<FreeForm2::Type::TypePrimitive> types;
  types.reserve(3);

  switch (p_op) {
    // The following operators support only bool:
    case BinaryOperator::_and:
    case BinaryOperator::_or:
      types.push_back(FreeForm2::Type::Bool);
      break;

    // The following operators support bool, int and float:
    case BinaryOperator::eq:
    case BinaryOperator::neq:
      types.push_back(FreeForm2::Type::Bool);
      __attribute__((__fallthrough__));

    // The following operators support int and float:
    case BinaryOperator::lt:
      __attribute__((__fallthrough__));
    case BinaryOperator::lte:
      __attribute__((__fallthrough__));
    case BinaryOperator::gt:
      __attribute__((__fallthrough__));
    case BinaryOperator::gte:
      __attribute__((__fallthrough__));
    case BinaryOperator::max:
      __attribute__((__fallthrough__));
    case BinaryOperator::min:
      __attribute__((__fallthrough__));
    case BinaryOperator::plus:
      __attribute__((__fallthrough__));
    case BinaryOperator::minus:
      __attribute__((__fallthrough__));
    case BinaryOperator::multiply:
      __attribute__((__fallthrough__));
    case BinaryOperator::divides:
      __attribute__((__fallthrough__));
    case BinaryOperator::mod:
      __attribute__((__fallthrough__));
    case BinaryOperator::pow:
      __attribute__((__fallthrough__));
    case BinaryOperator::log:
      types.push_back(FreeForm2::Type::Int);
      types.push_back(FreeForm2::Type::UInt64);
      types.push_back(FreeForm2::Type::Int32);
      types.push_back(FreeForm2::Type::UInt32);
      types.push_back(FreeForm2::Type::Float);
      break;

    // Bit operations only support ints.
    case BinaryOperator::_bitand:
      __attribute__((__fallthrough__));
    case BinaryOperator::_bitor:
      __attribute__((__fallthrough__));
    case BinaryOperator::bitshiftleft:
      __attribute__((__fallthrough__));
    case BinaryOperator::bitshiftright:
      types.push_back(FreeForm2::Type::Int);
      types.push_back(FreeForm2::Type::UInt64);
      types.push_back(FreeForm2::Type::Int32);
      types.push_back(FreeForm2::Type::UInt32);
      break;

    default:
      FreeForm2::Unreachable(__FILE__, __LINE__);
  }

  return types;
}

const FreeForm2::TypeImpl &FreeForm2::BinaryOperator::GetBestOperandType(
    Operation p_operator, const TypeImpl &p_operandType) {
  if (p_operandType.Primitive() == Type::Unknown) {
    return p_operandType;
  }

  const std::vector<Type::TypePrimitive> types =
      GetSupportedOperandTypes(p_operator);

  if (std::find(types.begin(), types.end(), p_operandType.Primitive()) !=
      types.end()) {
    return p_operandType;
  } else {
    return TypeImpl::GetInvalidType();
  }
}

const FreeForm2::TypeImpl &FreeForm2::BinaryOperator::GetResultType(
    Operation p_operator, const TypeImpl &p_operandType) {
  switch (p_operator) {
    case BinaryOperator::max:
      __attribute__((__fallthrough__));
    case BinaryOperator::min:
      __attribute__((__fallthrough__));
    case BinaryOperator::_and:
      __attribute__((__fallthrough__));
    case BinaryOperator::_or:
      __attribute__((__fallthrough__));
    case BinaryOperator::plus:
      __attribute__((__fallthrough__));
    case BinaryOperator::minus:
      __attribute__((__fallthrough__));
    case BinaryOperator::multiply:
      __attribute__((__fallthrough__));
    case BinaryOperator::mod:
      __attribute__((__fallthrough__));
    case BinaryOperator::pow:
      __attribute__((__fallthrough__));
    case BinaryOperator::_bitand:
      __attribute__((__fallthrough__));
    case BinaryOperator::_bitor:
      __attribute__((__fallthrough__));
    case BinaryOperator::bitshiftleft:
      __attribute__((__fallthrough__));
    case BinaryOperator::bitshiftright:
      __attribute__((__fallthrough__));
    case BinaryOperator::divides:
      return p_operandType;

    case BinaryOperator::log:
      return TypeImpl::GetFloatInstance(true);

    case BinaryOperator::eq:
      __attribute__((__fallthrough__));
    case BinaryOperator::neq:
      __attribute__((__fallthrough__));
    case BinaryOperator::lt:
      __attribute__((__fallthrough__));
    case BinaryOperator::lte:
      __attribute__((__fallthrough__));
    case BinaryOperator::gt:
      __attribute__((__fallthrough__));
    case BinaryOperator::gte:
      return TypeImpl::GetBoolInstance(true);

    default:
      FreeForm2::Unreachable(__FILE__, __LINE__);
  }
}
