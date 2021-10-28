/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include "Expression.h"

#include <basic_types.h>

#include <limits>
#include <sstream>

#include "FreeForm2Assert.h"

using namespace FreeForm2;

const FreeForm2::VariableID FreeForm2::VariableID::c_invalidID = {MAX_UINT32};

bool FreeForm2::VariableID::operator==(VariableID p_other) const {
  return m_value == p_other.m_value;
}

bool FreeForm2::VariableID::operator!=(VariableID p_other) const {
  return m_value != p_other.m_value;
}

bool FreeForm2::VariableID::operator<(VariableID p_other) const {
  return m_value < p_other.m_value;
}

FreeForm2::Result::IntType FreeForm2::ConstantValue::GetInt(
    const TypeImpl &p_type) const {
  switch (p_type.Primitive()) {
    case Type::Int:
      return m_int;
    case Type::UInt64:
      return m_uint64;
    case Type::Int32:
      return m_int32;
    case Type::UInt32:
      return m_uint32;
    default:
      Unreachable(__FILE__, __LINE__);
  }
}

FreeForm2::Expression::Expression(const Annotations &p_annotations)
    : m_annotations(p_annotations) {}

FreeForm2::Expression::~Expression() {}

void FreeForm2::Expression::AcceptReference(Visitor &) const {
  throw ParseError("Invalid l-value", GetSourceLocation());
}

bool FreeForm2::Expression::IsConstant() const { return false; }

FreeForm2::ConstantValue FreeForm2::Expression::GetConstantValue() const {
  // The following assertion should fail, unless a child class implemented
  // IsConstant without overriding GetConstantValue, in which case the
  // second assertion will trip.
  FF2_ASSERT(IsConstant());
  FF2_ASSERT(false &&
             "Expression-type class must override both IsConstant and "
             "GetConstantValue");
  Unreachable(__FILE__, __LINE__);
}

const FreeForm2::ValueBounds &FreeForm2::Expression::GetValueBounds() const {
  return m_annotations.m_valueBounds;
}

const FreeForm2::SourceLocation &FreeForm2::Expression::GetSourceLocation()
    const {
  return m_annotations.m_sourceLocation;
}

const FreeForm2::Annotations &FreeForm2::Expression::GetAnnotations() const {
  return m_annotations;
}

FreeForm2::ExpressionOwner::~ExpressionOwner() {}

FreeForm2::ValueBounds::ValueBounds()
    : m_lower(std::numeric_limits<Result::IntType>::min()),
      m_upper(std::numeric_limits<Result::IntType>::max()) {}

FreeForm2::ValueBounds::ValueBounds(const TypeImpl &p_type) {
  switch (p_type.Primitive()) {
    case Type::Int32: {
      m_lower = std::numeric_limits<Result::Int32Type>::min();
      m_upper = std::numeric_limits<Result::Int32Type>::max();
      break;
    }

    case Type::UInt32: {
      m_lower = std::numeric_limits<Result::UInt32Type>::min();
      m_upper = std::numeric_limits<Result::UInt32Type>::max();
      break;
    }

    default: {
      m_lower = std::numeric_limits<Result::IntType>::min();
      m_upper = std::numeric_limits<Result::IntType>::max();
      break;
    }
  }
}

FreeForm2::ValueBounds::ValueBounds(const TypeImpl &p_type,
                                    ConstantValue p_value) {
  switch (p_type.Primitive()) {
    case Type::Int32: {
      m_lower = m_upper = p_value.m_int32;
      break;
    }

    case Type::UInt32: {
      m_lower = m_upper = p_value.m_uint32;
      break;
    }

    case Type::Int: {
      m_lower = m_upper = p_value.m_int;
      break;
    }

    case Type::UInt64: {
      if (p_value.m_uint64 <=
          static_cast<Result::UInt64Type>(
              std::numeric_limits<Result::IntType>::max())) {
        m_lower = m_upper = static_cast<Result::IntType>(p_value.m_uint64);
      } else {
        m_lower = std::numeric_limits<Result::IntType>::min();
        m_upper = std::numeric_limits<Result::IntType>::max();
      }
      break;
    }

    default: {
      m_lower = std::numeric_limits<Result::IntType>::min();
      m_upper = std::numeric_limits<Result::IntType>::max();
      break;
    }
  }
}

FreeForm2::ValueBounds::ValueBounds(FreeForm2::Result::IntType p_lower,
                                    FreeForm2::Result::IntType p_upper)
    : m_lower(p_lower), m_upper(p_upper) {}

bool FreeForm2::ValueBounds::operator==(
    const FreeForm2::ValueBounds &p_other) const {
  return m_lower == p_other.m_lower && m_upper == p_other.m_upper;
}

bool FreeForm2::ValueBounds::operator!=(
    const FreeForm2::ValueBounds &p_other) const {
  return m_lower != p_other.m_lower || m_upper != p_other.m_upper;
}

const FreeForm2::ValueBounds FreeForm2::ValueBounds::c_empty(
    std::numeric_limits<Result::IntType>::max(),
    std::numeric_limits<Result::IntType>::min());

FreeForm2::Annotations::Annotations() {}

FreeForm2::Annotations::Annotations(FreeForm2::SourceLocation p_sourceLocation)
    : m_sourceLocation(p_sourceLocation) {}

FreeForm2::Annotations::Annotations(FreeForm2::SourceLocation p_sourceLocation,
                                    FreeForm2::ValueBounds p_valueBounds)
    : m_sourceLocation(p_sourceLocation), m_valueBounds(p_valueBounds) {}
