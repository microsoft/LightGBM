/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include "FreeForm2Result.h"

#include <boost/lexical_cast.hpp>
#include <iomanip>
#include <sstream>

#include "FreeForm2Assert.h"
#include "FreeForm2Tokenizer.h"
#include "ResultIteratorImpl.h"
#include "TypeImpl.h"
#include "TypeUtil.h"

using namespace FreeForm2;
using namespace boost;

FreeForm2::Result::~Result() {}

int FreeForm2::Result::Compare(const Result &p_other) const {
  if (!GetType().GetImplementation().IsSameAs(
          p_other.GetType().GetImplementation(), true)) {
    std::ostringstream err;
    err << "Mismatched compare between " << GetType() << " and "
        << p_other.GetType();
    throw std::runtime_error(err.str());
  }

  switch (GetType().Primitive()) {
    case Type::Bool: {
      bool left = GetBool();
      bool right = p_other.GetBool();

      if (left == right) {
        return 0;
      } else if (right) {
        return -1;
      } else {
        return 1;
      }
      break;
    }

    case Type::Int: {
      Result::IntType left = GetInt();
      Result::IntType right = p_other.GetInt();

      if (left < right) {
        return -1;
      } else if (left > right) {
        return 1;
      } else {
        return 0;
      }
      break;
    }

    case Type::UInt64: {
      Result::UInt64Type left = GetUInt64();
      Result::UInt64Type right = p_other.GetUInt64();

      if (left < right) {
        return -1;
      } else if (left > right) {
        return 1;
      } else {
        return 0;
      }
      break;
    }

    case Type::Int32: {
      int left = GetInt32();
      int right = p_other.GetInt32();

      if (left < right) {
        return -1;
      } else if (left > right) {
        return 1;
      } else {
        return 0;
      }
      break;
    }

    case Type::UInt32: {
      unsigned int left = GetUInt32();
      unsigned int right = p_other.GetUInt32();

      if (left < right) {
        return -1;
      } else if (left > right) {
        return 1;
      } else {
        return 0;
      }
      break;
    }

    case Type::Float: {
      return CompareFloat(GetFloat(), p_other.GetFloat());
    }

    case Type::Array: {
      ResultIterator leftPos = BeginArray();
      ResultIterator leftEnd = EndArray();
      ResultIterator rightPos = p_other.BeginArray();
      ResultIterator rightEnd = p_other.EndArray();

      while (leftPos != leftEnd && rightPos != rightEnd) {
        int cmp = leftPos->Compare(*rightPos);
        if (cmp != 0) {
          return cmp;
        }

        ++leftPos;
        ++rightPos;
      }

      if (leftPos == leftEnd && rightPos == rightEnd) {
        return 0;
      } else if (leftPos == leftEnd) {
        return -1;
      } else {
        return 1;
      }
      break;
    }

    default: {
      std::ostringstream err;
      err << "Comparison of unknown type '" << GetType() << "'";
      throw std::runtime_error(err.str());
    }
  }
}

void FreeForm2::Result::Print(std::ostream &p_out) const {
  switch (GetType().Primitive()) {
    case Type::Bool: {
      p_out << (GetBool() ? "true" : "false");
      break;
    }

    case Type::Int: {
      p_out << GetInt();
      break;
    }

    case Type::UInt64: {
      p_out << GetUInt64();
      break;
    }

    case Type::Int32: {
      p_out << GetInt32();
      break;
    }

    case Type::UInt32: {
      p_out << GetUInt32();
      break;
    }

    case Type::Float: {
      const std::streamsize savePrecision = p_out.precision();

      p_out << std::setprecision(9) << GetFloat();

      // Restore precision.
      p_out << std::setprecision(savePrecision);
      break;
    }

    case Type::Array: {
      p_out << "[";
      ResultIterator end = EndArray();
      bool first = true;
      for (ResultIterator iter = BeginArray(); iter != end; ++iter) {
        p_out << (first ? "" : " ");
        first = false;
        iter->Print(p_out);
      }
      p_out << "]";
      break;
    }

    default: {
      std::ostringstream err;
      err << "Printing unknown type '" << GetType() << "'";
      throw std::runtime_error(err.str());
    }
  }
}

std::ostream &FreeForm2::operator<<(std::ostream &p_out,
                                    const Result &p_result) {
  p_result.Print(p_out);
  return p_out;
}

int FreeForm2::Result::CompareFloat(FloatType p_left, FloatType p_right) {
  // This value was chosen to be compatible with the old freeforms.
  const Result::FloatType relativeError = 1E-6F;
  bool equal = false;

  // Check for identical values (this is needed to compare infinity).
  if (p_left == p_right) {
    return 0;
  }

  // Check whether right operand is small.
  if (p_right < relativeError && p_right > -relativeError) {
    // Right is small, so they're equal iff left is small.
    equal = (p_left < relativeError && p_left > -relativeError);
  } else {
    // Right isn't small, so check the difference between the two
    // (related to right operand).  They're equal iff the difference is
    // small.
    const Result::FloatType diff = (p_left - p_right) / p_right;
    equal = (diff < relativeError && diff > -relativeError);
  }

  if (equal) {
    return 0;
  } else if (p_left < p_right) {
    return -1;
  } else {
    return 1;
  }
}

FreeForm2::ResultIterator::ResultIterator(
    std::auto_ptr<ResultIteratorImpl> p_impl)
    : m_impl(p_impl) {}

FreeForm2::ResultIterator::ResultIterator(const ResultIterator &p_other)
    : m_impl(p_other.m_impl->Clone()) {}

FreeForm2::ResultIterator::~ResultIterator() {}

void FreeForm2::ResultIterator::increment() { return m_impl->increment(); }

void FreeForm2::ResultIterator::decrement() { return m_impl->decrement(); }

bool FreeForm2::ResultIterator::equal(const ResultIterator &p_other) const {
  return m_impl->Position() == p_other.m_impl->Position() &&
         m_impl->ElementSize() == p_other.m_impl->ElementSize();
}

const Result &FreeForm2::ResultIterator::dereference() const {
  return m_impl->dereference();
}

void FreeForm2::ResultIterator::advance(std::ptrdiff_t p_distance) {
  m_impl->advance(p_distance);
}

std::ptrdiff_t FreeForm2::ResultIterator::distance_to(
    const ResultIterator &p_other) const {
  FF2_ASSERT(m_impl->ElementSize() == p_other.m_impl->ElementSize());
  return p_other.m_impl->Position().second - m_impl->Position().second;
}
