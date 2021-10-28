/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include "ValueResult.h"

#include <boost/lexical_cast.hpp>
#include <limits>

#include "ArrayType.h"
#include "FreeForm2Assert.h"
#include "FreeForm2Tokenizer.h"
#include "FreeForm2Type.h"
#include "FreeForm2Utils.h"
#include "ResultIteratorImpl.h"
#include "TypeManager.h"

using namespace FreeForm2;

namespace {
// __declspec(noreturn)
void FailParseResult(SIZED_STRING p_result) {
  std::ostringstream err;
  err << "Failed to parse value from result '" << p_result << "'";
  throw std::runtime_error(err.str());
}

bool ParseBoolean(SIZED_STRING p_str) {
  std::string value(p_str.pcData, p_str.cbData);

  if (value == "true") {
    return true;
  } else if (value == "false") {
    return false;
  } else {
    FailParseResult(p_str);
  }
}

class ValueResultIteratorImpl : public ResultIteratorImpl {
 public:
  ValueResultIteratorImpl(const boost::shared_ptr<ValueResult> *p_pos,
                          unsigned int p_idx)
      : m_pos(p_pos), m_idx(p_idx) {}

  virtual ~ValueResultIteratorImpl() {}

 private:
  virtual void increment() {
    m_pos++;
    m_idx++;
  }

  virtual void decrement() {
    m_pos--;
    m_idx--;
  }

  virtual const Result &dereference() const { return **m_pos; }

  virtual void advance(std::ptrdiff_t p_distance) {
    m_pos += p_distance;
    m_idx += static_cast<unsigned int>(p_distance);
  }

  virtual std::auto_ptr<ResultIteratorImpl> Clone() const {
    return std::auto_ptr<ResultIteratorImpl>(
        new ValueResultIteratorImpl(m_pos, m_idx));
  }

  virtual std::pair<const char *, unsigned int> Position() const {
    return std::make_pair(reinterpret_cast<const char *>(m_pos), m_idx);
  }

  virtual unsigned int ElementSize() const { return sizeof(*m_pos); }

  const boost::shared_ptr<ValueResult> *m_pos;

  unsigned int m_idx;
};
}  // namespace

FreeForm2::ValueResult::ValueResult(FloatType p_float)
    : m_typeImpl(&TypeImpl::GetFloatInstance(true)), m_type(*m_typeImpl) {
  m_val.m_float = p_float;
}

FreeForm2::ValueResult::ValueResult(IntType p_int)
    : m_typeImpl(&TypeImpl::GetIntInstance(true)), m_type(*m_typeImpl) {
  m_val.m_int = p_int;
}

FreeForm2::ValueResult::ValueResult(UInt64Type p_int)
    : m_typeImpl(&TypeImpl::GetUInt64Instance(true)), m_type(*m_typeImpl) {
  m_val.m_uint64 = p_int;
}

FreeForm2::ValueResult::ValueResult(int p_int)
    : m_typeImpl(&TypeImpl::GetInt32Instance(true)), m_type(*m_typeImpl) {
  m_val.m_int32 = p_int;
}

FreeForm2::ValueResult::ValueResult(unsigned int p_int)
    : m_typeImpl(&TypeImpl::GetUInt32Instance(true)), m_type(*m_typeImpl) {
  m_val.m_uint32 = p_int;
}

FreeForm2::ValueResult::ValueResult(bool p_bool)
    : m_typeImpl(&TypeImpl::GetBoolInstance(true)), m_type(*m_typeImpl) {
  m_val.m_bool = p_bool;
}

FreeForm2::ValueResult::ValueResult(
    const ArrayType &p_arrayType,
    const std::vector<boost::shared_ptr<ValueResult> > &p_elements,
    unsigned int p_numElements)
    : m_typeImpl(&p_arrayType), m_type(*m_typeImpl), m_array(p_elements) {
  m_val.m_array.m_elements = m_array.empty() ? NULL : &m_array[0];
  m_val.m_array.m_numElements = p_numElements;
}

boost::shared_ptr<Result> FreeForm2::ValueResult::Parse(
    SIZED_STRING p_result, TypeManager &p_typeManager) {
  Tokenizer tok(p_result);
  boost::shared_ptr<ValueResult> result = Parse(tok, p_result, p_typeManager);

  // Check that there's no trailing junk.
  if (tok.GetToken() != TOKEN_END) {
    FailParseResult(p_result);
  }

  return boost::static_pointer_cast<Result>(result);
}

FreeForm2::ValueResult::~ValueResult() {}

const FreeForm2::Type &FreeForm2::ValueResult::GetType() const {
  return m_type;
}

FreeForm2::Result::IntType FreeForm2::ValueResult::GetInt() const {
  FF2_ASSERT(GetType().Primitive() == Type::Int);
  return m_val.m_int;
}

FreeForm2::Result::UInt64Type FreeForm2::ValueResult::GetUInt64() const {
  FF2_ASSERT(GetType().Primitive() == Type::UInt64);
  return m_val.m_uint64;
}

int FreeForm2::ValueResult::GetInt32() const {
  FF2_ASSERT(GetType().Primitive() == Type::Int32);
  return m_val.m_int32;
}

unsigned int FreeForm2::ValueResult::GetUInt32() const {
  FF2_ASSERT(GetType().Primitive() == Type::UInt32);
  return m_val.m_uint32;
}

FreeForm2::Result::FloatType FreeForm2::ValueResult::GetFloat() const {
  FF2_ASSERT(GetType().Primitive() == Type::Float);
  return m_val.m_float;
}

bool FreeForm2::ValueResult::GetBool() const {
  FF2_ASSERT(GetType().Primitive() == Type::Bool);
  return m_val.m_bool;
}

FreeForm2::ResultIterator FreeForm2::ValueResult::BeginArray() const {
  FF2_ASSERT(GetType().Primitive() == Type::Array);
  return ResultIterator(std::auto_ptr<ResultIteratorImpl>(
      new ValueResultIteratorImpl(m_val.m_array.m_elements, 0)));
}

FreeForm2::ResultIterator FreeForm2::ValueResult::EndArray() const {
  FF2_ASSERT(GetType().Primitive() == Type::Array);
  return ResultIterator(
      std::auto_ptr<ResultIteratorImpl>(new ValueResultIteratorImpl(
          m_val.m_array.m_elements + m_val.m_array.m_numElements,
          m_val.m_array.m_numElements)));
}

boost::shared_ptr<ValueResult> FreeForm2::ValueResult::Parse(
    Tokenizer &p_tok, SIZED_STRING p_original, TypeManager &p_typeManager) {
  Token token = p_tok.GetToken();
  boost::shared_ptr<ValueResult> result;

  switch (token) {
    case TOKEN_INT: {
      std::string value(p_tok.GetValue().pcData, p_tok.GetValue().cbData);
      result.reset(
          new ValueResult(boost::lexical_cast<Result::IntType>(value)));

      // Consume token.
      p_tok.Advance();
      break;
    }

    case TOKEN_FLOAT: {
      std::string value(p_tok.GetValue().pcData, p_tok.GetValue().cbData);
      result.reset(
          new ValueResult(boost::lexical_cast<Result::FloatType>(value)));

      // Consume token.
      p_tok.Advance();
      break;
    }

    case TOKEN_ATOM: {
      std::string value(p_tok.GetValue().pcData, p_tok.GetValue().cbData);

      if (value == "infinity") {
        result.reset(new ValueResult(
            std::numeric_limits<Result::FloatType>::infinity()));
      } else if (value == "-") {
        // The only valid possibility here would be -infinity.
        p_tok.Advance();

        value = std::string(p_tok.GetValue().pcData, p_tok.GetValue().cbData);

        if (p_tok.GetToken() == TOKEN_ATOM && value == "infinity") {
          result.reset(new ValueResult(
              -std::numeric_limits<Result::FloatType>::infinity()));
        } else {
          std::ostringstream err;
          err << "Failed to parse value from result '-'.";
          throw std::runtime_error(err.str());
        }
      } else {
        // Freeforms handle literal booleans as atoms, but we need to
        // recognise them here so we can produce literal boolean values.
        result.reset(new ValueResult(ParseBoolean(p_tok.GetValue())));
      }

      // Consume token.
      p_tok.Advance();
      break;
    }

    case TOKEN_OPEN_ARRAY: {
      unsigned int numElements = 0;
      std::vector<boost::shared_ptr<ValueResult> > array;
      p_tok.Advance();

      // Parse first element, if any.
      unsigned int sumElements = 0;
      unsigned int subDimensions = 0;
      Type::TypePrimitive basePrimitive = Type::Unknown;
      if (p_tok.GetToken() != TOKEN_CLOSE_ARRAY) {
        // Parse array element.
        array.push_back(boost::shared_ptr<ValueResult>(
            Parse(p_tok, p_original, p_typeManager)));
        ValueResult &curr = *array.back();

        if (curr.GetType().Primitive() != Type::Array) {
          sumElements += 1;
          subDimensions = 0;
          basePrimitive = curr.GetType().Primitive();
        } else {
          const ArrayType &arrayType = static_cast<const ArrayType &>(
              curr.GetType().GetImplementation());
          sumElements += arrayType.GetMaxElements();
          basePrimitive = arrayType.GetChildType().Primitive();
          subDimensions = arrayType.GetDimensionCount();
        }

        numElements++;
      }

      // Parse subsequent elements, if any.
      while (p_tok.GetToken() != TOKEN_CLOSE_ARRAY) {
        // Parse array element.
        array.push_back(boost::shared_ptr<ValueResult>(
            Parse(p_tok, p_original, p_typeManager)));
        ValueResult &curr = *array.back();

        if (curr.GetType().Primitive() != Type::Array) {
          sumElements += 1;
          if (subDimensions != 0) {
            std::ostringstream err;
            err << "Previous array element was an array, current is not.";
            throw std::runtime_error(err.str());
          }

          if (basePrimitive != curr.GetType().Primitive()) {
            std::ostringstream err;
            err << "Previous array element was a " << basePrimitive
                << ", current is a " << curr.GetType().Primitive() << ".";
            throw std::runtime_error(err.str());
          }
        } else {
          const ArrayType &arrayType = static_cast<const ArrayType &>(
              curr.GetType().GetImplementation());
          sumElements += arrayType.GetMaxElements();

          if (subDimensions != arrayType.GetDimensionCount()) {
            std::ostringstream err;
            err << "Previous array element was a " << subDimensions
                << "-dimensional array, current is "
                << arrayType.GetDimensionCount() << "-dimensional";
            throw std::runtime_error(err.str());
          }

          if (basePrimitive != arrayType.GetChildType().Primitive()) {
            std::ostringstream err;
            err << "Previous array element contained " << basePrimitive
                << ", current contains " << curr.GetType().Primitive() << ".";
            throw std::runtime_error(err.str());
          }
        }

        numElements++;
      }

      const ArrayType &type = p_typeManager.GetArrayType(
          TypeImpl::GetCommonType(basePrimitive, true), true, subDimensions + 1,
          sumElements);

      result.reset(new ValueResult(type, array, numElements));

      // Remove array close before proceeding.
      p_tok.Advance();
      break;
    }

    case TOKEN_CLOSE_ARRAY:
    case TOKEN_END:
    case TOKEN_OPEN:
    case TOKEN_CLOSE: {
      FailParseResult(p_original);
    }

    default: {
      Unreachable(__FILE__, __LINE__);
    }
  }
  return result;
}
