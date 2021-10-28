/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#pragma once

#ifndef FREEFORM2_VALUE_RESULT_H
#define FREEFORM2_VALUE_RESULT_H

#include <boost/shared_ptr.hpp>
#include <vector>

#include "Expression.h"
#include "FreeForm2Result.h"
#include "FreeForm2Tokenizer.h"
#include "FreeForm2Type.h"

namespace FreeForm2 {
class ArrayType;
class TypeManager;

class ValueResult : public Result {
 public:
  ValueResult(FloatType p_float);
  ValueResult(IntType p_int);
  ValueResult(UInt64Type p_int);
  ValueResult(Int32Type p_int);
  ValueResult(UInt32Type p_int);
  ValueResult(BoolType p_bool);

  // Parse a result from a string.
  static boost::shared_ptr<Result> Parse(SIZED_STRING p_result,
                                         TypeManager &p_typeManager);

  virtual ~ValueResult();

  virtual const Type &GetType() const override;
  virtual IntType GetInt() const override;
  virtual UInt64Type GetUInt64() const override;
  virtual Int32Type GetInt32() const override;
  virtual UInt32Type GetUInt32() const override;
  virtual FloatType GetFloat() const override;
  virtual BoolType GetBool() const override;
  virtual ResultIterator BeginArray() const override;
  virtual ResultIterator EndArray() const override;

 private:
  // Construct an array ValueResult; only called internally from Parse.
  ValueResult(const ArrayType &p_arrayType,
              const std::vector<boost::shared_ptr<ValueResult> > &p_elements,
              unsigned int p_numElements);

  // Internal method to parse a ValueResult from a string tokenizer.
  // p_original gives the original string that is being parsed, for use in
  // error messages.
  static boost::shared_ptr<ValueResult> Parse(Tokenizer &p_tok,
                                              SIZED_STRING p_original,
                                              TypeManager &p_typeManager);

  // Type of result.
  const TypeImpl *m_typeImpl;
  Type m_type;

  // Structure (no constructor, destructor so that it can be used in below
  // union) to represent array values.  Note that an array at this level
  // is simply a series of elements with a length (and has none of the
  // restrictions that our implementation might, such as requiring
  // 'square' arrays, or limiting dimensions).
  struct ArrayVal {
    const boost::shared_ptr<ValueResult> *m_elements;
    unsigned int m_numElements;
  };

  // Val defines the different value types used by freeform expressions.
  union Val {
    Result::FloatType m_float;
    Result::IntType m_int;
    Result::UInt64Type m_uint64;
    Result::Int32Type m_int32;
    Result::UInt32Type m_uint32;
    Result::BoolType m_bool;
    ArrayVal m_array;
  };

  // Result value.
  Val m_val;

  // Container to hold array elements, if any.  We separate this from the
  // value above to avoid union issues.
  std::vector<boost::shared_ptr<ValueResult> > m_array;
};
};  // namespace FreeForm2

#endif
