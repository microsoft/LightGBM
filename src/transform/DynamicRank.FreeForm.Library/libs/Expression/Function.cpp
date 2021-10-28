/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include "Function.h"

#include "FreeForm2Assert.h"
#include "FunctionType.h"
#include "RefExpression.h"
#include "Visitor.h"

FreeForm2::FunctionExpression::FunctionExpression(
    const Annotations &p_annotations, const FunctionType &p_type,
    const std::string &p_name, const std::vector<Parameter> &p_parameters,
    const Expression &p_body)
    : Expression(p_annotations),
      m_type(p_type),
      m_name(p_name),
      m_parameters(p_parameters),
      m_body(p_body) {}

const FreeForm2::FunctionType &FreeForm2::FunctionExpression::GetFunctionType()
    const {
  return m_type;
}

const FreeForm2::TypeImpl &FreeForm2::FunctionExpression::GetType() const {
  return m_type;
}

const std::string &FreeForm2::FunctionExpression::GetName() const {
  return m_name;
}

size_t FreeForm2::FunctionExpression::GetNumChildren() const {
  return GetNumParameters() + 1;
}

size_t FreeForm2::FunctionExpression::GetNumParameters() const {
  return m_parameters.size();
}

const FreeForm2::Expression &FreeForm2::FunctionExpression::GetBody() const {
  return m_body;
}

const std::vector<FreeForm2::FunctionExpression::Parameter>
    &FreeForm2::FunctionExpression::GetParameters() const {
  return m_parameters;
}

void FreeForm2::FunctionExpression::Accept(Visitor &p_visitor) const {
  size_t stackSize = p_visitor.StackSize();

  if (!p_visitor.AlternativeVisit(*this)) {
    for (size_t i = 0; i < m_parameters.size(); i++) {
      m_parameters[i].m_parameter->Accept(p_visitor);
    }

    m_body.Accept(p_visitor);

    p_visitor.Visit(*this);
  }

  FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}

FreeForm2::FunctionCallExpression::FunctionCallExpression(
    const Annotations &p_annotations, const Expression &p_function,
    const std::vector<const Expression *> &p_parameters)
    : Expression(p_annotations),
      m_function(p_function),
      m_numParameters(p_parameters.size()) {
  m_type = static_cast<const FunctionType *>(&p_function.GetType());

  // Note that we rely on this ctor not throwing exceptions during
  // allocation below.

  // We rely on our allocator to size this object to be big enough to
  // hold all children, and enforce this forcing construction via Alloc.
  for (size_t i = 0; i < m_numParameters; i++) {
    m_parameters[i] = p_parameters[i];
  }
}

void FreeForm2::FunctionCallExpression::Accept(Visitor &p_visitor) const {
  size_t stackSize = p_visitor.StackSize();

  if (!p_visitor.AlternativeVisit(*this)) {
    m_function.Accept(p_visitor);

    for (size_t i = 0; i < m_numParameters; i++) {
      if (GetFunctionType().BeginParameters()[i]->IsConst()) {
        m_parameters[i]->Accept(p_visitor);
      } else {
        m_parameters[i]->AcceptReference(p_visitor);
      }
    }

    p_visitor.Visit(*this);
  }

  FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}

const FreeForm2::FunctionType &
FreeForm2::FunctionCallExpression::GetFunctionType() const {
  return *m_type;
}

const FreeForm2::Expression &FreeForm2::FunctionCallExpression::GetFunction()
    const {
  return m_function;
}

const FreeForm2::TypeImpl &FreeForm2::FunctionCallExpression::GetType() const {
  return m_type->GetReturnType();
}

size_t FreeForm2::FunctionCallExpression::GetNumChildren() const {
  return GetNumParameters() + 1;
}

size_t FreeForm2::FunctionCallExpression::GetNumParameters() const {
  return m_numParameters;
}

const FreeForm2::Expression *const *
FreeForm2::FunctionCallExpression::GetParameters() const {
  return &m_parameters[0];
}

boost::shared_ptr<FreeForm2::FunctionCallExpression>
FreeForm2::FunctionCallExpression::Alloc(
    const Annotations &p_annotations, const Expression &p_function,
    const std::vector<const Expression *> &p_parameters) {
  FF2_ASSERT(p_function.GetType().Primitive() == Type::Function);

  size_t bytes =
      sizeof(FunctionCallExpression) +
      (std::max((size_t)1ULL, p_parameters.size()) - 1) * sizeof(Expression *);

  // Allocate a shared_ptr that deletes an FunctionCallExpression
  // allocated in a char[].
  boost::shared_ptr<FunctionCallExpression> exp(
      new (new char[bytes])
          FunctionCallExpression(p_annotations, p_function, p_parameters),
      DeleteAlloc);
  return exp;
}

void FreeForm2::FunctionCallExpression::DeleteAlloc(
    FunctionCallExpression *p_allocated) {
  // Manually call dtor for operator expression.
  p_allocated->~FunctionCallExpression();

  // Dispose of memory, which we allocated in a char[].
  char *mem = reinterpret_cast<char *>(p_allocated);
  delete[] mem;
}

FreeForm2::ReturnExpression::ReturnExpression(const Annotations &p_annotations,
                                              const Expression &p_value)
    : Expression(p_annotations), m_value(p_value) {}

void FreeForm2::ReturnExpression::Accept(Visitor &p_visitor) const {
  size_t stackSize = p_visitor.StackSize();

  if (!p_visitor.AlternativeVisit(*this)) {
    m_value.Accept(p_visitor);

    p_visitor.Visit(*this);
  }

  FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}

size_t FreeForm2::ReturnExpression::GetNumChildren() const { return 1; }

const FreeForm2::TypeImpl &FreeForm2::ReturnExpression::GetType() const {
  return m_value.GetType().AsConstType();
}

const FreeForm2::Expression &FreeForm2::ReturnExpression::GetValue() const {
  return m_value;
}
