/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include "PhiNode.h"

#include "FreeForm2Assert.h"
#include "Visitor.h"

boost::shared_ptr<FreeForm2::PhiNodeExpression>
FreeForm2::PhiNodeExpression::Alloc(const Annotations &p_annotations,
                                    size_t p_version,
                                    size_t p_incomingVersionsCount,
                                    const size_t *p_incomingVersions) {
  FF2_ASSERT(p_incomingVersionsCount > 0);

  size_t bytes = sizeof(PhiNodeExpression) +
                 (p_incomingVersionsCount - 1) * sizeof(unsigned long long);

  // Allocate a shared_ptr that deletes an BlockExpression
  // allocated in a char[].
  boost::shared_ptr<PhiNodeExpression> exp(
      new (new char[bytes])
          PhiNodeExpression(p_annotations, p_version, p_incomingVersionsCount,
                            p_incomingVersions),
      DeleteAlloc);
  return exp;
}

const FreeForm2::TypeImpl &FreeForm2::PhiNodeExpression::GetType() const {
  return FreeForm2::TypeImpl::GetVoidInstance();
}

size_t FreeForm2::PhiNodeExpression::GetNumChildren() const { return 0; }

void FreeForm2::PhiNodeExpression::Accept(Visitor &p_visitor) const {
  size_t stackSize = p_visitor.StackSize();

  if (!p_visitor.AlternativeVisit(*this)) {
    p_visitor.Visit(*this);
  }

  FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}

size_t FreeForm2::PhiNodeExpression::GetVersion() const { return m_version; }

size_t FreeForm2::PhiNodeExpression::GetIncomingVersionsCount() const {
  return m_incomingVersionsCount;
}

const size_t *FreeForm2::PhiNodeExpression::GetIncomingVersions() const {
  return m_incomingVersions;
}

FreeForm2::PhiNodeExpression::PhiNodeExpression(
    const Annotations &p_annotations, size_t p_version,
    size_t p_incomingVersionsCount, const size_t *p_incomingVersions)
    : Expression(p_annotations),
      m_version(p_version),
      m_incomingVersionsCount(p_incomingVersionsCount) {
  // We rely on the custom allocator Alloc to provide enough space
  // for all of the incomings.
  for (unsigned int i = 0; i < m_incomingVersionsCount; i++) {
    m_incomingVersions[i] = p_incomingVersions[i];
  }
}

void FreeForm2::PhiNodeExpression::DeleteAlloc(PhiNodeExpression *p_allocated) {
  // Manually call dtor for phi node expression.
  p_allocated->~PhiNodeExpression();

  // Dispose of memory, which we allocated in a char[].
  char *mem = reinterpret_cast<char *>(p_allocated);
  delete[] mem;
}
