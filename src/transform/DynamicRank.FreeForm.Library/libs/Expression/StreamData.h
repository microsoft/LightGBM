/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#pragma once

#ifndef FREEFORM2_STREAM_DATA_H
#define FREEFORM2_STREAM_DATA_H

#include "Expression.h"

namespace FreeForm2 {
// StreamDataExpression pulls either the stream count (opaque integer data
// field) or the length (calculated length of the stream instance) from
// currently matched stream instance.  These have traditionally been called
// PhraseCount/Length, or 'click phrase' data.  However, i've avoided those
// names because they're inaccurate and incomplete (as metastreams are a
// broader abstraction than click phrases, and the data in the metastream
// instance has no connection to phrases outside of click metastreams).
class StreamDataExpression : public Expression {
 public:
  StreamDataExpression(const Annotations &p_annotations, bool p_requestsLength);

  // Virtual methods inherited from Expression.
  virtual void Accept(Visitor &p_visitor) const override;
  virtual const TypeImpl &GetType() const override;
  virtual size_t GetNumChildren() const override;

  // Whether this expression requests the stream length, with the
  // alternative being the stream count.
  bool m_requestsLength;
};

// Singleton expression class to issue side-effect code to update the
// stream data (length, count) during matching.  This requires an
// expression to leverage the FSM architecture to decide when this
// must occur.  If we accumulate other similar expressions, they
// can be aggregated into a SystemEffectExpression, or similar.
class UpdateStreamDataExpression : public Expression {
 public:
  static const UpdateStreamDataExpression &GetInstance();

  // Virtual methods inherited from Expression.
  virtual void Accept(Visitor &p_visitor) const override;
  virtual const TypeImpl &GetType() const override;
  virtual size_t GetNumChildren() const override;

 private:
  UpdateStreamDataExpression(const Annotations &p_annotations);
};
};  // namespace FreeForm2

#endif
