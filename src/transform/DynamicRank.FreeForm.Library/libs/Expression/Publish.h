/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#pragma once

#ifndef FREEFORM2_PUBLISH_H
#define FREEFORM2_PUBLISH_H

#include "Expression.h"

namespace FreeForm2 {
// The Publish expression will declare the value of a feature.
class PublishExpression : public Expression {
 public:
  // Create a Publish expression that declares the value of a feature.
  PublishExpression(const Annotations &p_annotations,
                    const std::string &p_featureName,
                    const Expression &p_value);

  // Methods inherited from Expression
  virtual void Accept(Visitor &) const override;
  virtual size_t GetNumChildren() const override;
  virtual const TypeImpl &GetType() const override;

  // Return the value of the expression.
  const Expression &GetValue() const;

  // Return the feature name of the expression.
  const std::string &GetFeatureName() const;

 private:
  // The value of the feature being published.
  const Expression &m_value;

  // The name of the feature being published.
  const std::string m_featureName;
};

// The DirectPublish expression will declare the value of an element in a
// feature array.
class DirectPublishExpression : public Expression {
 public:
  // Allocate and construct a new DirectPublishExpression.
  static boost::shared_ptr<DirectPublishExpression> Alloc(
      const Annotations &p_annotations, const std::string &p_featureName,
      const Expression **p_indices, const unsigned int p_numIndices,
      const Expression &p_value);

  // Methods inherited from Expression
  virtual void Accept(Visitor &) const override;
  virtual size_t GetNumChildren() const override;
  virtual const TypeImpl &GetType() const override;

  // Return the value of the expression.
  const Expression &GetValue() const;

  // Return the feature name of the expression.
  const std::string &GetFeatureName() const;

  // Return the number of indices for this array.
  unsigned int GetNumIndices() const;

  // Return a pointer to the list of indices.
  const Expression *const *GetIndices() const;

 private:
  // Create a Publish expression that declares the value of a feature.
  DirectPublishExpression(const Annotations &p_annotations,
                          const std::string &p_featureName,
                          const Expression **p_indices,
                          const unsigned int p_numIndices,
                          const Expression &p_value);

  // The value of the feature being published.
  const Expression &m_value;

  // The name of the feature being published.
  const std::string m_featureName;

  // The number of indices.
  const unsigned int m_numIndices;

  // The destructor for the struct hack.
  static void DeleteAlloc(DirectPublishExpression *p_allocated);

  // The indices of the array element to publish.
  // Allocated using the struct hack.
  const Expression *m_indices[1];
};
}  // namespace FreeForm2

#endif
