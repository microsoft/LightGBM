/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include "FeatureSpec.h"

#include <basic_types.h>

#include <boost/algorithm/string.hpp>
#include <boost/bind.hpp>
#include <boost/foreach.hpp>
#include <boost/iterator.hpp>
#include <boost/range.hpp>
#include <functional>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "ArrayType.h"
#include "FreeForm2Assert.h"
#include "MigratedApi.h"
#include "TypeManager.h"
#include "TypeUtil.h"
#include "Visitor.h"

namespace {
void EscapeString(std::string &p_str) {
  static const char c_escapeSequences[][3] = {"\\\"", "\\n", "\\t", "\\\\"};
  static const char c_replaceSequences[][2] = {"\"", "\n", "\t", "\\"};
  static_assert(countof(c_escapeSequences) == countof(c_replaceSequences),
                "Replacement sequence arrays must be parallel");

  typedef boost::iterator_range<const char *> ConstCharPtrRange;
  for (size_t i = 0; i < countof(c_escapeSequences); i++) {
    // Construct string objects to prevent the compiler from whining
    // about unsafe parameters.
    const std::string escapeSequence(c_escapeSequences[i]);
    const std::string replaceSequence(c_replaceSequences[i]);
    boost::replace_all(p_str, escapeSequence, replaceSequence);
  }
}
}  // namespace

FreeForm2::FeatureSpecExpression::FeatureName::FeatureName()
    : m_isParameterized(false) {}

FreeForm2::FeatureSpecExpression::FeatureName::FeatureName(
    const std::string &p_name)
    : m_name(p_name), m_isParameterized(false) {}

FreeForm2::FeatureSpecExpression::FeatureName::FeatureName(
    const std::string &p_name, const ParameterMap &p_parameters)
    : m_name(p_name), m_params(p_parameters), m_isParameterized(true) {}

FreeForm2::FeatureSpecExpression::FeatureName
FreeForm2::FeatureSpecExpression::FeatureName::Parse(
    const std::string &p_name, bool p_isParameterized,
    const std::string &p_parameterization, const SourceLocation &p_location) {
  if (!p_isParameterized) {
    return FeatureName(p_name);
  } else {
    FeatureName name(p_name);
    name.m_isParameterized = true;
    if (p_parameterization.size() > 0) {
      try {
        std::string params = p_parameterization;
        EscapeString(params);

        FF2_ASSERT(params.front() == '"' && params.back() == '"');
        auto paramRange =
            boost::make_iterator_range(params.cbegin() + 1, params.cend() - 1);

        std::vector<std::string> values;
        for (auto iter = boost::make_split_iterator(paramRange,
                                                    boost::first_finder(","));
             !iter.eof(); ++iter) {
          typedef boost::iterator_range<std::string::const_iterator>
              StringRange;
          const StringRange range = *iter;
          values.clear();

          // Note that boost::equals does approximately the same thing
          // as the bind expression, but for some reason it causes a
          // compiler warning about unsafe parameters.
          boost::split(values, range,
                       boost::bind(&boost::is_equal::operator()<char, char>,
                                   boost::is_equal(), _1, '='));
          FF2_ASSERT(values.size() == 2);

          const std::string &paramName = values[0];
          const std::string &paramValue = values[1];
          name.m_params.insert(std::make_pair(paramName, paramValue));
        }
        return name;
      } catch (...) {
        std::ostringstream err;
        err << "Unable to parse parameterization " << p_parameterization
            << "; expected format is "
               "\"<parameter_name>=<parameter_value>,...\"";
        throw FreeForm2::ParseError(err.str(), p_location);
      }
    } else {
      return name;
    }
  }
}

const std::string &FreeForm2::FeatureSpecExpression::FeatureName::GetName()
    const {
  return m_name;
}

const FreeForm2::FeatureSpecExpression::FeatureName::ParameterMap &
FreeForm2::FeatureSpecExpression::FeatureName::GetParameters() const {
  return m_params;
}

bool FreeForm2::FeatureSpecExpression::FeatureName::IsParameterized() const {
  return m_isParameterized;
}

FreeForm2::SymbolTable::Symbol
FreeForm2::FeatureSpecExpression::FeatureName::GetSymbol() const {
  if (m_isParameterized) {
    if (m_paramStr.empty()) {
      std::ostringstream out;
      out << "\"";
      bool first = true;
      BOOST_FOREACH (const Parameter &param, m_params) {
        if (!first) {
          out << ',';
        }
        out << param.first << '=' << param.second;
        first = false;
      }
      out << "\"";
      m_paramStr = out.str();
    }
    return SymbolTable::Symbol(CStackSizedString(m_name.c_str()),
                               CStackSizedString(m_paramStr.c_str()));
  } else {
    return SymbolTable::Symbol(CStackSizedString(m_name.c_str()));
  }
}

bool FreeForm2::FeatureSpecExpression::FeatureName::operator==(
    const FeatureName &p_other) const {
  return GetName() == p_other.GetName() &&
         IsParameterized() == p_other.IsParameterized() &&
         GetParameters() == p_other.GetParameters();
}

bool FreeForm2::FeatureSpecExpression::FeatureName::operator<(
    const FeatureName &p_other) const {
  const int nameCompare = GetName().compare(p_other.GetName());
  if (nameCompare == 0) {
    if (IsParameterized() == p_other.IsParameterized()) {
      return GetParameters() < p_other.GetParameters();
    } else {
      return !IsParameterized();
    }
  } else {
    return nameCompare < 0;
  }
}

std::ostream &operator<<(
    std::ostream &p_out,
    const FreeForm2::FeatureSpecExpression::FeatureName &p_name) {
  return p_out << p_name.GetSymbol().ToString();
}

bool FreeForm2::FeatureSpecExpression::IgnoreParameterLess::operator()(
    const FeatureName &p_left, const FeatureName &p_right) const {
  return p_left.GetName() < p_right.GetName();
}

FreeForm2::FeatureSpecExpression::FeatureSpecExpression(
    const Annotations &p_annotations,
    const boost::shared_ptr<PublishFeatureMap> p_publishFeatureMap,
    const Expression &p_body, FeatureSpecType p_featureSpecType,
    bool p_returnsValue)
    : Expression(p_annotations),
      m_publishFeatureMap(p_publishFeatureMap),
      m_body(p_body),
      m_featureSpecType(p_featureSpecType),
      m_returnsValue(p_returnsValue) {
  FF2_ASSERT(p_publishFeatureMap != NULL && p_publishFeatureMap->size() > 0);

  // Check to make sure that the published types are valid and non-void.
  for (const auto &featureNameToType : *p_publishFeatureMap) {
    if (featureNameToType.second.Primitive() != Type::Unknown &&
        featureNameToType.second.IsValid() &&
        featureNameToType.second.Primitive() == Type::Void) {
      std::ostringstream err;
      err << "FeatureSpecExpression cannot have feature of type: '"
          << featureNameToType.second << "'";
      throw ParseError(err.str(), GetSourceLocation());
    }
  }
}

void FreeForm2::FeatureSpecExpression::Accept(Visitor &p_visitor) const {
  if (!p_visitor.AlternativeVisit(*this)) {
    m_body.Accept(p_visitor);

    p_visitor.Visit(*this);
  }
}

const FreeForm2::TypeImpl &FreeForm2::FeatureSpecExpression::GetType() const {
  if (m_returnsValue) {
    const TypeImpl &returnType = m_publishFeatureMap->begin()->second;

    if (!TypeUtil::IsAssignable(returnType, m_body.GetType())) {
      std::ostringstream err;
      err << "Expected feature to return type " << returnType
          << ", but found return type " << m_body.GetType();
      throw ParseError(err.str(), GetSourceLocation());
    }
    return returnType;
  } else {
    // If features are published, this FeatureSpec should be of type void.
    if (m_body.GetType().Primitive() != Type::Void) {
      std::ostringstream err;
      err << "Last statement of feature spec should be of type void";
      throw ParseError(err.str(), m_body.GetSourceLocation());
    }
    return m_body.GetType();
  }
}

size_t FreeForm2::FeatureSpecExpression::GetNumChildren() const { return 1; }

boost::shared_ptr<FreeForm2::FeatureSpecExpression::PublishFeatureMap>
FreeForm2::FeatureSpecExpression::GetPublishFeatureMap() const {
  return m_publishFeatureMap;
}

const FreeForm2::Expression &FreeForm2::FeatureSpecExpression::GetBody() const {
  return m_body;
}

bool FreeForm2::FeatureSpecExpression::IsDerived() const {
  return m_featureSpecType == DerivedFeature ||
         m_featureSpecType == AggregatedDerivedFeature;
}

FreeForm2::FeatureSpecExpression::FeatureSpecType
FreeForm2::FeatureSpecExpression::GetFeatureSpecType() const {
  return m_featureSpecType;
}

FreeForm2::ImportFeatureExpression::ImportFeatureExpression(
    const Annotations &p_annotations,
    const FreeForm2::FeatureSpecExpression::FeatureName &p_featureName,
    const std::vector<UInt32> &p_dimensions, VariableID p_id,
    TypeManager &p_typeManager)
    : Expression(p_annotations),
      m_featureName(p_featureName),
      m_type(p_typeManager.GetArrayType(
          TypeImpl::GetUInt32Instance(true), true,
          static_cast<unsigned int>(p_dimensions.size()), &p_dimensions[0],
          std::accumulate(p_dimensions.begin(), p_dimensions.end(), 1u,
                          std::multiplies<UInt32>()))),
      m_id(p_id) {}

FreeForm2::ImportFeatureExpression::ImportFeatureExpression(
    const Annotations &p_annotations,
    const FreeForm2::FeatureSpecExpression::FeatureName &p_featureName,
    VariableID p_id)
    : Expression(Annotations(p_annotations.m_sourceLocation,
                             ValueBounds(TypeImpl::GetUInt32Instance(true)))),
      m_featureName(p_featureName),
      m_type(TypeImpl::GetUInt32Instance(true)),
      m_id(p_id) {}

void FreeForm2::ImportFeatureExpression::Accept(Visitor &p_visitor) const {
  const size_t stackSize = p_visitor.StackSize();

  if (!p_visitor.AlternativeVisit(*this)) {
    p_visitor.Visit(*this);
  }

  FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}

const FreeForm2::TypeImpl &FreeForm2::ImportFeatureExpression::GetType() const {
  return m_type;
}

size_t FreeForm2::ImportFeatureExpression::GetNumChildren() const { return 0; }

const FreeForm2::FeatureSpecExpression::FeatureName &
FreeForm2::ImportFeatureExpression::GetFeatureName() const {
  return m_featureName;
}

FreeForm2::VariableID FreeForm2::ImportFeatureExpression::GetId() const {
  return m_id;
}

FreeForm2::FeatureGroupSpecExpression::FeatureGroupSpecExpression(
    const Annotations &p_annotations, const std::string &p_name,
    const std::vector<const FeatureSpecExpression *> &p_featureSpecs,
    bool p_isExtendedExperimental, bool p_isSmallExperimental,
    bool p_isBlockLevelFeature, bool p_isBodyBlockFeature,
    bool p_isForwardIndexFeature, const std::string &p_metaStreamName)
    : Expression(p_annotations),
      m_name(p_name),
      m_featureSpecs(p_featureSpecs),
      m_isExtendedExperimental(p_isExtendedExperimental),
      m_isSmallExperimental(p_isSmallExperimental),
      m_isBlockLevelFeature(p_isBlockLevelFeature),
      m_isBodyBlockFeature(p_isBodyBlockFeature),
      m_isForwardIndexFeature(p_isForwardIndexFeature),
      m_metaStreamName(p_metaStreamName) {
  FF2_ASSERT(p_featureSpecs.size() > 0);

  m_featureSpecType = p_featureSpecs[0]->GetFeatureSpecType();

  BOOST_FOREACH (const FeatureSpecExpression *featureSpec, p_featureSpecs) {
    if (m_featureSpecType != featureSpec->GetFeatureSpecType()) {
      std::ostringstream err;
      err << "All feature specifications within feature group '" << p_name
          << "' must be of the same type.";
      throw ParseError(err.str(), GetSourceLocation());
    }
  }
}

void FreeForm2::FeatureGroupSpecExpression::Accept(Visitor &p_visitor) const {
  if (!p_visitor.AlternativeVisit(*this)) {
    for (std::vector<const FeatureSpecExpression *>::const_iterator
             specExpression = m_featureSpecs.begin();
         specExpression != m_featureSpecs.end(); ++specExpression) {
      (*specExpression)->Accept(p_visitor);
    }

    p_visitor.Visit(*this);
  }
}

const FreeForm2::TypeImpl &FreeForm2::FeatureGroupSpecExpression::GetType()
    const {
  return TypeImpl::GetVoidInstance();
}

size_t FreeForm2::FeatureGroupSpecExpression::GetNumChildren() const {
  return m_featureSpecs.size();
}

FreeForm2::FeatureSpecExpression::FeatureSpecType
FreeForm2::FeatureGroupSpecExpression::GetFeatureSpecType() const {
  return m_featureSpecType;
}

const std::string &FreeForm2::FeatureGroupSpecExpression::GetName() const {
  return m_name;
}

const std::vector<const FreeForm2::FeatureSpecExpression *>
    &FreeForm2::FeatureGroupSpecExpression::GetFeatureSpecs() const {
  return m_featureSpecs;
}

bool FreeForm2::FeatureGroupSpecExpression::IsExtendedExperimental() const {
  return m_isExtendedExperimental;
}

bool FreeForm2::FeatureGroupSpecExpression::IsSmallExperimental() const {
  return m_isSmallExperimental;
}

bool FreeForm2::FeatureGroupSpecExpression::IsBlockLevelFeature() const {
  return m_isBlockLevelFeature;
}

bool FreeForm2::FeatureGroupSpecExpression::IsBodyBlockFeature() const {
  return m_isBodyBlockFeature;
}

bool FreeForm2::FeatureGroupSpecExpression::IsForwardIndexFeature() const {
  return m_isForwardIndexFeature;
}

const std::string &FreeForm2::FeatureGroupSpecExpression::GetMetaStreamName()
    const {
  return m_metaStreamName;
}

bool FreeForm2::FeatureGroupSpecExpression::IsPerStream() const {
  return !(m_featureSpecType ==
               FeatureSpecExpression::AggregatedDerivedFeature ||
           m_featureSpecType == FeatureSpecExpression::AbInitioFeature ||
           !m_metaStreamName.empty() || m_isBodyBlockFeature);
}

FreeForm2::AggregateContextExpression::AggregateContextExpression(
    const Annotations &p_annotations, const Expression &p_body)
    : Expression(p_annotations), m_body(p_body) {}

void FreeForm2::AggregateContextExpression::Accept(Visitor &p_visitor) const {
  const size_t stackSize = p_visitor.StackSize();

  if (!p_visitor.AlternativeVisit(*this)) {
    m_body.Accept(p_visitor);

    p_visitor.Visit(*this);
  }

  FF2_ASSERT(p_visitor.StackSize() == stackSize + p_visitor.StackIncrement());
}

const FreeForm2::TypeImpl &FreeForm2::AggregateContextExpression::GetType()
    const {
  return TypeImpl::GetVoidInstance();
}

size_t FreeForm2::AggregateContextExpression::GetNumChildren() const {
  return 1;
}

const FreeForm2::Expression &FreeForm2::AggregateContextExpression::GetBody()
    const {
  return m_body;
}
