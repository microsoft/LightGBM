/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include "Extern.h"

#include <boost/foreach.hpp>
#include <sstream>

#include "ArrayType.h"
#include "FreeForm2Assert.h"
#include "FreeForm2ExternalData.h"
#include "FreeForm2Result.h"
#include "ObjectType.h"
#include "TypeImpl.h"
#include "TypeManager.h"
#include "Visitor.h"

namespace {
// This class acts as an ExternalData implementor for external objects.
// This implementation relies on the backend to know how to resolve the
// objects, and not to call the user-defined resolver for objects.
class ExternalObject : public FreeForm2::ExternalData {
 public:
  ExternalObject(const char *p_name)
      : ExternalData(
            p_name, *FreeForm2::TypeManager::GetGlobalTypeManager().GetTypeInfo(
                        p_name)) {}

  virtual ~ExternalObject() {}
};
}  // namespace

// This struct holds information for resolving built-in objects.
struct FreeForm2::ExternExpression::BuiltInObject {
  BuiltInObject(const char *p_name) : m_object(p_name){};

  ExternalObject m_object;
};

namespace {
typedef FreeForm2::ExternExpression::BuiltInObject BuiltInObject;
static const BuiltInObject c_numberOfTuplesCommonImpl("NumberOfTuplesCommon");
static const BuiltInObject c_numberOfTuplesCommonNoDuplicateImpl(
    "NumberOfTuplesCommonNoDuplicate");
static const BuiltInObject c_numberOfTuplesInTriplesCommonImpl(
    "NumberOfTuplesInTriplesCommon");
static const BuiltInObject c_alterationAndTermWeightImpl(
    "AlterationAndTermWeightingCalculator");
static const BuiltInObject c_alterationWeightImpl(
    "AlterationWeightingCalculator");
static const BuiltInObject c_trueNearDoubleQueueImpl("TrueNearDoubleQueue");
static const BuiltInObject c_boundedQueueImpl("BoundedQueue");
}  // namespace

const FreeForm2::ExternExpression::BuiltInObject
    &FreeForm2::ExternExpression::c_numberOfTuplesCommonObject =
        c_numberOfTuplesCommonImpl;
const FreeForm2::ExternExpression::BuiltInObject
    &FreeForm2::ExternExpression::c_numberOfTuplesCommonNoDuplicateObject =
        c_numberOfTuplesCommonNoDuplicateImpl;
const FreeForm2::ExternExpression::BuiltInObject
    &FreeForm2::ExternExpression::c_numberOfTuplesInTriplesCommonObject =
        c_numberOfTuplesInTriplesCommonImpl;
const FreeForm2::ExternExpression::BuiltInObject
    &FreeForm2::ExternExpression::c_alterationAndTermWeightObject =
        c_alterationAndTermWeightImpl;
const FreeForm2::ExternExpression::BuiltInObject
    &FreeForm2::ExternExpression::c_alterationWeightObject =
        c_alterationWeightImpl;
const FreeForm2::ExternExpression::BuiltInObject
    &FreeForm2::ExternExpression::c_trueNearDoubleQueueObject =
        c_trueNearDoubleQueueImpl;
const FreeForm2::ExternExpression::BuiltInObject
    &FreeForm2::ExternExpression::c_boundedQueueObject = c_boundedQueueImpl;

const FreeForm2::ExternExpression::BuiltInObject *
FreeForm2::ExternExpression::GetObjectByName(const std::string &p_name) {
  static const FreeForm2::ExternExpression::BuiltInObject *const c_objects[] = {
      &c_numberOfTuplesCommonObject,
      &c_numberOfTuplesCommonNoDuplicateObject,
      &c_numberOfTuplesInTriplesCommonObject,
      &c_alterationAndTermWeightObject,
      &c_alterationWeightObject,
      &c_trueNearDoubleQueueObject,
      &c_boundedQueueObject};

  BOOST_FOREACH (const FreeForm2::ExternExpression::BuiltInObject *const obj,
                 c_objects) {
    if (p_name == obj->m_object.GetName()) {
      return obj;
    }
  }
  return nullptr;
}

const FreeForm2::ExternalData &FreeForm2::ExternExpression::GetObjectData(
    const BuiltInObject &p_object) {
  return p_object.m_object;
}

FreeForm2::ExternExpression::ExternExpression(const Annotations &p_annotations,
                                              const ExternalData &p_data,
                                              const TypeImpl &p_declaredType,
                                              VariableID p_id,
                                              TypeManager &p_typeManager)
    : Expression(Annotations(
          p_annotations.m_sourceLocation,
          p_data.IsCompileTimeConstant()
              ? ValueBounds(p_data.GetType(), p_data.GetCompileTimeValue())
              : ValueBounds(p_data.GetType()))),
      m_data(p_data),
      m_id(p_id) {
  if (m_data.GetType() != p_declaredType) {
    std::ostringstream err;
    err << "Incorrect type for external data member " << m_data.GetName()
        << ". Expected type " << m_data.GetType() << "; found type "
        << p_declaredType;
    throw ParseError(err.str(), GetSourceLocation());
  }
}

FreeForm2::ExternExpression::ExternExpression(const Annotations &p_annotations,
                                              const BuiltInObject &p_object,
                                              VariableID p_id,
                                              TypeManager &p_typeManager)
    : Expression(
          Annotations(p_annotations.m_sourceLocation,
                      p_object.m_object.IsCompileTimeConstant()
                          ? ValueBounds(p_object.m_object.GetType(),
                                        p_object.m_object.GetCompileTimeValue())
                          : ValueBounds(p_object.m_object.GetType()))),
      m_data(p_object.m_object),
      m_id(p_id) {}

FreeForm2::ExternExpression::ExternExpression(const Annotations &p_annotations,
                                              const ExternalData &p_data,
                                              const TypeImpl &p_type)
    : Expression(Annotations(
          p_annotations.m_sourceLocation,
          p_data.IsCompileTimeConstant()
              ? ValueBounds(p_data.GetType(), p_data.GetCompileTimeValue())
              : ValueBounds(p_data.GetType()))),
      m_data(p_data),
      m_id(VariableID::c_invalidID) {
  if (m_data.GetType() != p_type) {
    std::ostringstream err;
    err << "Incorrect type for external data member " << m_data.GetName()
        << ". Expected type " << m_data.GetType() << "; found type " << p_type;
    throw ParseError(err.str(), GetSourceLocation());
  }
}

void FreeForm2::ExternExpression::Accept(Visitor &p_visitor) const {
  if (!p_visitor.AlternativeVisit(*this)) {
    p_visitor.Visit(*this);
  }
}

size_t FreeForm2::ExternExpression::GetNumChildren() const { return 0; }

const FreeForm2::TypeImpl &FreeForm2::ExternExpression::GetType() const {
  return m_data.GetType();
}

bool FreeForm2::ExternExpression::IsConstant() const {
  return m_data.IsCompileTimeConstant();
}

FreeForm2::ConstantValue FreeForm2::ExternExpression::GetConstantValue() const {
  return m_data.GetCompileTimeValue();
}

const FreeForm2::ExternalData &FreeForm2::ExternExpression::GetData() const {
  return m_data;
}

FreeForm2::VariableID FreeForm2::ExternExpression::GetId() const {
  return m_id;
}
