/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include "NeuralInput.h"

#include <LightGBM/utils/log.h>
#include <math.h>
#include <stdlib.h>

#include <boost/algorithm/string.hpp>

#include "MigratedApi.h"
#include "UnionBondInput_types.h"

using namespace LightGBM;
using namespace DynamicRank;

const static std::vector<std::string> c_empty;

// Use caching for the results of neural inputs. Remove this line to not use
// cached results.
#define INPUT_CACHE

NeuralInput::NeuralInput() {}

NeuralInput::NeuralInput(const NeuralInputBondData &p_data) {
  if (p_data.m_segments.hasvalue()) {
    m_segments.reset(new std::vector<std::string>(p_data.m_segments.value()));
  }
}

NeuralInput::~NeuralInput() {}

void NeuralInput::FillBondData(NeuralInputBondData &p_data) const {
  if (m_segments != nullptr) {
    p_data.m_segments.set(*m_segments);
  }
}

double NeuralInput::EvaluateInput(UInt32 /*input*/) const {
  // This virtual method should never be called
  // There should be a version in the derived class
  return 0.0;
}

double NeuralInput::Evaluate(UInt32 **p_featureVectorArray,
                             UInt32 p_currentDocument,
                             UInt32 p_documentCount) const {
  // If derived class needs list based evaluation, it should have a version.
  //     Else call freeform evaluate function.
  if (p_currentDocument >= p_documentCount) {
    return 0.0;
  }
  return Evaluate(p_featureVectorArray[p_currentDocument]);
}

UInt32 NeuralInput::GetAssociatedFeature() const {
  return static_cast<UInt32>(-1);
}

double NeuralInput::GetSlope() const { return 1.0; }

double NeuralInput::GetIntercept() const { return 0; }

void NeuralInput::LoadSegments(DynamicRank::Config &p_config,
                               const char *szSection) {
  char szSegments[1024];
  if (!p_config.GetStringParameter(szSection, "Segments", szSegments,
                                   sizeof(szSegments))) {
    // Sectors not specified; will leave m_segments as a length zero vector
    return;
  }

  std::string tmp(szSegments);
  std::vector<std::string> segmentNames;
  boost::split(segmentNames, tmp, boost::is_any_of(", "));
  for (std::vector<std::string>::iterator it = segmentNames.begin();
       it != segmentNames.end(); ++it) {
    if (!it->empty()) {
      boost::algorithm::to_lower(*it);
      if (!m_segments) {
        m_segments.reset(new std::vector<std::string>());
      }
      m_segments->push_back(*it);
    }
  }
}

void NeuralInput::SetSegments(const std::vector<std::string> &p_segments) {
  if (p_segments.empty()) {
    m_segments.reset();
  } else {
    m_segments.reset(new std::vector<std::string>(p_segments));
  }
}

bool NeuralInput::Save(FILE *fpOutput, size_t nInputId,
                       const IFeatureMap & /*p_featureMap*/) const {
  // Format for an input in the config file is
  // [Input:index]
  // Name=...
  // Transform={linear, bucket, loglinear, etc.}
  // (transform-specific config)

  // All we can do here is write out the section name and the
  // feature name
  fprintf(fpOutput, "\n[Input:%Iu]\n", nInputId);

  // Print out the segments, if applicable
  if (m_segments && m_segments->size() != 0) {
    fprintf(fpOutput, "Segments=");
    char separatingComma[2] = {0, 0};

    for (std::vector<std::string>::const_iterator iter = m_segments->begin();
         iter != m_segments->end(); ++iter) {
      fprintf(fpOutput, "%s%s", separatingComma, (*iter).c_str());
      separatingComma[0] = ',';
    }
    fprintf(fpOutput, "\n");
  }

  return true;
}

bool NeuralInput::EqualInternal(const NeuralInput *p_input) const {
  if (!p_input) {
    return false;
  }

  if ((m_segments.get() == nullptr) != (p_input->m_segments.get() == nullptr)) {
    return false;
  }

  if (m_segments.get() != nullptr && p_input->m_segments.get() != nullptr &&
      *m_segments != *p_input->m_segments) {
    return false;
  }
  return true;
}

bool NeuralInput::Train(double /*dblLearningRate*/, double /*outputHigh*/,
                        double /*outputLow*/, double /*dblOutputDelta*/,
                        UInt32 /*inputHigh*/[], UInt32 /*inputLow*/[]) {
  // Default implementation does nothing.
  return false;
}

// Do a 'set COMPUTE_BOOST_STATIC_ASSERT=1' command, then build in order to get
// error output that contains the struct size.
#if defined(COMPUTE_BOOST_STATIC_ASSERT)
template <int s>
struct Sizeof_NeuralInput;
Sizeof_NeuralInput<sizeof(NeuralInput)> sizeof_NeuralInput;
#endif

// Use static assert to force the user to consider the external size and
// serialization if a new member is added. For each new member please add it to
// the template serialize and generate all the bin files for all the nets in
// searchgold. If the new member has an external size, then the size has to be
// reported within this function.
#ifdef DEBUG
BOOST_STATIC_ASSERT(sizeof(NeuralInput) == 16);
#else
BOOST_STATIC_ASSERT(sizeof(NeuralInput) == 16);
#endif

size_t NeuralInput::GetExternalSize() const {
  size_t s = 0;
  if (m_segments) {
    const std::vector<std::string> &v = *m_segments;
    s += sizeof(std::string) * v.capacity();
    for (size_t i = 0; i < v.size(); ++i) {
      s += v[i].capacity();
    }
  }
  return s;
}

const std::vector<std::string> &NeuralInput::GetSegments() const {
  if (m_segments)
    return *m_segments;
  else
    return c_empty;
}

BulkNeuralInput::BulkNeuralInput() {}

BulkNeuralInput::~BulkNeuralInput() {}

void BulkNeuralInput::Evaluate(UInt32 **p_featureVectorArray,
                               UInt32 p_currentDocument, UInt32 p_documentCount,
                               float p_output[]) const {
  if (p_currentDocument >= p_documentCount) {
    return;
  }
  Evaluate(p_featureVectorArray[p_currentDocument], p_output);
}

bool NeuralInputUnary::ReadAssociatedFeature(DynamicRank::Config &p_config,
                                             const char *szSection,
                                             IFeatureMap &p_featureMap,
                                             UInt32 *piFeature) {
  *piFeature = static_cast<UInt32>(-1);

  if (szSection == NULL) {
    return false;
  }

  char szName[256];
  if (p_config.GetStringParameter(szSection, "Name", szName, sizeof(szName))) {
    if (!p_featureMap.ObtainFeatureIndex(szName, *piFeature)) {
      Log::Warning(
          "DR:ReadAssociatedFeature: Could not find index for feature name: %s "
          "in section: %s",
          szName, szSection);
      return false;  // input error, invalid feature name
    }
    return true;
  }
  Log::Warning(
      "DR:ReadAssociatedFeature: Could not find 'Name' of the feature for "
      "section: %s",
      szSection);
  return false;  // feature is not specified
}

UInt32 NeuralInputUnary::GetAssociatedFeature() const { return m_iFeature; }

void NeuralInputUnary::GetAllAssociatedFeatures(
    std::vector<UInt32> &associatedFeaturesList) const {
  associatedFeaturesList.push_back(m_iFeature);
}

NeuralInputUnary::NeuralInputUnary() : NeuralInput(), m_iFeature(0) {}

NeuralInputUnary::NeuralInputUnary(int p_feature)
    : NeuralInput(), m_iFeature(p_feature) {}

NeuralInputUnary::NeuralInputUnary(const NeuralInputUnaryBondData &p_data)
    : NeuralInput(p_data), m_iFeature(p_data.m_iFeature) {}

void NeuralInputUnary::FillBondData(NeuralInputUnaryBondData &p_data) const {
  // Fill base class.
  NeuralInput::FillBondData(p_data);
  p_data.m_iFeature = m_iFeature;
}

bool NeuralInputUnary::Save(FILE *fpOutput, size_t nInputId,
                            const IFeatureMap &p_featureMap) const {
  NeuralInput::Save(fpOutput, nInputId, p_featureMap);

  // Create a stack buffer for the feature name
  char rgInput[1024];
  if (!p_featureMap.GetFeatureName(m_iFeature, rgInput, 1024)) {
    return false;
  }

  fprintf(fpOutput, "Name=%s\n", rgInput);

  return true;
}

void NeuralInputUnary::CopyFrom(const NeuralInputUnary &p_neuralInputUnary) {
  m_iFeature = p_neuralInputUnary.m_iFeature;
}

bool NeuralInputUnary::Equal(const NeuralInput *p_input) const {
  if (!NeuralInput::EqualInternal(p_input)) {
    return false;
  }

  const NeuralInputUnary *other =
      dynamic_cast<const NeuralInputUnary *>(p_input);
  if (!other) {
    return false;
  }
  if (m_iFeature != other->m_iFeature) {
    return false;
  }
  return true;
}

double NeuralInputUnary::Evaluate(UInt32 input[]) const {
  return EvaluateInput(input[m_iFeature]);
}

UInt32 NeuralInputUnary::GetFeature() const { return m_iFeature; }

size_t NeuralInputUnary::GetSize() const {
  return sizeof(*this) + GetExternalSize();
}

// Do a 'set COMPUTE_BOOST_STATIC_ASSERT=1' command, then build in order to get
// error output that contains the struct size.
#if defined(COMPUTE_BOOST_STATIC_ASSERT)
template <int s>
struct Sizeof_NeuralInputUnary;
Sizeof_NeuralInputUnary<sizeof(NeuralInputUnary) - sizeof(NeuralInput)>
    sizeof_NeuralInputUnary;
#endif

// Use static assert to force the user to consider the external size and
// serialization if a new member is added. For each new member please add it to
// the template serialize and generate all the bin files for all the nets in
// searchgold. If the new member has an external size, then the size has to be
// reported within this function.
#ifdef DEBUG
BOOST_STATIC_ASSERT(sizeof(NeuralInputUnary) - sizeof(NeuralInput) == 8);
#else
BOOST_STATIC_ASSERT(sizeof(NeuralInputUnary) - sizeof(NeuralInput) == 8);
#endif

size_t NeuralInputUnary::GetExternalSize() const {
  return NeuralInput::GetExternalSize();
}

NeuralInputLinear *NeuralInputLinear::Load(DynamicRank::Config &p_config,
                                           const char *szSection,
                                           IFeatureMap &p_featureMap) {
  UInt32 iFeature;

  if (!NeuralInputUnary::ReadAssociatedFeature(p_config, szSection,
                                               p_featureMap, &iFeature)) {
    return NULL;
  }

  double slope = 0.0;
  double intercept = 0.0;
  if ((!p_config.GetDoubleParameter(szSection, "Slope", &slope)) ||
      (!p_config.GetDoubleParameter(szSection, "Intercept", &intercept))) {
    Log::Warning(
        "NeuralInputLinear::Load: Slope or Intercept not provided (double "
        "value) for section: %s",
        szSection);
    return NULL;
  }

  NeuralInputLinear *pNeuralInputLinear =
      new NeuralInputLinear(iFeature, slope, intercept);
  return pNeuralInputLinear;
}

bool NeuralInputLinear::Equal(const NeuralInput *p_input) const {
  if (!NeuralInputUnary::Equal(p_input)) {
    return false;
  }

  const NeuralInputLinear *other =
      dynamic_cast<const NeuralInputLinear *>(p_input);
  if (!other) {
    return false;
  }
  if (m_slope != other->m_slope || m_intercept != other->m_intercept ||
      m_iFeature != other->m_iFeature) {
    return false;
  }

  return true;
}

NeuralInputLinear::NeuralInputLinear()
    : NeuralInputUnary(0), m_slope(0.0), m_intercept(0.0) {}

NeuralInputLinear::NeuralInputLinear(int id, double slope, double intercept)
    : NeuralInputUnary(id), m_slope(slope), m_intercept(intercept) {}

NeuralInputLinear::NeuralInputLinear(const NeuralInputLinearBondData &p_data)
    : NeuralInputUnary(p_data),
      m_slope(p_data.m_slope),
      m_intercept(p_data.m_intercept) {}

void NeuralInputLinear::FillBond(UnionBondInput &p_data) const {
  p_data.m_inputType = "linear";
  // Fill bond struct of the class.
  NeuralInputLinearBondData data;
  FillBondData(data);
  p_data.m_linear.set(data);
}

void NeuralInputLinear::FillBondData(NeuralInputLinearBondData &p_data) const {
  // Fill base class.
  NeuralInputUnary::FillBondData(p_data);
  p_data.m_slope = m_slope;
  p_data.m_intercept = m_intercept;
}

double NeuralInputLinear::GetMax() const {
  // The output of this node is 0 mean, 1 standard deviation. For the max
  // value of the output, we assume that it is 1 standard deviation away from
  // the mean.
  return 1;
}

double NeuralInputLinear::GetMin() const {
  // The output of this node is 0 mean, 1 standard deviation. For the min
  // value of the output, we assume that it is 1 standard deviation away from
  // the mean. Further, since the min input to the node is 0, we cannot have
  // a value that is less than m_intercept.
  return ((m_intercept > -1) ? m_intercept : -1);
}

double NeuralInputLinear::GetSlope() const { return m_slope; }

double NeuralInputLinear::GetIntercept() const { return m_intercept; }

double NeuralInputLinear::EvaluateInput(UInt32 val) const {
  return ((double)val * m_slope) + m_intercept;
}

bool NeuralInputLinear::Save(FILE *fpOutput, size_t nInputId,
                             const IFeatureMap &p_featureMap) const {
  // The base class can write out the generic portion
  NeuralInputUnary::Save(fpOutput, nInputId, p_featureMap);
  fprintf(fpOutput, "Transform=linear\n");
  fprintf(fpOutput, "Slope=%lg\n", m_slope);
  fprintf(fpOutput, "Intercept=%lg\n", m_intercept);

  return true;
}

size_t NeuralInputLinear::GetSize() const {
  return sizeof(*this) + GetExternalSize();
}

// Do a 'set COMPUTE_BOOST_STATIC_ASSERT=1' command, then build in order to get
// error output that contains the struct size.
#if defined(COMPUTE_BOOST_STATIC_ASSERT)
template <int s>
struct Sizeof_NeuralInputLinear;
Sizeof_NeuralInputLinear<sizeof(NeuralInputLinear) - sizeof(NeuralInputUnary)>
    sizeof_NeuralInputLinear;
#endif

// Use static assert to force the user to consider the external size and
// serialization if a new member is added. For each new member please add it to
// the template serialize and generate all the bin files for all the nets in
// searchgold. If the new member has an external size, then the size has to be
// reported within this function.
#ifdef DEBUG
BOOST_STATIC_ASSERT(sizeof(NeuralInputLinear) - sizeof(NeuralInputUnary) == 16);
#else
BOOST_STATIC_ASSERT(sizeof(NeuralInputLinear) - sizeof(NeuralInputUnary) == 16);
#endif

size_t NeuralInputLinear::GetExternalSize() const {
  return NeuralInputUnary::GetExternalSize();
}

NeuralInputLogLinear *NeuralInputLogLinear::Load(DynamicRank::Config &p_config,
                                                 const char *szSection,
                                                 IFeatureMap &p_featureMap) {
  UInt32 iFeature;

  if (!NeuralInputUnary::ReadAssociatedFeature(p_config, szSection,
                                               p_featureMap, &iFeature)) {
    return NULL;
  }

  double slope = 0.0;
  double intercept = 0.0;
  if ((!p_config.GetDoubleParameter(szSection, "Slope", &slope)) ||
      (!p_config.GetDoubleParameter(szSection, "Intercept", &intercept))) {
    return NULL;
  }

  NeuralInputLogLinear *pNeuralInputLogLinear =
      new NeuralInputLogLinear(iFeature, slope, intercept);
  return pNeuralInputLogLinear;
}

NeuralInputLogLinear::NeuralInputLogLinear(int id, double slope,
                                           double intercept)
    : NeuralInputLinear(id, slope, intercept) {}

NeuralInputLogLinear::NeuralInputLogLinear(
    const NeuralInputLogLinearBondData &p_data)
    : NeuralInputLinear(p_data) {}

void NeuralInputLogLinear::FillBond(UnionBondInput &p_data) const {
  p_data.m_inputType = "loglinear";
  // Fill bond struct of the class.
  NeuralInputLogLinearBondData data;
  FillBondData(data);
  p_data.m_loglinear.set(data);
}

void NeuralInputLogLinear::FillBondData(
    NeuralInputLogLinearBondData &p_data) const {
  // Fill base class.
  NeuralInputLinear::FillBondData(p_data);
}

bool NeuralInputLogLinear::Equal(const NeuralInput *p_input) const {
  if (!NeuralInputLinear::Equal(p_input)) {
    return false;
  }

  if (!dynamic_cast<const NeuralInputLogLinear *>(p_input)) {
    return false;
  }

  return true;
}

double NeuralInputLogLinear::EvaluateInput(UInt32 val) const {
  return (log((double)(val) + 1) * m_slope) + m_intercept;
}

bool NeuralInputLogLinear::Save(FILE *fpOutput, size_t nInputId,
                                const IFeatureMap &p_featureMap) const {
  // The base class can write out the generic portion
  NeuralInputUnary::Save(fpOutput, nInputId, p_featureMap);
  fprintf(fpOutput, "Transform=loglinear\n");
  fprintf(fpOutput, "Slope=%lg\n", m_slope);
  fprintf(fpOutput, "Intercept=%lg\n", m_intercept);

  return true;
}

size_t NeuralInputLogLinear::GetSize() const {
  return sizeof(*this) + GetExternalSize();
}

// Do a 'set COMPUTE_BOOST_STATIC_ASSERT=1' command, then build in order to get
// error output that contains the struct size.
#if defined(COMPUTE_BOOST_STATIC_ASSERT)
template <int s>
struct Sizeof_NeuralInputLogLinear;
Sizeof_NeuralInputLogLinear<sizeof(NeuralInputLogLinear) -
                            sizeof(NeuralInputLinear)>
    sizeof_NeuralInputLogLinear;
#endif

// Use static assert to force the user to consider the external size and
// serialization if a new member is added. For each new member please add it to
// the template serialize and generate all the bin files for all the nets in
// searchgold. If the new member has an external size, then the size has to be
// reported within this function.
#ifdef DEBUG
BOOST_STATIC_ASSERT(sizeof(NeuralInputLogLinear) - sizeof(NeuralInputLinear) ==
                    0);
#else
BOOST_STATIC_ASSERT(sizeof(NeuralInputLogLinear) - sizeof(NeuralInputLinear) ==
                    0);
#endif

size_t NeuralInputLogLinear::GetExternalSize() const {
  return NeuralInputLinear::GetExternalSize();
}

NeuralInputBucket::NeuralInputBucket(int p_id, double p_min,
                                     bool p_mininclusive, double p_max,
                                     bool p_maxinclusive)
    : NeuralInputUnary(p_id) {
  m_fMinInclusive = p_mininclusive;
  m_fMaxInclusive = p_maxinclusive;
  if (p_min < 0) {
    m_nMinValue = 0;
  } else {
    m_nMinValue = static_cast<UInt32>(p_min);
    if (!p_mininclusive) {
      m_nMinValue++;
    }
  }

  m_nMaxValue = static_cast<UInt32>(p_max);
  if (p_maxinclusive) {
    m_nMaxValue++;
  }
}

NeuralInputBucket::NeuralInputBucket(const NeuralInputBucketBondData &p_data)
    : NeuralInputUnary(p_data),
      m_fMinInclusive(p_data.m_fMinInclusive),
      m_fMaxInclusive(p_data.m_fMaxInclusive),
      m_nMinValue(p_data.m_nMinValue),
      m_nMaxValue(p_data.m_nMaxValue) {}

void NeuralInputBucket::FillBond(UnionBondInput &p_data) const {
  p_data.m_inputType = "bucket";
  // Fill bond struct of the class.
  NeuralInputBucketBondData data;
  FillBondData(data);
  p_data.m_bucket.set(data);
}

void NeuralInputBucket::FillBondData(NeuralInputBucketBondData &p_data) const {
  // Fill base class.
  NeuralInputUnary::FillBondData(p_data);
  p_data.m_fMinInclusive = m_fMinInclusive;
  p_data.m_fMaxInclusive = m_fMaxInclusive;
  p_data.m_nMinValue = m_nMinValue;
  p_data.m_nMaxValue = m_nMaxValue;
}

bool NeuralInputBucket::Equal(const NeuralInput *p_input) const {
  if (!NeuralInputUnary::Equal(p_input)) {
    return false;
  }

  const NeuralInputBucket *other =
      dynamic_cast<const NeuralInputBucket *>(p_input);
  if (!other) {
    return false;
  }
  if (m_fMinInclusive != other->m_fMinInclusive ||
      m_fMaxInclusive != other->m_fMaxInclusive ||
      m_nMinValue != other->m_nMinValue || m_nMaxValue != other->m_nMaxValue ||
      m_iFeature != other->m_iFeature) {
    return false;
  }

  return true;
}

bool NeuralInputBucket::GetMinInclusive() const { return m_fMinInclusive; }

bool NeuralInputBucket::GetMaxInclusive() const { return m_fMaxInclusive; }

UInt32 NeuralInputBucket::GetMinValue() const { return m_nMinValue; }

UInt32 NeuralInputBucket::GetMaxValue() const { return m_nMaxValue; }

double NeuralInputBucket::GetMax() const {
  // The output of this node is 0 or 1. So the max value is 1.
  return 1;
}

double NeuralInputBucket::GetMin() const {
  // The output of this node is 0 or 1. So the max value is 0.
  return 0;
}

double NeuralInputBucket::EvaluateInput(UInt32 val) const {
  if (m_nMinValue <= val && val < m_nMaxValue) {
    return 1.0;
  }

  return 0.0;
}

bool NeuralInputBucket::Save(FILE *fpOutput, size_t nInputId,
                             const IFeatureMap &p_featureMap) const {
  // The base class can write out the generic portion
  NeuralInputUnary::Save(fpOutput, nInputId, p_featureMap);
  fprintf(fpOutput, "Transform=bucket\n");
  double dblMinValue = (double)m_nMinValue;
  double dblMaxValue = (double)m_nMaxValue;

  if (!m_fMinInclusive) {
    // Min by default is inclusive, so if we have to write out a
    // non-exclusive form we have to subtract.
    dblMinValue -= 1.0;
  }

  if (m_fMaxInclusive) {
    // Max by default is exclusive, so we have to write out the
    // exclusive variant
    dblMaxValue -= 1.0;
  }

  fprintf(fpOutput, "MinValue=%lf\n", dblMinValue);
  fprintf(fpOutput, "MaxValue=%lf\n", dblMaxValue);
  fprintf(fpOutput, "MinInclusive=%s\n", m_fMinInclusive ? "true" : "false");
  fprintf(fpOutput, "MaxInclusive=%s\n", m_fMaxInclusive ? "true" : "false");

  return true;
}

NeuralInputBucket *NeuralInputBucket::Load(DynamicRank::Config &p_config,
                                           const char *szSection,
                                           IFeatureMap &p_featureMap) {
  UInt32 iFeature;

  if (!NeuralInputUnary::ReadAssociatedFeature(p_config, szSection,
                                               p_featureMap, &iFeature)) {
    return NULL;
  }

  double min = 0.0;
  double max = 0.0;
  bool minincl = false;
  bool maxincl = false;
  if ((!p_config.GetDoubleParameter(szSection, "MinValue", &min)) ||
      (!p_config.GetDoubleParameter(szSection, "MaxValue", &max)) ||
      (!p_config.GetBoolParameter(szSection, "MinInclusive", &minincl)) ||
      (!p_config.GetBoolParameter(szSection, "MaxInclusive", &maxincl))) {
    return NULL;
  }

  NeuralInputBucket *pNeuralInputBucket =
      new NeuralInputBucket(iFeature, min, minincl, max, maxincl);
  if (!pNeuralInputBucket) {
    return NULL;
  }

  return pNeuralInputBucket;
}

size_t NeuralInputBucket::GetSize() const {
  return sizeof(*this) + GetExternalSize();
}

// Do a 'set COMPUTE_BOOST_STATIC_ASSERT=1' command, then build in order to get
// error output that contains the struct size.
#if defined(COMPUTE_BOOST_STATIC_ASSERT)
template <int s>
struct Sizeof_NeuralInputBucket;
Sizeof_NeuralInputBucket<sizeof(NeuralInputBucket) - sizeof(NeuralInputUnary)>
    sizeof_NeuralInputBucket;
#endif

size_t NeuralInputBucket::GetExternalSize() const {
  return NeuralInputUnary::GetExternalSize();
}

NeuralInputRational *NeuralInputRational::Load(DynamicRank::Config &p_config,
                                               const char *szSection,
                                               IFeatureMap &p_featureMap) {
  UInt32 iFeature;

  if (!NeuralInputUnary::ReadAssociatedFeature(p_config, szSection,
                                               p_featureMap, &iFeature)) {
    return NULL;
  }

  double dblDampingFactor =
      p_config.GetDoubleParameter(szSection, "DampingFactor", 0.0);
  if (dblDampingFactor <= 0.0) {
    return NULL;
  }

  NeuralInputRational *pNeuralInputRational =
      new NeuralInputRational(iFeature, dblDampingFactor);
  return pNeuralInputRational;
}

NeuralInputRational::NeuralInputRational(int p_id, double p_dblDampingFactor)
    : NeuralInputUnary(p_id),
      // Take the absolute value in order to make the field
      // nicely defined (no poles).
      m_dblDampingFactor(fabs(p_dblDampingFactor)) {}

NeuralInputRational::NeuralInputRational(
    const NeuralInputRationalBondData &p_data)
    : NeuralInputUnary(p_data), m_dblDampingFactor(p_data.m_dblDampingFactor) {}

void NeuralInputRational::FillBond(UnionBondInput &p_data) const {
  p_data.m_inputType = "rational";
  // Fill bond struct of the class.
  NeuralInputRationalBondData data;
  FillBondData(data);
  p_data.m_rational.set(data);
}

void NeuralInputRational::FillBondData(
    NeuralInputRationalBondData &p_data) const {
  // Fill base class.
  NeuralInputUnary::FillBondData(p_data);
  p_data.m_dblDampingFactor = m_dblDampingFactor;
}

bool NeuralInputRational::Equal(const NeuralInput *p_input) const {
  if (!NeuralInputUnary::Equal(p_input)) {
    return false;
  }

  const NeuralInputRational *other =
      dynamic_cast<const NeuralInputRational *>(p_input);
  if (!other) {
    return false;
  }
  if (m_dblDampingFactor != other->m_dblDampingFactor) {
    return false;
  }

  return true;
}

double NeuralInputRational::EvaluateInput(UInt32 input) const {
  double dblOutput = (double)input / ((double)input + m_dblDampingFactor);

  return dblOutput;
}

double NeuralInputRational::GetDampingFactor() const {
  return m_dblDampingFactor;
}

double NeuralInputRational::GetMin() const { return 0.0; }

double NeuralInputRational::GetMax() const { return 1.0; }

bool NeuralInputRational::Save(FILE *fpOutput, size_t nInputId,
                               const IFeatureMap &p_featureMap) const {
  // The base class can write out the generic portion
  NeuralInputUnary::Save(fpOutput, nInputId, p_featureMap);
  fprintf(fpOutput, "Transform=rational\n");
  fprintf(fpOutput, "DampingFactor=%lg\n", m_dblDampingFactor);

  return true;
}

size_t NeuralInputRational::GetSize() const {
  return sizeof(*this) + GetExternalSize();
}

// Do a 'set COMPUTE_BOOST_STATIC_ASSERT=1' command, then build in order to get
// error output that contains the struct size.
#if defined(COMPUTE_BOOST_STATIC_ASSERT)
template <int s>
struct Sizeof_NeuralInputRational;
Sizeof_NeuralInputRational<sizeof(NeuralInputRational) -
                           sizeof(NeuralInputUnary)>
    sizeof_NeuralInputRational;
#endif

// Use static assert to force the user to consider the external size and
// serialization if a new member is added. For each new member please add it to
// the template serialize and generate all the bin files for all the nets in
// searchgold. If the new member has an external size, then the size has to be
// reported within this function.
#ifdef DEBUG
BOOST_STATIC_ASSERT(sizeof(NeuralInputRational) - sizeof(NeuralInputUnary) ==
                    8);
#else
BOOST_STATIC_ASSERT(sizeof(NeuralInputRational) - sizeof(NeuralInputUnary) ==
                    8);
#endif

size_t NeuralInputRational::GetExternalSize() const {
  return NeuralInputUnary::GetExternalSize();
}

NeuralInputCached::~NeuralInputCached() {}

NeuralInputCached::NeuralInputCached(size_t nCacheSize,
                                     NeuralInputUnary *pChild)
    : NeuralInputUnary(pChild->GetAssociatedFeature()) {
  SetSegments(pChild->GetSegments());

  m_input.reset(pChild);
  m_resultCache.reset(new double[nCacheSize]);

  if (!m_resultCache.get()) {
    m_cacheSize = 0;  // not enough memory to cache results
    return;
  }
  m_cacheSize = nCacheSize;
  for (UInt32 i = 0; i < nCacheSize; i++) {
    m_resultCache[i] = m_input->EvaluateInput(i);
  }
}

void NeuralInputCached::FillBond(UnionBondInput &p_data) const {
  m_input->FillBond(p_data);
}

bool NeuralInputCached::Equal(const NeuralInput *p_input) const {
  if (!NeuralInputUnary::Equal(p_input)) {
    return false;
  }

  const NeuralInputCached *other =
      dynamic_cast<const NeuralInputCached *>(p_input);
  if (!other) {
    return false;
  }
  if (m_cacheSize != other->m_cacheSize ||
      !m_input->Equal(other->m_input.get())) {
    return false;
  }

  return true;
}

double NeuralInputCached::EvaluateInput(UInt32 val) const {
  if (val < m_cacheSize) {
    return m_resultCache[val];
  }

  return m_input->EvaluateInput(val);
}

double NeuralInputCached::GetMax() const { return m_input->GetMax(); }

double NeuralInputCached::GetMin() const { return m_input->GetMin(); }

bool NeuralInputCached::Save(FILE *fpOutput, size_t nInputId,
                             const IFeatureMap &p_featureMap) const {
  NeuralInputUnary *input = const_cast<NeuralInputUnary *>(m_input.get());
  input->SetSegments(GetSegments());

  return m_input->Save(fpOutput, nInputId, p_featureMap);
}

bool NeuralInputCached::Train(double dblLearningRate, double outputHigh,
                              double outputLow, double dblOutputDelta,
                              UInt32 inputHigh[], UInt32 inputLow[]) {
  bool ret;

  // Zero out cache since training can change the transform.
  ret = m_input->Train(dblLearningRate, outputHigh, outputLow, dblOutputDelta,
                       inputHigh, inputLow);
  if (ret) m_cacheSize = 0;
  return ret;
}

NeuralInputUnary *NeuralInputCached::Load(size_t nCacheSize,
                                          NeuralInputUnary *pChild) {
#ifdef INPUT_CACHE

  if (pChild) {
    return new NeuralInputCached(nCacheSize, pChild);
  }
#endif

  return pChild;
}

const NeuralInputUnary *NeuralInputCached::GetBaseInput() const {
  return m_input.get();
}

size_t NeuralInputCached::GetSize() const {
  return sizeof(*this) + GetExternalSize();
}

// Do a 'set COMPUTE_BOOST_STATIC_ASSERT=1' command, then build in order to get
// error output that contains the struct size.
#if defined(COMPUTE_BOOST_STATIC_ASSERT)
template <int s>
struct Sizeof_NeuralInputCached;
Sizeof_NeuralInputCached<sizeof(NeuralInputCached) - sizeof(NeuralInputUnary)>
    sizeof_NeuralInputCached;
#endif

// Use static assert to force the user to consider the external size and
// serialization if a new member is added. For each new member please add it to
// the template serialize and generate all the bin files for all the nets in
// searchgold. If the new member has an external size, then the size has to be
// reported within this function.
#ifdef DEBUG
BOOST_STATIC_ASSERT(sizeof(NeuralInputCached) - sizeof(NeuralInputUnary) == 24);
#else
BOOST_STATIC_ASSERT(sizeof(NeuralInputCached) - sizeof(NeuralInputUnary) == 24);
#endif

size_t NeuralInputCached::GetExternalSize() const {
  size_t externalSize = 0;

  if (m_cacheSize && m_resultCache.get()) {
    externalSize += m_cacheSize * sizeof(m_resultCache[0]);
  }

  externalSize += m_input->GetSize();

  externalSize += NeuralInputUnary::GetExternalSize();

  return externalSize;
}

NeuralInputTanh::NeuralInputTanh()
    : m_cInputs(0), m_locked(false), m_threshold(0.0) {
  memset(m_rgId, 0, c_maxInputs * sizeof(m_rgId[0]));
  memset(m_rgWeights, 0, c_maxInputs * sizeof(m_rgWeights[0]));
}

NeuralInputTanh::NeuralInputTanh(const NeuralInputTanhBondData &p_data)
    : NeuralInput(p_data),
      m_cInputs(p_data.m_cInputs),
      m_locked(p_data.m_locked),
      m_threshold(p_data.m_threshold) {
  for (size_t i = 0; i < p_data.m_rgId.size(); ++i) {
    m_rgId[i] = p_data.m_rgId[i];
  }

  for (size_t i = 0; i < p_data.m_rgWeights.size(); ++i) {
    m_rgWeights[i] = p_data.m_rgWeights[i];
  }
}

void NeuralInputTanh::FillBond(UnionBondInput &p_data) const {
  p_data.m_inputType = "tanh";
  // Fill bond struct of the class.
  NeuralInputTanhBondData data;
  FillBondData(data);
  UnionNeuralInputTanhBondData unionData;
  unionData.m_cached = false;
  unionData.m_neuralInputTanhBondData.set(data);
  p_data.m_tanh.set(unionData);
}

void NeuralInputTanh::FillBondData(NeuralInputTanhBondData &p_data) const {
  // Fill base class.
  NeuralInput::FillBondData(p_data);
  p_data.m_cInputs = static_cast<UInt32>(m_cInputs);
  p_data.m_locked = m_locked;
  p_data.m_threshold = m_threshold;
  for (size_t i = 0; i < c_maxInputs; i++) {
    p_data.m_rgId.push_back(m_rgId[i]);
    p_data.m_rgWeights.push_back(m_rgWeights[i]);
  }
}

bool NeuralInputTanh::Equal(const NeuralInput *p_input) const {
  if (!NeuralInput::EqualInternal(p_input)) {
    return false;
  }

  const NeuralInputTanh *other = dynamic_cast<const NeuralInputTanh *>(p_input);
  if (!other) {
    return false;
  }
  if (m_cInputs != other->m_cInputs || m_locked != other->m_locked ||
      m_threshold != other->m_threshold ||
      memcmp(m_rgId, other->m_rgId, c_maxInputs * sizeof(m_rgId[0])) ||
      memcmp(m_rgWeights, other->m_rgWeights,
             c_maxInputs * sizeof(m_rgWeights[0]))) {
    return false;
  }

  return true;
}

NeuralInputTanh *NeuralInputTanh::Load(DynamicRank::Config &p_config,
                                       const char *szSection,
                                       IFeatureMap &p_featureMap) {
  char buff[256];

  // This transform is different than the others because it may have
  // "Name" defined and still have multiple inputs...
  if (szSection == NULL) {
    return NULL;
  }

  NeuralInputTanh *result = new NeuralInputTanh();
  result->m_cInputs = 0;
  result->m_locked = p_config.GetBoolParameter(szSection, "locked", false);
  result->m_threshold =
      p_config.GetDoubleParameter(szSection, "Threshold", 0.0);

  int inputs = 0;
  bool featurePresentInFeatureMap = true;
  while (inputs < result->c_maxInputs) {
    if (inputs == 0) {
      // strcpy_s(buff, sizeof(buff), "Name");
      strcpy(buff, "Name");
    } else {
      if (_snprintf_s(buff, sizeof(buff), _TRUNCATE, "Name:%d", inputs + 1) ==
          -1) {
        return NULL;
      }
    }
    if (!p_config.GetStringParameter(szSection, buff, buff, sizeof(buff))) {
      break;
    }

    // continue to read configuration even if its not present in the feature
    // map, it will be needed for cloning the net
    if (!p_featureMap.ObtainFeatureIndex(buff, result->m_rgId[inputs])) {
      featurePresentInFeatureMap = false;
    }

    if (inputs == 0) {
      strcpy(buff, "Weight");
    } else {
      if (_snprintf_s(buff, sizeof(buff), _TRUNCATE, "Weight:%d", inputs + 1) ==
          -1) {
        return NULL;
      }
    }

    result->m_rgWeights[inputs] =
        p_config.GetDoubleParameter(szSection, buff, 0.01);

    // Unless we increment input count, the value we stored in the array
    // m_rgWeights wont matter. If this variable is set to false once,
    // it will be false for all the subsequent iterations of the loop
    // there cannot be name:3 present without name:2 in the featuremap
    // this is as per the old behavior as of Dec 15, 2008.
    if (featurePresentInFeatureMap) {
      result->m_cInputs++;
    }
    ++inputs;
  }

  return result;
}

void NeuralInputTanh::GetAllAssociatedFeatures(
    std::vector<UInt32> &associatedFeaturesList) const {
  for (size_t i = 0; i < m_cInputs; ++i) {
    associatedFeaturesList.push_back(m_rgId[i]);
  }
}

double NeuralInputTanh::GetMin() const { return -1.0; }

double NeuralInputTanh::GetMax() const { return 1.0; }

double NeuralInputTanh::Evaluate(UInt32 input[]) const {
  double sum = m_threshold;
  for (size_t i = 0; i < m_cInputs; ++i) {
    sum += log((double)input[m_rgId[i]] + 1) * m_rgWeights[i];
  }
  return tanh(sum);
}

bool NeuralInputTanh::Save(FILE *fpOutput, size_t nInputId,
                           const IFeatureMap &p_featureMap) const {
  // All we can do here is write out the section name and the
  // feature name
  NeuralInput::Save(fpOutput, nInputId, p_featureMap);

  fprintf(fpOutput, "transform=tanh\n");

  // Create a stack buffer for the feature name
  char rgInput[1024];
  fprintf(fpOutput, "Threshold=%lg\n", m_threshold);
  fprintf(fpOutput, "Locked=%s\n", m_locked ? "TRUE" : "FALSE");

  for (size_t i = 0; i < m_cInputs; i++) {
    if (!p_featureMap.GetFeatureName(m_rgId[i], rgInput, 1024)) {
      return false;
    }
    if (i == 0) {
      fprintf(fpOutput, "Name=%s\n", rgInput);
      fprintf(fpOutput, "Weight=%lg\n", m_rgWeights[i]);
    } else {
      fprintf(fpOutput, "Name:%Id=%s\n", i + 1, rgInput);
      fprintf(fpOutput, "Weight:%Id=%lg\n", i + 1, m_rgWeights[i]);
    }
  }
  return true;
}

bool NeuralInputTanh::Train(double dblLearningRate, double outputHigher,
                            double outputLower, double dblOutputDelta,
                            UInt32 inputHigh[], UInt32 inputLow[]) {
  double dblGradientHigher = (1 - outputHigher * outputHigher);
  double dblDeltaHigher = dblOutputDelta * dblGradientHigher;

  double dblGradientLower = (1 - outputLower * outputLower);
  double dblDeltaLower = dblOutputDelta * dblGradientLower;
  double dblOldWeight = m_rgWeights[0];

  // Compute node weights
  for (size_t j = 0; j < m_cInputs; j++) {
    m_rgWeights[j] -= dblLearningRate *
                      (dblDeltaHigher * log((double)inputHigh[m_rgId[j]] + 1) -
                       dblDeltaLower * log((double)inputLow[m_rgId[j]] + 1));
  }

  // Compute hidden node threshold
  if (m_locked && m_cInputs == 1) {
    // for a locked input change the threshold to match the new weight
    m_threshold = m_threshold * m_rgWeights[0] / dblOldWeight;
  } else {
    m_threshold -= dblLearningRate * (dblDeltaHigher - dblDeltaLower);
  }
  return true;
}

size_t NeuralInputTanh::GetSize() const {
  return sizeof(*this) + GetExternalSize();
}

// Do a 'set COMPUTE_BOOST_STATIC_ASSERT=1' command, then build in order to get
// error output that contains the struct size.
#if defined(COMPUTE_BOOST_STATIC_ASSERT)
template <int s>
struct Sizeof_NeuralInputTanh;
Sizeof_NeuralInputTanh<sizeof(NeuralInputTanh) - sizeof(NeuralInput)>
    sizeof_NeuralInputTanh;
#endif

// Use static assert to force the user to consider the external size and
// serialization if a new member is added. For each new member please add it to
// the template serialize and generate all the bin files for all the nets in
// searchgold. If the new member has an external size, then the size has to be
// reported within this function.
#ifdef DEBUG
BOOST_STATIC_ASSERT(sizeof(NeuralInputTanh) - sizeof(NeuralInput) == 384);
#else
BOOST_STATIC_ASSERT(sizeof(NeuralInputTanh) - sizeof(NeuralInput) == 384);
#endif

size_t NeuralInputTanh::GetExternalSize() const {
  return NeuralInput::GetExternalSize();
}

double NeuralInputTanhUnary::EvaluateInput(UInt32 input) const {
  return tanh(m_dblThreshold + log((double)input + 1) * m_dblWeights);
}

bool NeuralInputTanhUnary::Save(FILE *fpOutput, size_t nInputId,
                                const IFeatureMap &p_featureMap) const {
  NeuralInputUnary::Save(fpOutput, nInputId, p_featureMap);

  fprintf(fpOutput, "Transform=tanh\n");
  fprintf(fpOutput, "Threshold=%lg\n", m_dblThreshold);
  fprintf(fpOutput, "Weight=%lg\n", m_dblWeights);
  fprintf(fpOutput, "Locked=%s\n", m_fLocked ? "TRUE" : "FALSE");

  return true;
}

NeuralInputTanhUnary::NeuralInputTanhUnary(UInt32 iFeature, double dblWeights,
                                           double dblThreshold, bool fLocked)
    : m_dblThreshold(dblThreshold),
      m_dblWeights(dblWeights),
      m_fLocked(fLocked),
      NeuralInputUnary(iFeature) {}

NeuralInputTanhUnary::NeuralInputTanhUnary(
    const NeuralInputTanhUnaryBondData &p_data)
    : NeuralInputUnary(p_data),
      m_fLocked(p_data.m_fLocked),
      m_dblWeights(p_data.m_dblWeights),
      m_dblThreshold(p_data.m_dblThreshold) {}

void NeuralInputTanhUnary::FillBond(UnionBondInput &p_data) const {
  p_data.m_inputType = "tanh";
  // Fill bond struct of the class.
  NeuralInputTanhUnaryBondData data;
  FillBondData(data);
  UnionNeuralInputTanhBondData unionData;
  unionData.m_cached = true;
  unionData.m_neuralInputTanhUnaryBondData.set(data);
  p_data.m_tanh.set(unionData);
}

void NeuralInputTanhUnary::FillBondData(
    NeuralInputTanhUnaryBondData &p_data) const {
  // Fill base class.
  NeuralInputUnary::FillBondData(p_data);

  p_data.m_fLocked = m_fLocked;
  p_data.m_dblWeights = m_dblWeights;
  p_data.m_dblThreshold = m_dblThreshold;
}

bool NeuralInputTanhUnary::Equal(const NeuralInput *p_input) const {
  if (!NeuralInputUnary::Equal(p_input)) {
    return false;
  }

  const NeuralInputTanhUnary *other =
      dynamic_cast<const NeuralInputTanhUnary *>(p_input);
  if (!other) {
    return false;
  }
  if (m_fLocked != other->m_fLocked || m_dblWeights != other->m_dblWeights ||
      m_dblThreshold != other->m_dblThreshold) {
    return false;
  }

  return true;
}

NeuralInputTanhUnary *NeuralInputTanhUnary::Load(DynamicRank::Config &p_config,
                                                 const char *szSection,
                                                 IFeatureMap &p_featureMap) {
  UInt32 iFeature;

  if (!NeuralInputUnary::ReadAssociatedFeature(p_config, szSection,
                                               p_featureMap, &iFeature)) {
    return NULL;
  }

  // Make sure it is not a Multiple-Input tanh Transform...
  char szSecondFeatureName[256];
  if (p_config.GetStringParameter(szSection, "Name:2", szSecondFeatureName,
                                  sizeof(szSecondFeatureName))) {
    return NULL;
  }

  double dblThreshold =
      p_config.GetDoubleParameter(szSection, "Threshold", 0.0);
  bool fLocked = p_config.GetBoolParameter(szSection, "locked", false);
  double dblWeight = p_config.GetDoubleParameter(szSection, "Weight", 0.01);

  NeuralInputTanhUnary *pNeuralInputTanhUnary =
      new NeuralInputTanhUnary(iFeature, dblWeight, dblThreshold, fLocked);
  return pNeuralInputTanhUnary;
}

double NeuralInputTanhUnary::GetWeight() const { return m_dblWeights; }

double NeuralInputTanhUnary::GetThreshold() const { return m_dblThreshold; }

double NeuralInputTanhUnary::GetMin() const { return -1.0; }

double NeuralInputTanhUnary::GetMax() const { return 1.0; }

bool NeuralInputTanhUnary::Train(double dblLearningRate, double outputHigher,
                                 double outputLower, double dblOutputDelta,
                                 UInt32 inputHigh[], UInt32 inputLow[]) {
  double dblGradientHigher = (1 - outputHigher * outputHigher);
  double dblDeltaHigher = dblOutputDelta * dblGradientHigher;

  double dblGradientLower = (1 - outputLower * outputLower);
  double dblDeltaLower = dblOutputDelta * dblGradientLower;
  double dblOldWeight = m_dblWeights;

  // Compute node weight
  m_dblWeights -= dblLearningRate *
                  (dblDeltaHigher * log((double)inputHigh[m_iFeature] + 1) -
                   dblDeltaLower * log((double)inputLow[m_iFeature] + 1));

  // Compute hidden node threshold
  if (m_fLocked) {
    // for a locked input change the threshold to match the new weight
    m_dblThreshold = m_dblThreshold * m_dblWeights / dblOldWeight;
  } else {
    m_dblThreshold -= dblLearningRate * (dblDeltaHigher - dblDeltaLower);
  }
  return true;
}

size_t NeuralInputTanhUnary::GetSize() const {
  return sizeof(*this) + GetExternalSize();
}

// Do a 'set COMPUTE_BOOST_STATIC_ASSERT=1' command, then build in order to get
// error output that contains the struct size.
#if defined(COMPUTE_BOOST_STATIC_ASSERT)
template <int s>
struct Sizeof_NeuralInputTanhUnary;
Sizeof_NeuralInputTanhUnary<sizeof(NeuralInputTanhUnary) -
                            sizeof(NeuralInputUnary)>
    sizeof_NeuralInputTanhUnary;
#endif

size_t NeuralInputTanhUnary::GetExternalSize() const {
  return NeuralInputUnary::GetExternalSize();
}

double NeuralInputUseAsFloat::EvaluateInput(UInt32 input) const {
  return (double)(*(float *)(&input));
}

bool NeuralInputUseAsFloat::Save(FILE *fpOutput, size_t nInputId,
                                 const IFeatureMap &p_featureMap) const {
  NeuralInputUnary::Save(fpOutput, nInputId, p_featureMap);

  fprintf(fpOutput, "Transform=floatdata\n");

  return true;
}

NeuralInputUseAsFloat::NeuralInputUseAsFloat() : NeuralInputUnary() {}

NeuralInputUseAsFloat::NeuralInputUseAsFloat(UInt32 iFeature)
    : NeuralInputUnary(iFeature) {}

NeuralInputUseAsFloat::NeuralInputUseAsFloat(
    const NeuralInputUseAsFloatBondData &p_data)
    : NeuralInputUnary(p_data) {}

void NeuralInputUseAsFloat::FillBond(UnionBondInput &p_data) const {
  p_data.m_inputType = "floatdata";
  // Fill bond struct of the class.
  NeuralInputUseAsFloatBondData data;
  FillBondData(data);
  p_data.m_floatdata.set(data);
}

void NeuralInputUseAsFloat::FillBondData(
    NeuralInputUseAsFloatBondData &p_data) const {
  // Fill base class.
  NeuralInputUnary::FillBondData(p_data);
}

void NeuralInputUseAsFloat::CopyFrom(
    const NeuralInputUseAsFloat &p_neuralInputUseAsFloat) {
  NeuralInputUnary::CopyFrom(p_neuralInputUseAsFloat);
}

bool NeuralInputUseAsFloat::Equal(const NeuralInput *p_input) const {
  if (!NeuralInputUnary::Equal(p_input)) {
    return false;
  }

  if (!dynamic_cast<const NeuralInputUseAsFloat *>(p_input)) {
    return false;
  }

  return true;
}

NeuralInputUseAsFloat *NeuralInputUseAsFloat::Load(
    DynamicRank::Config &p_config, const char *szSection,
    IFeatureMap &p_featureMap) {
  UInt32 iFeature;

  if (!NeuralInputUnary::ReadAssociatedFeature(p_config, szSection,
                                               p_featureMap, &iFeature)) {
    return NULL;
  }

  NeuralInputUseAsFloat *pNeuralInputUseAsFloat =
      new NeuralInputUseAsFloat(iFeature);
  return pNeuralInputUseAsFloat;
}

double NeuralInputUseAsFloat::GetMin() const { return 0.0; }

double NeuralInputUseAsFloat::GetMax() const { return 1.0; }

size_t NeuralInputUseAsFloat::GetSize() const {
  return sizeof(*this) + GetExternalSize();
}

// Do a 'set COMPUTE_BOOST_STATIC_ASSERT=1' command, then build in order to get
// error output that contains the struct size.
#if defined(COMPUTE_BOOST_STATIC_ASSERT)
template <int s>
struct Sizeof_NeuralInputUseAsFloat;
Sizeof_NeuralInputUseAsFloat<sizeof(NeuralInputUseAsFloat) -
                             sizeof(NeuralInputUnary)>
    sizeof_NeuralInputUseAsFloat;
#endif

// Use static assert to force the user to consider the external size and
// serialization if a new member is added. For each new member please add it to
// the template serialize and generate all the bin files for all the nets in
// searchgold. If the new member has an external size, then the size has to be
// reported within this function.
#ifdef DEBUG
BOOST_STATIC_ASSERT(sizeof(NeuralInputUseAsFloat) - sizeof(NeuralInputUnary) ==
                    0);
#else
BOOST_STATIC_ASSERT(sizeof(NeuralInputUseAsFloat) - sizeof(NeuralInputUnary) ==
                    0);
#endif

size_t NeuralInputUseAsFloat::GetExternalSize() const {
  return NeuralInputUnary::GetExternalSize();
}

BulkOptimizationUnsupported::BulkOptimizationUnsupported(
    const std::string &p_message)
    : std::runtime_error(p_message) {}
