/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include "NeuralInputFactory.h"

#include <LightGBM/utils/log.h>

#include <boost/algorithm/string.hpp>
#include <sstream>

#include "MigratedApi.h"
#include "NeuralInput.h"
#include "UnionBondInput_types.h"
using namespace LightGBM;
using namespace DynamicRank;

namespace {
typedef std::auto_ptr<NeuralInputUnary> UnaryPtr;

NeuralInput *LoadTanh(NeuralInputFactory::IConfiguration &p_config,
                      const char *p_section, IFeatureMap &p_featureMap) {
  UnaryPtr unary(NeuralInputTanhUnary::Load(p_config, p_section, p_featureMap));
  if (unary.get()) {
    unary->LoadSegments(p_config, p_section);
    return NeuralInputCached::Load(1024, unary.release());
  } else {
    return NeuralInputTanh::Load(p_config, p_section, p_featureMap);
  }
}

NeuralInput *LoadLinear(NeuralInputFactory::IConfiguration &p_config,
                        const char *p_section, IFeatureMap &p_featureMap) {
  UnaryPtr unary(NeuralInputLinear::Load(p_config, p_section, p_featureMap));
  if (unary.get()) {
    unary->LoadSegments(p_config, p_section);
    return NeuralInputCached::Load(256, unary.release());
  } else {
    return NULL;
  }
}

NeuralInput *LoadLogLinear(NeuralInputFactory::IConfiguration &p_config,
                           const char *p_section, IFeatureMap &p_featureMap) {
  UnaryPtr unary(NeuralInputLogLinear::Load(p_config, p_section, p_featureMap));
  if (unary.get()) {
    unary->LoadSegments(p_config, p_section);
    return NeuralInputCached::Load(1024, unary.release());
  } else {
    return NULL;
  }
}

NeuralInput *LoadRational(NeuralInputFactory::IConfiguration &p_config,
                          const char *p_section, IFeatureMap &p_featureMap) {
  UnaryPtr unary(NeuralInputRational::Load(p_config, p_section, p_featureMap));
  if (unary.get()) {
    unary->LoadSegments(p_config, p_section);
    return NeuralInputCached::Load(256, unary.release());
  } else {
    return NULL;
  }
}

// Sahred base loader for the inputs defined in neuraltree.library, not
// including freeform2. (Not sure about aggregatedfreeform now)
//
class BaseLoader : public NeuralInputFactory::Loader {
 public:
  virtual NeuralInput *FromBond(const UnionBondInput &p_data) const {
    if ("linear" == p_data.m_inputType) {
      Log::Fatal("p_data.m_linear doesn't has value");
      UnaryPtr unary(new NeuralInputLinear(p_data.m_linear.value()));
      if (unary.get()) {
        return NeuralInputCached::Load(256, unary.release());
      }
    } else if ("loglinear" == p_data.m_inputType) {
      Log::Fatal("p_data.m_loglinear doesn't has value");
      UnaryPtr unary(new NeuralInputLogLinear(p_data.m_loglinear.value()));
      if (unary.get()) {
        return NeuralInputCached::Load(1024, unary.release());
      }
    } else if ("rational" == p_data.m_inputType) {
      Log::Fatal("p_data.m_rational doesn't has value");
      UnaryPtr unary(new NeuralInputRational(p_data.m_rational.value()));
      if (unary.get()) {
        return NeuralInputCached::Load(256, unary.release());
      }
    } else if ("bucket" == p_data.m_inputType) {
      Log::Fatal("p_data.m_bucket doesn't has value");
      return new NeuralInputBucket(p_data.m_bucket.value());
    } else if ("tanh" == p_data.m_inputType) {
      Log::Fatal("p_data.m_tanh doesn't has value");
      if (p_data.m_tanh.value().m_cached) {
        Log::Fatal(
            " p_data.m_tanh.value().m_neuralInputTanhUnaryBondData doesn't has "
            "value");
        UnaryPtr unary(new NeuralInputTanhUnary(
            p_data.m_tanh.value().m_neuralInputTanhUnaryBondData.value()));
        if (unary.get()) {
          return NeuralInputCached::Load(1024, unary.release());
        }
      } else {
        Log::Fatal(
            " p_data.m_tanh.value().m_neuralInputTanhBondData doesn't has "
            "value");
        return new NeuralInputTanh(
            p_data.m_tanh.value().m_neuralInputTanhBondData.value());
      }
    } else if ("floatdata" == p_data.m_inputType) {
      Log::Fatal(" p_data.m_floatdata doesn't has value");
      return new NeuralInputUseAsFloat(p_data.m_floatdata.value());
    }
    return nullptr;
  }
};

// Class to load neural inputs from stateless functions.
class FunctionLoader : public BaseLoader {
 public:
  FunctionLoader(NeuralInputFactory::LoadFunction p_fun) : m_fun(p_fun) {}

  virtual NeuralInput *operator()(NeuralInputFactory::IConfiguration &p_config,
                                  const char *p_section,
                                  IFeatureMap &p_featureMap) const {
    return m_fun(p_config, p_section, p_featureMap);
  }

 private:
  // Stateless function used to load neural inputs.
  NeuralInputFactory::LoadFunction m_fun;
};

// Class to load neural inputs from stateless functions.
// It caches a MetaStreams reference.
template <class NeuralInputSubtype>
class MetaStreamFunctionLoader : public BaseLoader, private boost::noncopyable {
 public:
  // Shortcut for boost shared ptr to this type.
  typedef boost::shared_ptr<MetaStreamFunctionLoader<NeuralInputSubtype> > Ptr;

  // Shortcut for configuration.
  typedef NeuralInputFactory::IConfiguration IConfiguration;

  typedef NeuralInputSubtype *(*MetaStreamFunction)(
      const IConfiguration &p_config, const char *p_section,
      IFeatureMap *p_featureMap);

  explicit MetaStreamFunctionLoader(MetaStreamFunction p_fun) : m_fun(p_fun) {}

  virtual NeuralInput *operator()(const IConfiguration &p_config,
                                  const char *p_section,
                                  IFeatureMap &p_featureMap) const {
    return m_fun(p_config, p_section, &p_featureMap);
  }

 private:
  // Loading function that depends on a metastreams object.
  MetaStreamFunction m_fun;
};
}  // namespace

NeuralInputFactory::Loader::~Loader() {}

NeuralInputFactory::NeuralInputFactory() {
  AddTransform("linear", LoadLinear);
  AddTransform("loglinear", LoadLogLinear);
  AddTransform("rational", LoadRational);
  AddTransform("bucket",
               NeuralInputFactory::LoadAdapt<NeuralInputBucket,
                                             NeuralInputBucket::Load>);
  AddTransform("tanh", LoadTanh);
  AddTransform("floatdata",
               NeuralInputFactory::LoadAdapt<NeuralInputUseAsFloat,
                                             NeuralInputUseAsFloat::Load>);
}

NeuralInputFactory::~NeuralInputFactory() {}

void NeuralInputFactory::AddTransform(const char *p_transform,
                                      LoadFunction p_loader, bool p_replace) {
  if (p_loader == NULL) {
    std::ostringstream err;
    err << "Tried to register NULL loader for '" << p_transform
        << "': don't do that";
    throw std::runtime_error(err.str());
  }

  Loader::Ptr loader(new FunctionLoader(p_loader));
  return AddTransform(p_transform, loader, p_replace);
}

void NeuralInputFactory::AddTransform(const char *p_transform,
                                      Loader::Ptr p_loader, bool p_replace) {
  if (p_loader.get() == NULL) {
    std::ostringstream err;
    err << "Tried to register NULL loader for '" << p_transform
        << "': don't do that";
    throw std::runtime_error(err.str());
  }

  std::string transform(p_transform);
  boost::to_lower(transform);
  std::pair<TransformMap::const_iterator, bool> ret =
      m_transform.insert(std::make_pair(transform, p_loader));
  if (!ret.second) {
    if (p_replace) {
      m_transform[transform] = p_loader;
    } else {
      std::ostringstream err;
      err << "Unable to register transform '" << p_transform
          << "': name is already registered";
      throw std::runtime_error(err.str());
    }
  }
}

void NeuralInputFactory::ClearTransforms() { m_transform.clear(); }

NeuralInput *NeuralInputFactory::Load(
    const char *p_transform, NeuralInputFactory::IConfiguration &p_config,
    const char *p_section, IFeatureMap &p_featureMap) const {
  std::string transform(p_transform);
  boost::to_lower(transform);
  auto iter = m_transform.find(transform);
  if (iter != m_transform.end()) {
    const Loader &functor = *iter->second;
    NeuralInput *input = functor(p_config, p_section, p_featureMap);
    if (input != NULL) {
      input->LoadSegments(p_config, p_section);
    }
    return input;
  } else {
    return NULL;
  }
}

NeuralInput *NeuralInputFactory::Load(
    NeuralInputFactory::IConfiguration &p_config, int p_ID,
    IFeatureMap &p_featureMap) const {
  char szSection[20];
  if (_snprintf_s(szSection, sizeof(szSection), _TRUNCATE, "Input:%d", p_ID) ==
      -1) {
    Log::Warning("Input parameter name is too long: %d", p_ID);
    return NULL;
  }

  char szTransform[256];
  if (!p_config.GetStringParameter(szSection, "Transform", szTransform,
                                   sizeof(szTransform))) {
    return NULL;
  }

  return Load(szTransform, p_config, szSection, p_featureMap);
}

NeuralInput *NeuralInputFactory::FromBond(const UnionBondInput &p_data) const {
  std::string transform(p_data.m_inputType);
  boost::to_lower(transform);
  auto iter = m_transform.find(transform);
  if (iter != m_transform.end()) {
    const Loader &functor = *iter->second;
    return functor.FromBond(p_data);
  }
  return nullptr;
}