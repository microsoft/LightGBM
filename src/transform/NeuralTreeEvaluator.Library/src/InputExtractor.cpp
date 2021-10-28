/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include "InputExtractor.h"

#include <LightGBM/utils/log.h>
#include <NeuralInputFactory.h>
#include <NeuralInputFreeForm2.h>

#include <boost/algorithm/string.hpp>
#include <fstream>
#include <sstream>

#include "LocalFactoryHolder.h"
#include "MigratedApi.h"

using namespace DynamicRank;
using namespace LightGBM;

InputExtractor::InputExtractor(
    DynamicRank::Config *config,
    boost::shared_ptr<MinimalFeatureMap> featureMap,
    std::vector<boost::shared_ptr<const DynamicRank::NeuralInput> > &inputs) {
  m_config = config;
  m_featureMap = featureMap;
  for (std::vector<boost::shared_ptr<const NeuralInput> >::const_iterator it =
           inputs.cbegin();
       it != inputs.cend(); ++it) {
    m_inputs.push_back(*it);
  }
}

InputExtractor::~InputExtractor(void) {}

InputExtractor *InputExtractor::CreateFromConfig(DynamicRank::Config *config) {
  LocalFactoryHolder factoryHolder;
  // input id starts from 0.
  size_t inputId = 0;
  char section[20];
  _snprintf_s(section, sizeof(section), _TRUNCATE, "Input:%Iu", inputId);
  if (!config->DoesSectionExist(section)) {
    inputId += 1;
    _snprintf_s(section, sizeof(section), _TRUNCATE, "Input:%Iu", inputId);
  }
  NeuralInputFactory &inputFactory = factoryHolder.GetInputFactory();
  boost::shared_ptr<MinimalFeatureMap> featureMap(new MinimalFeatureMap());
  std::vector<boost::shared_ptr<const NeuralInput> > inputList;
  bool hasAggregatedFreeForms = false;

  while (config->DoesSectionExist(section)) {
    boost::shared_ptr<NeuralInput> input(
        inputFactory.Load(*config, (int)inputId, *featureMap));
    if (input == NULL) {
      Log::Fatal("Unable to load input in section %s", section);
      return NULL;
    }
    _snprintf_s(section, sizeof(section), _TRUNCATE, "Input:%Iu", ++inputId);
    inputList.push_back(input);
  }
  factoryHolder.PostLoad(hasAggregatedFreeForms);
  return new InputExtractor(config, featureMap, inputList);
}

InputExtractor *InputExtractor::CreateFromInputStr(const string &str) {
  DynamicRank::Config *config = DynamicRank::Config::GetRawConfiguration(str);
  if (config == NULL) {
    Log::Warning(
        "Unable to read input string as valid transform, in bad format or does "
        "not exist");
    return NULL;
  }
  return InputExtractor::CreateFromConfig(config);
}

InputExtractor *InputExtractor::CreateFromFreeform2(const char *freeform) {
  boost::shared_ptr<MinimalFeatureMap> featureMap(new MinimalFeatureMap());
  boost::shared_ptr<FreeForm2::NeuralInputFreeForm2> input;
  try {
    input = boost::shared_ptr<FreeForm2::NeuralInputFreeForm2>(
        new FreeForm2::NeuralInputFreeForm2(std::string(freeform), "freeform2",
                                            *featureMap));
    if (input == NULL) {
      Log::Fatal("CreateFromFreeform: Unable to parse freeform2 %s", freeform);
      return NULL;
    }
    std::unique_ptr<FreeForm2::Compiler> comp(
        FreeForm2::CompilerFactory::CreateExecutableCompiler(2));
    input->Compile(comp.get());
  } catch (const std::exception &) {
    // Failed to compile.
    return NULL;
  }

  std::vector<boost::shared_ptr<const NeuralInput> > inputList;
  inputList.push_back(input);
  return new InputExtractor(NULL, featureMap, inputList);
}

MinimalFeatureMap *InputExtractor::GetFeatureMap() const {
  return m_featureMap.get();
}

UInt32 InputExtractor::GetInputCount() const { return (UInt32)m_inputs.size(); }

const NeuralInput *InputExtractor::GetInput(UInt32 index) const {
  if (index >= m_inputs.size()) {
    return NULL;
  }
  return m_inputs[index].get();
}

const std::string InputExtractor::GetInputName(UInt32 index) const {
  UInt32 offset = 0;
  char section[20];
  _snprintf_s(section, sizeof(section), _TRUNCATE, "Input:0");
  if (!m_config->DoesSectionExist(section)) offset = 1;

  _snprintf_s(section, sizeof(section), _TRUNCATE, "Input:%d", index + offset);
  if (!m_config->DoesSectionExist(section)) {
    return "";
  }
  std::string strTransform;
  m_config->GetStringParameter(section, "Transform", strTransform);
  const char *transform = strTransform.c_str();
  std::string strName;

  // Linear/tanh/bucket/other transforms are unary, and all refer to their
  // feature by the "Name" key.
  const DynamicRank::NeuralInputCached *cachedInput =
      dynamic_cast<const DynamicRank::NeuralInputCached *>(
          m_inputs[index].get());
  if (cachedInput != NULL && dynamic_cast<const NeuralInputUnary *>(
                                 cachedInput->GetBaseInput()) != NULL) {
    m_config->GetStringParameter(section, "Name", strName);
  } else if (boost::iequals(transform, "freeform")) {
    m_config->GetStringParameter(section, "Expression", strName);
  } else if (boost::iequals(transform, "freeform2")) {
    strName = "";
    char line[20];
    std::string lineContents;
    UInt32 lineno = 1;
    _snprintf_s(line, sizeof(line), _TRUNCATE, "Line%u", lineno);
    while (m_config->DoesParameterExist(section, line)) {
      if (strName.length() != 0) {
        strName.append(" ");
      }
      m_config->GetStringParameter(section, line, lineContents);
      strName.append(lineContents);
      lineContents = "";
      _snprintf_s(line, sizeof(line), _TRUNCATE, "Line%u", ++lineno);
    }
  } else {
    strName = "????";
  }
  strTransform.append(".");
  strTransform.append(strName);
  return strTransform;
}
