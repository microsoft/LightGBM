/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include "NeuralInputFreeForm2.h"

#include <LightGBM/utils/log.h>
#include <limits.h>

#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/cast.hpp>
#include <boost/scoped_array.hpp>
#include <cfloat>
#include <sstream>

#include "BinaryOperator.h"
#include "Compiler.h"
#include "Conditional.h"
#include "ConvertExpression.h"
#include "Executable.h"
#include "Expression.h"
#include "FreeForm2Assert.h"
#include "FreeForm2Compiler.h"
#include "FreeForm2CompilerFactory.h"
#include "FreeForm2Executable.h"
#include "FreeForm2Program.h"
#include "FreeForm2Result.h"
#include "FreeForm2Utils.h"
#include "LiteralExpression.h"
#include "LlvmCompiler.h"
#include "OperatorExpression.h"
#include "Program.h"
#include "RefExpression.h"
#include "SimpleExpressionOwner.h"
#include "TypeManager.h"

using namespace LightGBM;

namespace {
// Gets the feature index for an input which transforms a single feature.
bool ReadAssociatedFeature(DynamicRank::Config &p_config, const char *p_section,
                           DynamicRank::IFeatureMap &p_featureMap,
                           UInt32 *p_feature) {
  *p_feature = static_cast<UInt32>(-1);

  if (p_section == NULL) {
    return false;
  }

  char szName[256];
  if (p_config.GetStringParameter(p_section, "Name", szName, sizeof(szName))) {
    if (!p_featureMap.ObtainFeatureIndex(szName, *p_feature)) {
      Log::Warning(
          "DR:ReadAssociatedFeature Could not find index for feature name: %s "
          "in section: %s",
          szName, p_section);
      return false;  // input error, invalid feature name
    }
    return true;
  }

  Log::Warning(
      "DR:ReadAssociatedFeature Could not find Name of the feature for "
      "section: %s",
      p_section);
  return false;  // feature is not specified
}
}  // namespace

FreeForm2::NeuralInputFreeForm2::NeuralInputFreeForm2()
    : m_transform("FreeForm2") {}

FreeForm2::NeuralInputFreeForm2::NeuralInputFreeForm2(
    const std::string &p_input, const char *p_transform,
    DynamicRank::IFeatureMap &p_map)
    : m_input(p_input),
      m_transform(p_transform),
      m_map(&p_map),
      m_fun(NULL),
      m_program(Program::Parse<Program::sexpression>(
          CStackSizedString(p_input.c_str(), p_input.size()), p_map, true,
          nullptr, NULL)) {
  Init();
}

FreeForm2::NeuralInputFreeForm2::NeuralInputFreeForm2(
    const std::string &p_input, const char *p_transform,
    DynamicRank::IFeatureMap &p_map, boost::shared_ptr<Program> p_program)
    : m_input(p_input),
      m_transform(p_transform),
      m_map(&p_map),
      m_fun(NULL),
      m_program(p_program) {
  Init();
}

void FreeForm2::NeuralInputFreeForm2::Init() {
  // Infer actual features used by this expression.
  std::set<UInt32> features;
  SetFromNeuralNetFeatures actualFeatures(features);
  m_program->ProcessFeaturesUsed(actualFeatures);

  // Copy the features into m_features.
  m_features.resize(features.size());
  std::copy(features.begin(), features.end(), m_features.begin());
}

double FreeForm2::NeuralInputFreeForm2::Evaluate(UInt32 p_input[]) const {
  try {
    return m_fun(NULL, p_input, NULL);
  } catch (const std::exception &e) {
  }
  Unreachable(__FILE__, __LINE__);
}

bool FreeForm2::NeuralInputFreeForm2::Train(
    double p_learningRate, double p_outputHigh, double p_outputLow,
    double p_outputDelta, UInt32 p_inputHigh[], UInt32 p_inputLow[]) {
  return false;
}

void FreeForm2::NeuralInputFreeForm2::GetAllAssociatedFeatures(
    std::vector<UInt32> &p_associatedFeaturesList) const {
  std::copy(m_features.begin(), m_features.end(),
            std::back_inserter(p_associatedFeaturesList));
}

void FreeForm2::NeuralInputFreeForm2::Compile(Compiler *p_compiler) {
  // Only compile once.
  if (m_program == NULL) {
    return;
  }

  try {
    if (p_compiler != NULL) {
      std::unique_ptr<CompilerResults> results =
          p_compiler->Compile(*m_program, false);
      const ExecutableCompilerResults *exec =
          boost::polymorphic_downcast<ExecutableCompilerResults *>(
              results.get());
      m_exec = exec->GetExecutable();
    } else {
      std::unique_ptr<Compiler> compiler(
          CompilerFactory::CreateExecutableCompiler(
              Compiler::c_defaultOptimizationLevel,
              CompilerFactory::SingleDocumentEvaluation));
      std::unique_ptr<CompilerResults> results =
          compiler->Compile(*m_program, false);
      const ExecutableCompilerResults *exec =
          boost::polymorphic_downcast<ExecutableCompilerResults *>(
              results.get());
      m_exec = exec->GetExecutable();
    }

    m_fun = m_exec->EvaluationFunction();

    // Free the expression tree from memory.
    m_program.reset();
  } catch (const std::exception &p_except) {
    Log::Fatal("NeuralInputFreeForm2::Compile,Failed to compile %s: %s",
               m_transform, p_except.what());
    Log::Fatal("NeuralInputFreeForm2::Compile,Failed to compile %s: %s",
               m_transform, GetStringRepresentation().c_str());
    throw;
  }
}

const FreeForm2::Program &FreeForm2::NeuralInputFreeForm2::GetProgram() const {
  return *m_program.get();
}

double FreeForm2::NeuralInputFreeForm2::GetMin() const { return -DBL_MAX; }

double FreeForm2::NeuralInputFreeForm2::GetMax() const { return DBL_MAX; }

size_t FreeForm2::NeuralInputFreeForm2::GetSize() const {
  return sizeof(NeuralInputFreeForm2) + GetExternalSize();
}

size_t FreeForm2::NeuralInputFreeForm2::GetExternalSize() const {
  size_t externalSize = sizeof(UInt32) * m_features.size();

  if (m_exec.get()) {
    externalSize += sizeof(FreeForm2::Executable) + m_exec->GetExternalSize();
  }

  externalSize += DynamicRank::NeuralInput::GetExternalSize();
  externalSize += m_input.capacity() * sizeof(std::string::value_type);
  return externalSize;
}

bool FreeForm2::NeuralInputFreeForm2::Save(
    FILE *p_out, size_t p_input, const DynamicRank::IFeatureMap &p_map) const {
  // Write header.
  bool success = DynamicRank::NeuralInput::Save(p_out, p_input, p_map);
  success = success && fprintf(p_out, "Transform=%s\n",
                               (m_transform != NULL ? m_transform : "<NULL>"));
  typedef boost::split_iterator<std::string::const_iterator> SplitIter;

  unsigned int numLine = 1;
  for (SplitIter iter = boost::make_split_iterator(
           m_input, boost::token_finder(boost::is_any_of("\r\n")));
       iter != SplitIter(); ++iter) {
    if (iter->size() > 0) {
      const char *str = &(*iter->begin());
      success = success && fprintf(p_out, "Line%u=%.*s\n", numLine,
                                   static_cast<int>(iter->size()), str);
      numLine++;
    }
  }

  return success;
}

std::string FreeForm2::NeuralInputFreeForm2::LoadProgram(
    DynamicRank::Config &p_config, const char *p_section,
    const DynamicRank::IFeatureMap *p_featureMap, const char *p_transform) {
  // Read multiple lines from config, and assemble them into a program.
  unsigned int numLine = 1;
  bool found = true;
  std::ostringstream program;
  do {
    std::ostringstream lineName;
    lineName << "Line" << numLine;
    std::string lineStr = lineName.str();
    std::string lineValue;
    found = p_config.GetStringParameter(p_section, lineStr.c_str(), lineValue);
    numLine++;
    program << lineValue << std::endl;
  } while (found);

  if (numLine == 2 && !found) {
    Log::Warning("NeuralInputFreeForm2::Load NeuralInputFreeForm2::Load %s",
                 p_section);
    return "";
  }

  // Guard against skipped line numbers by refusing to load when we find
  // something that looks like one.
  for (unsigned int i = 0; i < 10; i++) {
    std::ostringstream lineName;
    lineName << "Line" << numLine + i;
    std::string lineStr = lineName.str();
    std::string lineValue;
    found = p_config.GetStringParameter(p_section, lineStr.c_str(), lineValue);

    if (found) {
      Log::Warning(
          "NeuralInputFreeForm2::Found ignored parameter %s in section %s: did "
          "you skip line number %s?",
          lineStr.c_str(), p_section, to_string(numLine + i).c_str());
      return "";
    }
  }

  return program.str();
}

FreeForm2::NeuralInputFreeForm2 *FreeForm2::NeuralInputFreeForm2::Load(
    DynamicRank::Config &p_config, const char *p_section,
    DynamicRank::IFeatureMap *p_featureMap, const char *p_transform) {
  std::string programStr =
      LoadProgram(p_config, p_section, p_featureMap, p_transform);
  if (programStr.empty()) {
    return NULL;
  }

  try {
    return new NeuralInputFreeForm2(programStr, p_transform, *p_featureMap);
  } catch (const std::exception &p_except) {
    Log::Warning(
        "NeuralInputFreeForm2::Load Failed to load freeform2: %s (program is "
        "'%s')",
        p_except.what(), programStr.c_str());
    return NULL;
  }
}

std::string FreeForm2::NeuralInputFreeForm2::GetStringRepresentation() const {
  return m_input;
}

bool FreeForm2::NeuralInputFreeForm2::IsFreeForm2() const {
  // Only this one says true!!!
  // We will call BatchSerialize on Inputs that said yes to IsFreeForm2()
  // When do bond serialization.
  return true;
}

bool FreeForm2::NeuralInputFreeForm2::Equal(
    const DynamicRank::NeuralInput *p_other) const {
  const NeuralInputFreeForm2 *t =
      dynamic_cast<const NeuralInputFreeForm2 *>(p_other);

  if (t == nullptr || m_input != t->m_input ||
      m_features.size() != t->m_features.size()) {
    return false;
  }

  for (size_t i = 0; i < m_features.size(); ++i) {
    if (m_features[i] != t->m_features[i]) {
      return false;
    }
  }

  if ((m_exec == nullptr) != (t->m_exec == nullptr)) {
    return false;
  }
  if (m_exec == nullptr) {
    return true;
  }

  const LlvmExecutableImpl *left =
      dynamic_cast<const LlvmExecutableImpl *>(&m_exec->GetImplementation());
  const LlvmExecutableImpl *right =
      dynamic_cast<const LlvmExecutableImpl *>(&t->m_exec->GetImplementation());

  if ((left == nullptr) != (right == nullptr)) {
    return false;
  }

  if (left && right) {
    return (*left == *right);
  }

  return true;
}

void FreeForm2::NeuralInputFreeForm2::FillBond(
    DynamicRank::UnionBondInput &p_data) const {
  // Selector.
  p_data.m_inputType = c_freeform2_tranform;

  DynamicRank::NeuralInputFreeForm2BondData data;
  data.m_input = m_input;
  for (size_t i = 0; i < m_features.size(); ++i) {
    data.m_features.push_back(m_features[i]);
  }

  p_data.m_freeform2.set(data);

  // Other members will use BatchSerialize.
}

FreeForm2::NeuralInputFreeForm2::NeuralInputFreeForm2(
    const DynamicRank::UnionBondInput &p_data) {
  if ("freeform2" != p_data.m_inputType) {
    Log::Fatal("Input type '%s' is not supported. Accepted type is freeform2.",
               p_data.m_inputType.c_str());
  }
  const DynamicRank::NeuralInputFreeForm2BondData &data =
      p_data.m_freeform2.value();
  for (size_t i = 0; i < data.m_features.size(); ++i) {
    m_features.push_back(data.m_features[i]);
  }
  m_input = data.m_input;

  // Other members will use BatchUnSerialize.
}

void FreeForm2::NeuralInputCompiler::Compile(
    const std::vector<NeuralInputFreeForm2 *> &p_inputs,
    FreeForm2::Compiler &p_compiler) {
  if (p_inputs.empty()) {
    return;
  }

  for (unsigned int i = 0; i < p_inputs.size(); i++) {
    if (p_inputs[i] == NULL) {
      Log::Fatal("Input freeform is null.");
    }
    p_inputs[i]->Compile(&p_compiler);
  }
}
