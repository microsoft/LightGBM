/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include "LocalFactoryHolder.h"

using namespace DynamicRank;

LocalFactoryHolder::LocalFactoryHolder() {
  m_ffv2loader = boost::shared_ptr<
      FreeForm2::CompiledNeuralInputLoader<FreeForm2::NeuralInputFreeForm2> >(
      new FreeForm2::CompiledNeuralInputLoader<FreeForm2::NeuralInputFreeForm2>(
          "freeform2"));

  m_inputFactory.AddTransform("freeform2", m_ffv2loader);
}

NeuralInputFactory &LocalFactoryHolder::GetInputFactory() {
  return m_inputFactory;
}

void LocalFactoryHolder::PostLoad(bool useAggregatedCompiler) {
  std::unique_ptr<FreeForm2::Compiler> compiler(
      FreeForm2::CompilerFactory::CreateExecutableCompiler(0));
  m_ffv2loader->Compile(*compiler);
}