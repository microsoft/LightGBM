/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#pragma once

#include "FreeForm2.h"
#include "NeuralInputFreeForm2.h"

// Several different areas may require access to an input and ensemble factory.
class LocalFactoryHolder
{
private:
    DynamicRank::NeuralInputFactory m_inputFactory;
    boost::shared_ptr<FreeForm2::CompiledNeuralInputLoader<FreeForm2::NeuralInputFreeForm2> > m_ffv2loader;

public:
    LocalFactoryHolder();
    DynamicRank::NeuralInputFactory &GetInputFactory();
    // Call this function after loading has finished, but
    // evaluation has not yet begun.
    void PostLoad(bool useAggregatedCompiler);
};
