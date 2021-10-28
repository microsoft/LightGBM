/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include "FreeForm2Utils.h"

#include "FreeForm2Assert.h"
// #include <Logging.h>
#include <memory.h>

#include <iostream>
#include <sstream>
// #include <windows.h>

void FreeForm2::VectorFromNeuralNetFeatures::ProcessFeature(
    UInt32 p_featureIndex) {
  m_associatedFeaturesList.push_back(p_featureIndex);
}

void FreeForm2::VectorFromNeuralNetFeatures::ProcessFeature(
    UInt32 p_featureIndex, const std::vector<std::string> &p_segments) {
  // Ignore segment information.
  ProcessFeature(p_featureIndex);
}

void FreeForm2::SetFromNeuralNetFeatures::ProcessFeature(
    UInt32 p_featureIndex) {
  m_associatedFeaturesList.insert(p_featureIndex);
}

void FreeForm2::SetFromNeuralNetFeatures::ProcessFeature(
    UInt32 p_featureIndex, const std::vector<std::string> &p_segments) {
  // Ignore segment information.
  ProcessFeature(p_featureIndex);
}

std::ostream &FreeForm2::operator<<(std::ostream &p_out, SIZED_STRING p_str) {
  return p_out.write(p_str.pcData, static_cast<std::streamsize>(p_str.cbData));
}

void FreeForm2::LogHardwareException(DWORD p_exceptionCode,
                                     const Executable::FeatureType p_features[],
                                     const DynamicRank::IFeatureMap &p_map,
                                     const char *p_sourceFile,
                                     unsigned int p_sourceLine) {
  // Blech, windows programming.  Use FormatMessage and some LoadLibrary
  // trickery to get windows to format our exception code into text.
}

bool FreeForm2::IsSimpleName(SIZED_STRING p_name) {
  for (unsigned int i = 0; i < p_name.cbData; i++) {
    if (!isalnum(p_name.pbData[i]) && p_name.pcData[i] != '-' &&
        p_name.pcData[i] != '_') {
      return false;
    }
  }

  return true;
}

void FreeForm2::WriteCompressedVectorRLE(const UInt32 *p_data,
                                         size_t p_numElements,
                                         std::ostream &p_out) {
  FF2_ASSERT(p_data != NULL);
  enum { MatchingNulls, MatchingNonNulls } state = MatchingNonNulls;
  UInt32 numNulls = 0;
  const UInt32 null = 0;
  for (size_t i = 0; i < p_numElements; i++) {
    switch (state) {
      case MatchingNulls: {
        if (p_data[i] == null) {
          FF2_ASSERT(numNulls < MAX_UINT32);
          numNulls++;
        } else {
          state = MatchingNonNulls;
          p_out.write(reinterpret_cast<const char *>(&numNulls),
                      sizeof(UInt32));
          p_out.write(reinterpret_cast<const char *>(&p_data[i]),
                      sizeof(UInt32));
        }
        break;
      }

      case MatchingNonNulls: {
        if (p_data[i] == null) {
          state = MatchingNulls;
          p_out.write(reinterpret_cast<const char *>(&null), sizeof(UInt32));
          numNulls = 1;
        } else {
          p_out.write(reinterpret_cast<const char *>(&p_data[i]),
                      sizeof(UInt32));
        }
        break;
      }
    }
  }
}

void FreeForm2::ReadCompressedVectorRLE(UInt32 *p_data, size_t p_numElements,
                                        std::istream &p_in) {
  FF2_ASSERT(p_data != NULL);
  size_t i = 0;
  while (i < p_numElements) {
    UInt32 value = 0;
    p_in.read(reinterpret_cast<char *>(&value), sizeof(UInt32));
    if (value == 0) {
      p_in.read(reinterpret_cast<char *>(&value), sizeof(UInt32));
      memset(&p_data[i], 0, sizeof(UInt32) * value);
      i += value;
    } else {
      p_data[i] = value;
      i++;
    }
  }
}
