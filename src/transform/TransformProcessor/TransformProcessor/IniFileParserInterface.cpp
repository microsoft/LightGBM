#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <iostream>
#include <vector>
#include <LightGBM/utils/log.h>
#include "IniFileParserInterface.h"

using namespace std;
using namespace LightGBM;

IniFileParserInterface* IniFileParserInterface::Createfromfreeform2(string freeform) {
  void* from_freeform2;
  try {
    from_freeform2 = InputExtractorCreateFromFreeformV2(freeform.c_str());
  } catch (const char* e) {
    Log::Fatal("Something wrong when loading Createfromfreeform2, detail:  %s", e);
  }

  return new IniFileParserInterface(from_freeform2);
}

IniFileParserInterface* IniFileParserInterface::CreateFromInputStr(const string& str) {
  void* from_inputini;
  try {
    from_inputini = InputExtractorCreateFromInputStr(str);
  } catch (const char* e) {
    Log::Fatal("Something wrong when loading CreateFromInputStr, detail:  %s", e);
  }

  return new IniFileParserInterface(from_inputini);
}

void IniFileParserInterface::C_GetInputFeatures(void* input_ptr, uint32_t* features, uint32_t size_of_features, uint32_t* feature_count) {
  try {
    InputGetFeatures(input_ptr, features, size_of_features, feature_count);
  } catch (const char* e) {
    Log::Fatal("Something wrong when loading GetInputFeatures, detail:  %s", e);
  }
}

double IniFileParserInterface::C_EvaluateInput(void* input_ptr, uint32_t* input1) {
  double res;
  try {
    res = InputEvaluate(input_ptr, input1);
  } catch (const char* e) {
    Log::Fatal("Something wrong when loading EvaluateInput, detail:  %s", e);
  }

  return res;
}

bool IniFileParserInterface::C_IsCopyInput(void* p_input) {
  bool res;
  try {
    res = InputIsCopy(p_input);
  } catch (const char* e) {
    Log::Fatal("Something wrong when loading IsCopyInput, detail:  %s", e);
  }

  return res;
}

void* IniFileParserInterface::C_GetInput(void* input_extractor_ptr, uint32_t index) {
  void* res;
  try{
    res = InputExtractorGetInput(input_extractor_ptr, index);
  } catch (const char* e) {
    Log::Fatal("Something wrong when loading GetInput, detail:  %s", e);
  }

  return res;
}

void* IniFileParserInterface::GetFeatureMap(void* feature_map_ptr) {
  void* res;
  try {
    res = InputExtractorGetFeatureMap(feature_map_ptr);
  } catch (const char* e) {
    Log::Fatal("Something wrong when loading GetFeatureMap, detail:  %s", e);
  }

  return res;
}

uint32_t IniFileParserInterface::C_GetInputCount(void* input_extractor_ptr) {
  uint32_t res;
  try {
    res = InputExtractorGetInputCount(input_extractor_ptr);
  } catch (const char* e) {
    Log::Fatal("Something wrong when loading GetInputCount, detail:  %s", e);
  }

  return res;
}

bool IniFileParserInterface::C_GetInputName(void* input_extractor_ptr, uint32_t feature_index, char* buffer, uint32_t size_of_buffer, uint32_t* result_length) {
  bool res;
  try {
    res = InputExtractorGetInputName(input_extractor_ptr, feature_index, buffer, size_of_buffer, result_length);
  } catch (const char* e) {
    Log::Fatal("Something wrong when loading GetInputName, detail:  %s", e);
  }

  return res;
}

bool IniFileParserInterface::C_GetFeatureName(void* input_extractor_ptr, uint32_t feature_index, char* buffer, uint32_t size_of_buffer, uint32_t* result_length) {
  bool res;
  try {
    res = FeatureMapGetFeatureName(input_extractor_ptr, feature_index, buffer, size_of_buffer, result_length);
  } catch (const char* e) {
    Log::Fatal("Something wrong when loading GetFeatureName, detail:  %s", e);
  }

  return res;
}

uint32_t IniFileParserInterface::C_GetFeatureCount(void* input_extractor_ptr) {
  uint32_t res;
  try {
    res = FeatureMapGetFeatureCount(input_extractor_ptr);
  } catch (const char* e) {
    Log::Fatal("Something wrong when loading GetFeatureCount, detail:  %s", e);
  }

  return res;
}

void IniFileParserInterface::GetInputFeatures(void* input_ptr, uint32_t* features, uint32_t size_of_features, uint32_t* feature_count) {
  return IniFileParserInterface::C_GetInputFeatures(input_ptr, features, size_of_features, feature_count);
}

bool IniFileParserInterface::IsCopyInput(void* p_input) {
  return IniFileParserInterface::C_IsCopyInput(p_input);
}

void* IniFileParserInterface::GetInput(uint32_t index) {
  return C_GetInput(this->input_extractor_ptr_, index);
}

uint32_t IniFileParserInterface::GetInputCount() {
  return C_GetInputCount(this->input_extractor_ptr_);
}

uint32_t IniFileParserInterface::GetFeatureCount() {
  return C_GetFeatureCount(this->feature_map_ptr_);
}

double IniFileParserInterface::EvaluateInput(void* input_ptr, uint32_t* input1) {
  return IniFileParserInterface::C_EvaluateInput(input_ptr, input1);
}

string IniFileParserInterface::GetInputName(uint32_t input_id) {
  return this->StringFetch(&IniFileParserInterface::C_GetInputName, (uint32_t)input_id, this->input_extractor_ptr_);
}

string IniFileParserInterface::GetFeatureName(uint32_t input_id) {
  return this->StringFetch(&IniFileParserInterface::C_GetFeatureName, (uint32_t)input_id, this->feature_map_ptr_);
}

string IniFileParserInterface::StringFetch(
  bool(*fetcher)(void* input_extractor_ptr, uint32_t feature_index, char* buffer, uint32_t size_of_buffer, uint32_t* result_length),
  uint32_t id,
  void* object_ptr) {
  std::vector<unsigned char> array(100);
  uint32_t num = 0;
  uint32_t* num_ptr = &num;

  if (!(*fetcher)(object_ptr, id, reinterpret_cast<char*>(&array[0]), (uint32_t)array.size(), (uint32_t*)num_ptr)) {
	if ((long)num <= (long)array.size()) {
	  return NULL;
	}
	array.resize((int)num);
	if (!(*fetcher)(object_ptr, id, reinterpret_cast<char*>(&array[0]), (uint32_t)array.size(), (uint32_t*)num_ptr)) {
	  return NULL;
	}
  }

  std::string string1(reinterpret_cast<const char*>(&array[0]), num);
  return string1;
}

IniFileParserInterface::IniFileParserInterface(void* p_input_extractor) {
  input_extractor_ptr_ = p_input_extractor;
  feature_map_ptr_ = this->IniFileParserInterface::GetFeatureMap(input_extractor_ptr_);
}
