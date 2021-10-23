#pragma once
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stddef.h>
#include <iostream>
#include <string>
#include <vector>
#include "Parser.h"
#include "FeatureEvaluator.h"
#include "TransformProcessorFeatureMap.h"
#include <InputExtraction.h>
#include <InputComputation.h>
#include <FeatureMap.h>


using namespace std;


class IniFileParserInterface : public Parser {
 public:
  static IniFileParserInterface* CreateFromInputStr(const string& str);
  static IniFileParserInterface* Createfromfreeform2(string freeform);
  static void C_GetInputFeatures(void* input_ptr, uint32_t* features, uint32_t size_of_features, uint32_t* feature_count);
  static double C_EvaluateInput(void* input_ptr, uint32_t* input1);
  static bool C_IsCopyInput(void* p_input);
  static void* C_GetInput(void* input_extractor_ptr, uint32_t index);
  static uint32_t C_GetInputCount(void* input_extractor_ptr);
  static bool C_GetInputName(void* input_extractor_ptr, uint32_t feature_index, char* buffer, uint32_t size_of_buffer, uint32_t* result_length);
  static bool C_GetFeatureName(void* input_extractor_ptr, uint32_t feature_index, char* buffer, uint32_t size_of_buffer, uint32_t* result_length);
  static uint32_t C_GetFeatureCount(void* input_extractor_ptr);
  static void* GetFeatureMap(void* feature_map_ptr);
  void* GetInput(uint32_t index);
  string StringFetch(bool(*fp)(void* input_extractor_ptr, uint32_t feature_index, char* buffer, uint32_t size_of_buffer, uint32_t* result_length), uint32_t id, void* object_ptr);
  string GetInputName(uint32_t input_id);
  string GetFeatureName(uint32_t input_id);
  void GetInputFeatures(void* input_ptr, uint32_t* features, uint32_t size_of_features, uint32_t* feature_count);
  uint32_t GetInputCount();
  double EvaluateInput(void* input_ptr, uint32_t* input1);
  bool IsCopyInput(void* p_input);
  uint32_t GetFeatureCount();
  ~IniFileParserInterface(void);

 private:
  void* input_extractor_ptr_;
  void* feature_map_ptr_;
  IniFileParserInterface(void* p_input_extractor);   
};
