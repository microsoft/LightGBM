#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <iostream>
#include <string>
#include <vector>

using namespace std;


class Parser {
 public:
  virtual void* GetInput(uint32_t index) = 0;
  virtual string GetInputName(uint32_t input_id) = 0;
  virtual string GetFeatureName(uint32_t input_id) = 0;
  virtual uint32_t GetInputCount() = 0;
  virtual uint32_t GetFeatureCount() = 0;
  virtual void GetInputFeatures(void* input_ptr, uint32_t* features, uint32_t size_of_features, uint32_t* feature_count) = 0;
  virtual bool IsCopyInput(void* p_input) = 0;
  virtual double EvaluateInput(void* input_ptr, uint32_t* input1) = 0;
};
