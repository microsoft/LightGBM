/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#pragma once

#include <iostream>
#include <map>
#include <string>
#include <vector>
using namespace std;

namespace DynamicRank {

class Config {
 private:
  string _path;
  map<string, map<string, string> > _config;
  Config(map<string, map<string, string> > &config) { _config = config; };

 public:
  static Config *GetRawConfiguration(string str);
  bool DoesSectionExist(char *section);
  bool DoesParameterExist(const char *section, const char *parameterName);
  bool GetStringParameter(const char *section, const char *parameterName,
                          string &value);
  bool GetStringParameter(const char *section, const char *parameterName,
                          char *value, size_t valueSize);
  bool GetDoubleParameter(const char *section, const char *parameterName,
                          double *value);
  double GetDoubleParameter(const char *section, const char *parameterName,
                            double defaultValue);
  bool GetBoolParameter(const char *section, const char *parameterName,
                        bool *value);
  bool GetBoolParameter(const char *section, const char *parameterName,
                        bool defaultValue);
};
}  // namespace DynamicRank