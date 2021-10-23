#include "FeaSpecConfig.h"
#include <fstream>
#include <regex>
#include <boost/algorithm/string/trim.hpp>
#include <boost/lexical_cast.hpp>
#include <stdlib.h>
#include <LightGBM/utils/log.h>

using namespace std;
using namespace boost;
using namespace DynamicRank;
using namespace LightGBM;


Config* Config::GetRawConfiguration(string str){
    regex sectionPattern("\\[(Input:\\d+)\\]");
    regex linePattern("(Line\\d+)\\=([\\s\\S]*)");
	regex equalPattern("(.*)\\=([\\s\\S]*)");
    smatch result;
    string sectionId;
    map<string, string> sectionContent;
    sectionContent.clear();
    map<string, map<string, string>> config;
    size_t pos = 0;
    string line, delimiter = "\n";
    while ((pos = str.find(delimiter)) != string::npos) {
        line = str.substr(0, pos);
        trim(line);
        if (!line.empty()) {
            if (regex_match(line, result, sectionPattern)) {
                if (!sectionContent.empty()) {
                    config.insert(map<string, map<string, string>>::value_type(sectionId, sectionContent));
                }
                sectionId = result[1];
                sectionContent.clear();
            }
            else if (regex_match(line, result, linePattern)) {
                sectionContent.insert(map<string, string>::value_type(result[1], result[2]));
            }
            else if (regex_match(line, result, equalPattern)) {
                sectionContent.insert(map<string, string>::value_type(result[1], result[2]));
            }
            else {
                Log::Warning("Cannot resolve pattern '%s'. Ignore it.", line.c_str());
            }
        }
        str.erase(0, pos + delimiter.length());
    }
    // Insert last section.
    config.insert(map<string, map<string, string>>::value_type(sectionId, sectionContent));
    return new Config(config);
}


bool Config::DoesSectionExist(char* section) {
    string sectionNameStr(section);
    return _config.count(sectionNameStr) > 0;
}


bool Config::DoesParameterExist(const char* section, const char* parameterName) {
    string sectionNameStr(section);
    string paramNameStr(parameterName);
    return _config[sectionNameStr].count(parameterName) > 0;
}


bool Config::GetStringParameter(const char* section, const char* parameterName, string& value) {
    string sectionNameStr(section);
    string parameterNameStr(parameterName);
    map<string, string> sectionContent = _config[sectionNameStr];
    if (sectionContent.count(parameterName) == 0) {
        return false;
    }
    else {
        value = sectionContent[parameterNameStr];
        return true;
    }
}


bool Config::GetStringParameter(const char* section, const char* parameterName, char* value, size_t valueSize) {
	string sectionNameStr(section);
	string parameterNameStr(parameterName);
	map<string, string> sectionContent = _config[sectionNameStr];
	if (sectionContent.count(parameterName) == 0) {
		return false;
	}
	else {
		string valueStr = sectionContent[parameterNameStr];
		if (valueStr.length() > valueSize)
			return false;
		else {
			strcpy(value, valueStr.data());
			return true;
		}
		
	}
}


bool Config::GetDoubleParameter(const char* section, const char* parameterName, double* value) {
    string sectionNameStr(section);
    string parameterNameStr(parameterName);
    map<string, string> sectionContent = _config[sectionNameStr];
    if (sectionContent.count(parameterName) == 0) {
        return false;
    }
    else {
        *value = atof(sectionContent[parameterNameStr].c_str());
        return true;
    }
}


double Config::GetDoubleParameter(const char* section, const char* parameterName, double defaultValue) {
	double value;
	if (Config::GetDoubleParameter(section, parameterName, &value))
		return value;
	else
		return defaultValue;
}


bool Config::GetBoolParameter(const char* section, const char* parameterName, bool* value) {
    string sectionNameStr(section);
    string parameterNameStr(parameterName);
    map<string, string> sectionContent = _config[sectionNameStr];
    if (sectionContent.count(parameterName) == 0) {
        return false;
    }
    else {
        *value = boost::lexical_cast<bool>(sectionContent[parameterNameStr]);
        return true;
    }
}

bool Config::GetBoolParameter(const char* section, const char* parameterName, bool defaultValue) {
	bool value;
	if (Config::GetBoolParameter(section, parameterName, &value))
		return value;
	else
		return defaultValue;
}
