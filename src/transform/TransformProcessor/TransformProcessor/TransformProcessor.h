#pragma once
#include <iostream>
#include <vector>
#include <string>
#include "IniFileParserInterface.h"
#include "TransformProcessorFeatureMap.h"
#include "FeatureEvaluator.h"
#include "FeatureEvaluatorExtendedInfo.h"

using namespace std;
using std::string;

// Object-oriented packaging of TransformProcessor
class TransformedData
{
private:
	double _label;
	vector<pair<int, double> > _sparse_features;
public:
	TransformedData(double label, vector<pair<int, double> > sparse_features):
		_label(label), _sparse_features(sparse_features){}
	double Label() {return this->_label;};
	vector<pair<int, double> > Features() {return this->_sparse_features;};
	string ToString(int precision=5);
	static vector<string> ToString(vector<TransformedData> dataset, int precision=5);
};

class TransformProcessor
{
private:
	int _label_id;
	std::map<string, int> _column_index_mapping;
	vector<FeatureEvaluatorExtendedInfo> _feature_evaluator_list;
	unsigned int _feature_index_start;
	IniFileParserInterface* _from_input_str = nullptr;
	static bool CloseToZero(double val);
	static double ConvertStrToDouble(string val);
	static const double EPS;
public:
	TransformProcessor(const string& transform_str, const std::string& header_str, int label_id);
	void Parse(const char* str, vector<string>* out_feature_strs, double* out_label, string delimiter="\t");
	void Apply(vector<string>* input_row, vector<pair<int, double>>* out_features);
	vector<TransformedData> ApplyForFile(string data_path);
	int GetFeatureCount();
};
