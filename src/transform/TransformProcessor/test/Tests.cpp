#include "Tests.h"
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <assert.h>
#include <fstream>
#include <string>
#include <LightGBM/utils/log.h>
#include <LightGBM/utils/common.h>

using namespace std;
using namespace LightGBM;

void TestGetFeatureCount(){
  std::unique_ptr<TransformProcessor> transform;
  transform.reset(new TransformProcessor(Common::LoadStringFromFile("./TestData/integration/SmoothedTrainInputIni"),
                                         Common::LoadStringFromFile("./TestData/integration/Header.tsv", 1), 7));
  assert(transform->GetFeatureCount() == 4865);
}

void TestTransformE2E(){
  string test_data_path = "./TestData/integration/";
  string input_data_path = test_data_path + "Input.tsv";
  string input_ini_path = test_data_path + "SmoothedTrainInputIni";
  string input_head_path = test_data_path + "Header.tsv";
  string expected_output_path = test_data_path + "ExpectedOutput.txt";
  string actual_output_path = test_data_path + "Actual.txt";
  int label_id = 7;

  vector<string> transformed_data;
  std::unique_ptr<TransformProcessor> transform;
  transform.reset(new TransformProcessor(Common::LoadStringFromFile(input_ini_path.c_str()), Common::LoadStringFromFile(input_head_path.c_str(), 1), label_id));
  transformed_data = TransformedData::ToString(transform->ApplyForFile(input_data_path));
  ofstream output_fout(actual_output_path.c_str());
  for(auto str : transformed_data)
      output_fout << str << endl;
  output_fout.close();

  ifstream expectin(expected_output_path.c_str());
  ifstream actualin(actual_output_path.c_str());
  string expect_line;
  string actual_line;
  std::getline(expectin, expect_line);
  std::getline(actualin, actual_line);
  if(expect_line.compare(actual_line) != 0) {
    Log::Fatal("Failed to compare expect output and actual output!");
  } else {
    Log::Info("All passed");
  }
}

int main()
{
  TestGetFeatureCount();
  TestTransformE2E();
  return 0;
}