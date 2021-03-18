// This is a very simple benchmark for comparing performance of Atof and AtofPrecise.

#include <cstdlib>
#include <string>
#include <fstream>
#include <iostream>

#include <io/parser.hpp>

namespace LightGBM {

void ParseCSV(const std::string& fpath, int ncol) {
  CSVParser parser(-1, ncol);

  std::ifstream infile(fpath);
  if (! infile) {
    std::cerr << "fail to open " << fpath;
    std::exit(1);
  }

  std::string line;
  double label;
  std::vector<std::pair<int, double>> oneline_features;
  while (getline(infile, line)) {
    parser.ParseOneLine(line.c_str(), &oneline_features, &label);
//    printf("%f\n", oneline_features[0].second);
    oneline_features.clear();
  }
}

}  // namespace LightGBM

int main(int argc, const char* argv[]) {
  if (argc != 3) {
    printf("usage: parser <fname> <ncol>\n");
    exit(1);
  }

  const char* fpath = argv[1];
  long ncol = strtol(argv[2], nullptr, 10);
  if (errno != 0) {
    fprintf(stderr, "fail to parse ncol\n");
    exit(1);
  }

  LightGBM::ParseCSV(fpath, ncol);

  return 0;
}