#include <LightGBM/boosting.h>
#include "gbdt.h"

namespace LightGBM {

Boosting* Boosting::CreateBoosting(BoostingType type, const char* filename) {
  if (filename[0] == '\0') {
    if (type == BoostingType::kGBDT) {
      return new GBDT();
    } else {
      return nullptr;
    }
  } else {
    Boosting* ret = nullptr;
    TextReader<size_t> model_reader(filename, true);
    model_reader.ReadAllLines();
    std::string type_infile = model_reader.first_line();
    if (type_infile == std::string("gbdt") && type == BoostingType::kGBDT) {
      ret = new GBDT();
    }
    if (ret != nullptr) {
      std::stringstream str_buf;
      for (auto& line : model_reader.Lines()) {
        str_buf << line << '\n';
      }
      ret->ModelsFromString(str_buf.str());
    }
    return ret;
  }
}

Boosting* Boosting::CreateBoosting(const char* filename) {
  Boosting* ret = nullptr;
  TextReader<size_t> model_reader(filename, true);
  model_reader.ReadAllLines();
  std::string type = model_reader.first_line();
  if (type == std::string("gbdt")) {
    ret = new GBDT();
  }
  if (ret != nullptr) {
    std::stringstream str_buf;
    for (auto& line : model_reader.Lines()) {
      str_buf << line << '\n';
    }
    ret->ModelsFromString(str_buf.str());
  }
  return ret;
}

}  // namespace LightGBM
