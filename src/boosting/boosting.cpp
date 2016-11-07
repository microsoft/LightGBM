#include <LightGBM/boosting.h>
#include "gbdt.h"

namespace LightGBM {

BoostingType GetBoostingTypeFromModelFile(const char* filename) {
  TextReader<size_t> model_reader(filename, true);
  std::string type = model_reader.first_line();
  if (type == std::string("gbdt")) {
    return BoostingType::kGBDT;
  }
  return BoostingType::kUnknow;
}

void LoadFileToBoosting(Boosting* boosting, const char* filename) {
  if (boosting != nullptr) {
    TextReader<size_t> model_reader(filename, true);
    model_reader.ReadAllLines();
    std::stringstream str_buf;
    for (auto& line : model_reader.Lines()) {
      str_buf << line << '\n';
    }
    boosting->LoadModelFromString(str_buf.str());
  }
}

Boosting* Boosting::CreateBoosting(BoostingType type, const char* filename) {
  if (filename == nullptr || filename[0] == '\0') {
    if (type == BoostingType::kGBDT) {
      return new GBDT();
    } else {
      return nullptr;
    }
  } else {
    Boosting* ret = nullptr;
    auto type_in_file = GetBoostingTypeFromModelFile(filename);
    if (type_in_file == type) {
      if (type == BoostingType::kGBDT) {
        ret = new GBDT();
      }
      LoadFileToBoosting(ret, filename);
    } else {
      Log::Fatal("Boosting type in parameter is not the same as the type in the model file");
    }
    return ret;
  }
}

Boosting* Boosting::CreateBoosting(const char* filename) {
  auto type = GetBoostingTypeFromModelFile(filename);
  Boosting* ret = nullptr;
  if (type == BoostingType::kGBDT) {
    ret = new GBDT();
  }
  LoadFileToBoosting(ret, filename);
  return ret;
}

}  // namespace LightGBM
