#include <LightGBM/boosting.h>
#include "gbdt.h"
#include "dart.hpp"

namespace LightGBM {

std::string GetBoostingTypeFromModelFile(const char* filename) {
  TextReader<size_t> model_reader(filename, true);
  std::string type = model_reader.first_line();
  return type;
}

void Boosting::LoadFileToBoosting(Boosting* boosting, const char* filename) {
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

Boosting* Boosting::CreateBoosting(const std::string& type, const char* filename) {
  if (filename == nullptr || filename[0] == '\0') {
    if (type == std::string("gbdt")) {
      return new GBDT();
    } else if (type == std::string("dart")) {
      return new DART();
    } else {
      return nullptr;
    }
  } else {
    std::unique_ptr<Boosting> ret;
    auto type_in_file = GetBoostingTypeFromModelFile(filename);
    if (type_in_file == std::string("tree")) {
      if (type == std::string("gbdt")) {
        ret.reset(new GBDT());
      } else if (type == std::string("dart")) {
        ret.reset(new DART());
      }
      LoadFileToBoosting(ret.get(), filename);
    } else {
      Log::Fatal("unknow submodel type in model file %s", filename);
    }
    return ret.release();
  }
}

Boosting* Boosting::CreateBoosting(const char* filename) {
  auto type = GetBoostingTypeFromModelFile(filename);
  std::unique_ptr<Boosting> ret;
  if (type == std::string("tree")) {
    ret.reset(new GBDT());
  } else {
    Log::Fatal("unknow submodel type in model file %s", filename);
  }
  LoadFileToBoosting(ret.get(), filename);
  return ret.release();
}

}  // namespace LightGBM
