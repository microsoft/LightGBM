#include <LightGBM/boosting.h>
#include "gbdt.h"
#include "dart.hpp"
#include "goss.hpp"
#include "rf.hpp"

namespace LightGBM {

std::string GetBoostingTypeFromModelFile(const char* filename) {
  TextReader<size_t> model_reader(filename, true);
  std::string type = model_reader.first_line();
  return type;
}

bool Boosting::LoadFileToBoosting(Boosting* boosting, const std::string& format, const char* filename) {
  if (boosting != nullptr) {
    if (format == std::string("text")) {
      TextReader<size_t> model_reader(filename, true);
      model_reader.ReadAllLines();
      std::stringstream str_buf;
      for (auto& line : model_reader.Lines()) {
        str_buf << line << '\n';
      }
      if (!boosting->LoadModelFromString(str_buf.str())) {
        return false;
      }
    } else if (format == std::string("proto")) {
      if (!boosting->LoadModelFromProto(filename)) {
        return false;
      }
    } else {
      Log::Fatal("Unknown model format during loading: %s", format.c_str());
    }
  }
  return true;
}

Boosting* Boosting::CreateBoosting(const std::string& type, const std::string& format, const char* filename) {
  if (filename == nullptr || filename[0] == '\0') {
    if (type == std::string("gbdt")) {
      return new GBDT();
    } else if (type == std::string("dart")) {
      return new DART();
    } else if (type == std::string("goss")) {
      return new GOSS();
    } else if (type == std::string("rf")) {
      return new RF();
    } else {
      return nullptr;
    }
  } else {
    std::unique_ptr<Boosting> ret;
    if (format == std::string("proto") || GetBoostingTypeFromModelFile(filename) == std::string("tree")) {
      if (type == std::string("gbdt")) {
        ret.reset(new GBDT());
      } else if (type == std::string("dart")) {
        ret.reset(new DART());
      } else if (type == std::string("goss")) {
        ret.reset(new GOSS());
      } else if (type == std::string("rf")) {
        return new RF();
      } else {
        Log::Fatal("unknown boosting type %s", type.c_str());
      }
      LoadFileToBoosting(ret.get(), format, filename);
    } else {
      Log::Fatal("unknown model format or submodel type in model file %s", filename);
    }
    return ret.release();
  }
}

}  // namespace LightGBM
