#include <LightGBM/boosting.h>
#include "gbdt.h"

namespace LightGBM {

Boosting* Boosting::CreateBoosting(BoostingType type,
                         const BoostingConfig* config) {
  if (type == BoostingType::kGBDT) {
    return new GBDT(config);
  } else {
    return nullptr;
  }
}

}  // namespace LightGBM
