#include <iostream>
#include "LightGBM/c_api.h"

using namespace std;

int main() {
  cout << "start\n";

  BoosterHandle boosterHandle;
  int num_iterations;
  LGBM_BoosterCreateFromModelfile("./LightGBM_model.txt", &num_iterations, &boosterHandle);
  cout << "Model iterations " << num_iterations<< "\n";

  double values[] = {1.000000000000000000e+00,8.692932128906250000e-01,-6.350818276405334473e-01,2.256902605295181274e-01,3.274700641632080078e-01,-6.899932026863098145e-01,7.542022466659545898e-01,-2.485731393098831177e-01,-1.092063903808593750e+00,0.000000000000000000e+00,1.374992132186889648e+00,-6.536741852760314941e-01,9.303491115570068359e-01,1.107436060905456543e+00,1.138904333114624023e+00,-1.578198313713073730e+00,-1.046985387802124023e+00,0.000000000000000000e+00,6.579295396804809570e-01,-1.045456994324922562e-02,-4.576716944575309753e-02,3.101961374282836914e+00,1.353760004043579102e+00,9.795631170272827148e-01,9.780761599540710449e-01,9.200048446655273438e-01,7.216574549674987793e-01,9.887509346008300781e-01,8.766783475875854492e-01}; // score = 0.487278

  FastConfigHandle fastConfigHandle;
  LGBM_BoosterPredictForMatSingleRowFastInit(boosterHandle, C_API_DTYPE_FLOAT64, 28, "", &fastConfigHandle);

  int64_t dummy_out_len;
  double score[1];
  for (size_t i = 0; i < 3e5; ++i) {
    LGBM_BoosterPredictForMatSingleRowFast(fastConfigHandle, values, C_API_PREDICT_NORMAL, num_iterations, &dummy_out_len, score);
  }

  LGBM_FastConfigFree(fastConfigHandle);

  cout << "len=" << dummy_out_len << endl;

  cout << "Score = " << score[0] << "\n";

  cout << "end\n";
}
