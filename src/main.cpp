/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <LightGBM/application.h>

#include <iostream>

#include "network/linkers.h"

int main(int argc, char** argv) {
  bool success = false;
  try {
    // LGBM_CUDA
    std::chrono::duration<double, std::milli> main_time;
    auto start_main_time = std::chrono::steady_clock::now();

    LightGBM::Application app(argc, argv);
    app.Run();

#ifdef USE_MPI
    LightGBM::Linkers::MpiFinalizeIfIsParallel();
#endif

    // LGBM_CUDA
    main_time = std::chrono::steady_clock::now() - start_main_time;
    LightGBM::Log::Info("main::main time: %f sec", main_time * 1e-3);
    success = true;
  }
  catch (const std::exception& ex) {
    std::cerr << "Met Exceptions:" << std::endl;
    std::cerr << ex.what() << std::endl;
  }
  catch (const std::string& ex) {
    std::cerr << "Met Exceptions:" << std::endl;
    std::cerr << ex << std::endl;
  }
  catch (...) {
    std::cerr << "Unknown Exceptions" << std::endl;
  }

  if (!success) {
#ifdef USE_MPI
    LightGBM::Linkers::MpiAbortIfIsParallel();
#endif

    exit(-1);
  }
}
