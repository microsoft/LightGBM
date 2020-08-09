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
    //LightGBM::Application app(argc, argv);



    // code from: https://stackoverflow.com/a/43373070
    std::vector<char*> argv_new(argv, argv + argc);
    argv_new.push_back("config=c:\\cpp_code\\lgb_pw\\refit.conf");
    argv_new.push_back(nullptr);
    LightGBM::Application app(argc + 1, &argv_new[0]);


    app.Run();

#ifdef USE_MPI
    LightGBM::Linkers::MpiFinalizeIfIsParallel();
#endif

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
