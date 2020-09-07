/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <LightGBM/application.h>

#include <iostream>
#include <string>

#include "network/linkers.h"

int main(/*int argc, char** argv*/) {
  bool success = false;
  try {
    std::string conf_path_string = std::string("config=src/train.conf");
    char** argv = new char*[2];
    argv[0] = nullptr;
    char* conf_path = new char[conf_path_string.size() + 1];
    for (size_t i = 0; i < conf_path_string.size(); ++i) {
      conf_path[i] = conf_path_string[i];
    }
    conf_path[conf_path_string.size()] = '\0';
    argv[1] = conf_path;
    LightGBM::Application app(2, argv);
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
