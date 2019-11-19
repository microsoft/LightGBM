/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <LightGBM/application.h>

#include <iostream>

int main(int argc, char** argv) {
  try {
    LightGBM::Application app(argc, argv);

    std::string exception_str;
    try
    {
      app.Run();
    }
    catch (const std::exception& ex) {
      exception_str = ex.what();
    }
    catch (const std::string& ex) {
      exception_str = ex;
    }
    catch (...) {
      exception_str = "Unknown Exceptions";
    }

    if (!exception_str.empty())
    {
#ifdef USE_MPI
      // In the MPI mode, if there is an exception and you try to exit this block, the destructor for "app" is called,
      // which in turn calls MPI_Finalize(). But that causes this process to hang and the whole MPI job can hang as a result.
      // Instead we may want to call MPI_Abort(). However that right now doesn't do much more call abort(). So we just do it here.
      if (app.IsParallel())
      {
        std::cerr << "Aborting because of exception in MPI mode:" << std::endl;
        std::cerr << exception_str << std::endl;
        std::cerr.flush();
        abort();
      }
#endif
      throw exception_str;
    }
  }
  catch (const std::exception& ex) {
    std::cerr << "Met Exceptions:" << std::endl;
    std::cerr << ex.what() << std::endl;
    exit(-1);
  }
  catch (const std::string& ex) {
    std::cerr << "Met Exceptions:" << std::endl;
    std::cerr << ex << std::endl;
    exit(-1);
  }
  catch (...) {
    std::cerr << "Unknown Exceptions" << std::endl;
    exit(-1);
  }
}
