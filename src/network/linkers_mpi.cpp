/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifdef USE_MPI

#include "linkers.h"

namespace LightGBM {

Linkers::Linkers(Config) {
  is_init_ = false;
  int argc = 0;
  char**argv = nullptr;
  int flag = 0;
  MPI_SAFE_CALL(MPI_Initialized(&flag));  // test if MPI has been initialized
  if (!flag) {  // if MPI not started, start it
    MPI_SAFE_CALL(MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &flag));
  }
  MPI_SAFE_CALL(MPI_Comm_size(MPI_COMM_WORLD, &num_machines_));
  MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank_));
  // wait for all client start up
  MPI_SAFE_CALL(MPI_Barrier(MPI_COMM_WORLD));
  bruck_map_ = BruckMap::Construct(rank_, num_machines_);
  recursive_halving_map_ = RecursiveHalvingMap::Construct(rank_, num_machines_);
  is_init_ = true;
}

Linkers::~Linkers() {
  if (is_init_) {
    MPI_SAFE_CALL(MPI_Finalize());
  }
}


}  // namespace LightGBM
#endif  // USE_MPI
