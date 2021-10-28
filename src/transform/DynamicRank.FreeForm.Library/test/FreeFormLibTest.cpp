/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include <LightGBM/utils/log.h>

#include <iostream>

#include "FreeFormLibTestSet.h"

using namespace LightGBM;

int main() {
  Log::Info("-------- FreeForm Library Test Starts --------");
  FreeFormLibTestSet::TestParser("(if (== Foo Bar) 1 0)");
  FreeFormLibTestSet::TestParser(
      "(* (ln1 NumberOfCompleteMatches_IETBSatModel-IM-Prod) "
      "OriginalQueryMaxNumberOfPerfectMatches_BingClicks-Prod)");
  FreeFormLibTestSet::TestNeuralInputLoadSave();
  Log::Info("-------- FreeForm Library Test Finished --------");
  return 0;
}
