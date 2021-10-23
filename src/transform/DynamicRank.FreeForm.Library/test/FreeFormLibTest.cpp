#include <iostream>
#include "FreeFormLibTestSet.h"
#include <LightGBM/utils/log.h>

using namespace LightGBM;

int main()
{
    Log::Info("-------- FreeForm Library Test Starts --------");
    FreeFormLibTestSet::TestParser("(if (== Foo Bar) 1 0)");
    FreeFormLibTestSet::TestParser("(* (ln1 NumberOfCompleteMatches_IETBSatModel-IM-Prod) OriginalQueryMaxNumberOfPerfectMatches_BingClicks-Prod)");
    FreeFormLibTestSet::TestNeuralInputLoadSave();
    Log::Info("-------- FreeForm Library Test Finished --------");
    return 0;
}
