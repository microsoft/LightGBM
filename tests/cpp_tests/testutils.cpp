/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <string>

#include <testutils.h>
#include <LightGBM/c_api.h>

namespace LightGBM {

/*!
* Creates a Dataset from the internal repository examples.
*/
int TestUtils::LoadDatasetFromExamples(const char* filename, const char* config, DatasetHandle *out) {
  std::string fullPath("..\\examples\\");
  fullPath += filename;
  // TODO check file exists
  return LGBM_DatasetCreateFromFile(
    fullPath.c_str(),
    config,
    nullptr,
    out);

    /*print(LIB.LGBM_GetLastError())
    num_data = ctypes.c_int(0)
    LIB.LGBM_DatasetGetNumData(handle, ctypes.byref(num_data))
    num_feature = ctypes.c_int(0)
    LIB.LGBM_DatasetGetNumFeature(handle, ctypes.byref(num_feature))
    print(f'#data: {num_data.value} #feature: {num_feature.value}')
    return handle*/
}
}

