/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_TESTS_UTILS_H_
#define LIGHTGBM_TESTS_UTILS_H_

#include <LightGBM/c_api.h>

namespace LightGBM {

class TestUtils {
  public:

    /*!
    * Creates a Dataset from the internal repository examples.
    */
    static int LoadDatasetFromExamples(const char* filename, const char* config, DatasetHandle *out);
};
}

#endif  // LIGHTGBM_TESTS_UTILS_H_

