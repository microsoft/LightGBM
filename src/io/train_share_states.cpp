/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#include <LightGBM/train_share_states.h>

namespace LightGBM {
  TrainingShareStates* TrainingShareStates::CreateTrainingShareStates(bool single_precision_hist_buffer) {
    if (single_precision_hist_buffer) {
        return new TrainingShareStatesFloatWithBuffer();
    } else {
        return new TrainingShareStatesDouble();
    }
  }
}  // namespace LightGBM