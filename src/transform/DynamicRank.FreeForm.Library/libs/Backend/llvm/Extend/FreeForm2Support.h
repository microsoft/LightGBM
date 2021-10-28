/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LLVM_FREEFORM2_SUPPORT_H
#define LLVM_FREEFORM2_SUPPORT_H

#include <llvm/CodeGen/MachineRelocation.h>

#include <vector>

namespace llvm {
class ExecutionEngine;

const std::vector<MachineRelocation> &GetMachineRelocations(
    const ExecutionEngine *p_engine);
}  // namespace llvm
#endif
