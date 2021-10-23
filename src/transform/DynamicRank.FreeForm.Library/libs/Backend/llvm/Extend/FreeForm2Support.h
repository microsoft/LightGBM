
#ifndef LLVM_FREEFORM2_SUPPORT_H
#define LLVM_FREEFORM2_SUPPORT_H

#include <llvm/CodeGen/MachineRelocation.h>
#include <vector>

namespace llvm
{
    class ExecutionEngine;

    const std::vector<MachineRelocation>& GetMachineRelocations(const ExecutionEngine* p_engine);
}
#endif
