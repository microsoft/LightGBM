#include <llvm/CodeGen/JITCodeEmitter.h>
#include "FreeForm2Support.h"
#include "JITExtend.h"

namespace llvm
{
    struct FakeJITEmitter : public llvm::JITCodeEmitter {
        void *MemMgr;

        // When outputting a function stub in the context of some other function, we
        // save BufferBegin/BufferEnd/CurBufferPtr here.
        uint8_t *SavedBufferBegin, *SavedBufferEnd, *SavedCurBufferPtr;

        // When reattempting to JIT a function after running out of space, we store
        // the estimated size of the function we're trying to JIT here, so we can
        // ask the memory manager for at least this much space.  When we
        // successfully emit the function, we reset this back to zero.
        uintptr_t SizeEstimate;

        /// Relocations - These are the relocations that the function needs, as
        /// emitted.
        std::vector<llvm::MachineRelocation> m_relocations;
    };

    const std::vector<MachineRelocation>& GetMachineRelocations(const ExecutionEngine* p_engine)
    {
        const JIT* jit = static_cast<const JIT*>(p_engine);
        return GetJitMachineRelocations(jit);
    }
}
