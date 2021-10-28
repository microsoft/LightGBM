/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
//= == --JIT.cpp - LLVM Just in Time Compiler-- -- -- -- -- -- -- -- -- -- -- --
//-- -- -- == = //
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tool implements a just-in-time compiler for LLVM, allowing direct
// execution of LLVM bitcode in an efficient manner.
//
//===----------------------------------------------------------------------===//

#include "JITExtend.h"

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/CodeGen/JITCodeEmitter.h>
#include <llvm/CodeGen/MachineCodeInfo.h>
// #include <llvm/Config/config.h>
#include <llvm/ExecutionEngine/GenericValue.h>
#include <llvm/ExecutionEngine/JITEventListener.h>
#include <llvm/ExecutionEngine/JITMemoryManager.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Dwarf.h>
#include <llvm/Support/DynamicLibrary.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/ManagedStatic.h>
#include <llvm/Support/MutexGuard.h>
#include <llvm/Target/TargetJITInfo.h>
#include <llvm/Target/TargetMachine.h>

using namespace llvm;

#ifdef __APPLE__
// Apple gcc defaults to -fuse-cxa-atexit (i.e. calls __cxa_atexit instead
// of atexit). It passes the address of linker generated symbol __dso_handle
// to the function.
// This configuration change happened at version 5330.
#include <AvailabilityMacros.h>
#if defined(MAC_OS_X_VERSION_10_4) &&                           \
    ((MAC_OS_X_VERSION_MIN_REQUIRED > MAC_OS_X_VERSION_10_4) || \
     (MAC_OS_X_VERSION_MIN_REQUIRED == MAC_OS_X_VERSION_10_4 && \
      __APPLE_CC__ >= 5330))
#ifndef HAVE___DSO_HANDLE
#define HAVE___DSO_HANDLE 1
#endif
#endif
#endif

#if HAVE___DSO_HANDLE
extern void *__dso_handle __attribute__((__visibility__("hidden")));
#endif

namespace {

static struct RegisterJIT {
  RegisterJIT() { JIT::Register(); }
} JITRegistrator;

}  // namespace

extern "C" void LLVMLinkInJIT() {}

/// createJIT - This is the factory method for creating a JIT for the current
/// machine, it does not fall back to the interpreter.  This takes ownership
/// of the module.
ExecutionEngine *JIT::createJIT(Module *M, std::string *ErrorStr,
                                JITMemoryManager *JMM, bool GVsWithCode,
                                TargetMachine *TM) {
  // Try to register the program as a source of symbols to resolve against.
  //
  // FIXME: Don't do this here.
  sys::DynamicLibrary::LoadLibraryPermanently(nullptr, nullptr);

  // If the target supports JIT code generation, create the JIT.
  if (TargetJITInfo *TJ = TM->getJITInfo()) {
    return new JIT(M, *TM, *TJ, JMM, GVsWithCode);
  } else {
    if (ErrorStr) *ErrorStr = "target does not support JIT code generation";
    return nullptr;
  }
}

namespace {
/// This class supports the global getPointerToNamedFunction(), which allows
/// bugpoint or gdb users to search for a function by name without any context.
class JitPool {
  SmallPtrSet<JIT *, 1> JITs;  // Optimize for process containing just 1 JIT.
  mutable sys::Mutex Lock;

 public:
  void Add(JIT *jit) {
    MutexGuard guard(Lock);
    JITs.insert(jit);
  }
  void Remove(JIT *jit) {
    MutexGuard guard(Lock);
    JITs.erase(jit);
  }
  void *getPointerToNamedFunction(const char *Name) const {
    MutexGuard guard(Lock);
    assert(JITs.size() != 0 && "No Jit registered");
    // search function in every instance of JIT
    for (SmallPtrSet<JIT *, 1>::const_iterator Jit = JITs.begin(),
                                               end = JITs.end();
         Jit != end; ++Jit) {
      if (Function *F = (*Jit)->FindFunctionNamed(Name))
        return (*Jit)->getPointerToFunction(F);
    }
    // The function is not available : fallback on the first created (will
    // search in symbol of the current program/library)
    return (*JITs.begin())->getPointerToNamedFunction(Name);
  }
};
ManagedStatic<JitPool> AllJits;
}  // namespace
extern "C" {
// getPointerToNamedFunction - This function is used as a global wrapper to
// JIT::getPointerToNamedFunction for the purpose of resolving symbols when
// bugpoint is debugging the JIT. In that scenario, we are loading an .so and
// need to resolve function(s) that are being mis-codegenerated, so we need to
// resolve their addresses at runtime, and this is the way to do it.
void *getPointerToNamedFunction(const char *Name) {
  return AllJits->getPointerToNamedFunction(Name);
}
}

JIT::JIT(Module *M, TargetMachine &tm, TargetJITInfo &tji,
         JITMemoryManager *jmm, bool GVsWithCode)
    : ExecutionEngine(M),
      TM(tm),
      TJI(tji),
      JMM(jmm ? jmm : JITMemoryManager::CreateDefaultMemManager()),
      AllocateGVsWithCode(GVsWithCode),
      isAlreadyCodeGenerating(false) {
  setDataLayout(TM.getDataLayout());

  jitstate = new JITState(M);

  // Initialize JCE
  JCE = createEmitter(*this, JMM, TM);

  // Register in global list of all JITs.
  AllJits->Add(this);

  // Add target data
  MutexGuard locked(lock);
  FunctionPassManager &PM = jitstate->getPM();
  M->setDataLayout(TM.getDataLayout());
  PM.add(new DataLayoutPass(M));

  // Turn the machine code intermediate representation into bytes in memory that
  // may be executed.
  if (TM.addPassesToEmitMachineCode(PM, *JCE, !getVerifyModules())) {
    report_fatal_error("Target does not support machine code emission!");
  }

  // Initialize passes.
  PM.doInitialization();
}

JIT::~JIT() {
  // Cleanup.
  AllJits->Remove(this);
  delete jitstate;
  delete JCE;
  // JMM is a ownership of JCE, so we no need delete JMM here.
  delete &TM;
}

/// addModule - Add a new Module to the JIT.  If we previously removed the last
/// Module, we need re-initialize jitstate with a valid Module.
void JIT::addModule(Module *M) {
  MutexGuard locked(lock);

  if (Modules.empty()) {
    assert(!jitstate && "jitstate should be NULL if Modules vector is empty!");

    jitstate = new JITState(M);

    FunctionPassManager &PM = jitstate->getPM();
    M->setDataLayout(TM.getDataLayout());
    PM.add(new DataLayoutPass(M));

    // Turn the machine code intermediate representation into bytes in memory
    // that may be executed.
    if (TM.addPassesToEmitMachineCode(PM, *JCE, !getVerifyModules())) {
      report_fatal_error("Target does not support machine code emission!");
    }

    // Initialize passes.
    PM.doInitialization();
  }

  ExecutionEngine::addModule(M);
}

/// removeModule - If we are removing the last Module, invalidate the jitstate
/// since the PassManager it contains references a released Module.
bool JIT::removeModule(Module *M) {
  bool result = ExecutionEngine::removeModule(M);

  MutexGuard locked(lock);

  if (jitstate && jitstate->getModule() == M) {
    delete jitstate;
    jitstate = nullptr;
  }

  if (!jitstate && !Modules.empty()) {
    jitstate = new JITState(Modules[0]);

    FunctionPassManager &PM = jitstate->getPM();
    M->setDataLayout(TM.getDataLayout());
    PM.add(new DataLayoutPass(M));

    // Turn the machine code intermediate representation into bytes in memory
    // that may be executed.
    if (TM.addPassesToEmitMachineCode(PM, *JCE, !getVerifyModules())) {
      report_fatal_error("Target does not support machine code emission!");
    }

    // Initialize passes.
    PM.doInitialization();
  }
  return result;
}

/// run - Start execution with the specified function and arguments.
///
GenericValue JIT::runFunction(Function *F,
                              const std::vector<GenericValue> &ArgValues) {
  assert(F && "Function *F was null at entry to run()");

  void *FPtr = getPointerToFunction(F);
  assert(FPtr && "Pointer to fn's code was null after getPointerToFunction");
  FunctionType *FTy = F->getFunctionType();
  Type *RetTy = FTy->getReturnType();

  assert((FTy->getNumParams() == ArgValues.size() ||
          (FTy->isVarArg() && FTy->getNumParams() <= ArgValues.size())) &&
         "Wrong number of arguments passed into function!");
  assert(FTy->getNumParams() == ArgValues.size() &&
         "This doesn't support passing arguments through varargs (yet)!");

  // Handle some common cases first.  These cases correspond to common `main'
  // prototypes.
  if (RetTy->isIntegerTy(32) || RetTy->isVoidTy()) {
    switch (ArgValues.size()) {
      case 3:
        if (FTy->getParamType(0)->isIntegerTy(32) &&
            FTy->getParamType(1)->isPointerTy() &&
            FTy->getParamType(2)->isPointerTy()) {
          int (*PF)(int, char **, const char **) =
              (int (*)(int, char **, const char **))(intptr_t)FPtr;

          // Call the function.
          GenericValue rv;
          rv.IntVal = APInt(32, PF(ArgValues[0].IntVal.getZExtValue(),
                                   (char **)GVTOP(ArgValues[1]),
                                   (const char **)GVTOP(ArgValues[2])));
          return rv;
        }
        break;
      case 2:
        if (FTy->getParamType(0)->isIntegerTy(32) &&
            FTy->getParamType(1)->isPointerTy()) {
          int (*PF)(int, char **) = (int (*)(int, char **))(intptr_t)FPtr;

          // Call the function.
          GenericValue rv;
          rv.IntVal = APInt(32, PF(ArgValues[0].IntVal.getZExtValue(),
                                   (char **)GVTOP(ArgValues[1])));
          return rv;
        }
        break;
      case 1:
        if (FTy->getParamType(0)->isIntegerTy(32)) {
          GenericValue rv;
          int (*PF)(int) = (int (*)(int))(intptr_t)FPtr;
          rv.IntVal = APInt(32, PF(ArgValues[0].IntVal.getZExtValue()));
          return rv;
        }
        if (FTy->getParamType(0)->isPointerTy()) {
          GenericValue rv;
          int (*PF)(char *) = (int (*)(char *))(intptr_t)FPtr;
          rv.IntVal = APInt(32, PF((char *)GVTOP(ArgValues[0])));
          return rv;
        }
        break;
    }
  }

  // Handle cases where no arguments are passed first.
  if (ArgValues.empty()) {
    GenericValue rv;
    switch (RetTy->getTypeID()) {
      default:
        llvm_unreachable("Unknown return type for function call!");
      case Type::IntegerTyID: {
        unsigned BitWidth = cast<IntegerType>(RetTy)->getBitWidth();
        if (BitWidth == 1)
          rv.IntVal = APInt(BitWidth, ((bool (*)())(intptr_t)FPtr)());
        else if (BitWidth <= 8)
          rv.IntVal = APInt(BitWidth, ((char (*)())(intptr_t)FPtr)());
        else if (BitWidth <= 16)
          rv.IntVal = APInt(BitWidth, ((short (*)())(intptr_t)FPtr)());
        else if (BitWidth <= 32)
          rv.IntVal = APInt(BitWidth, ((int (*)())(intptr_t)FPtr)());
        else if (BitWidth <= 64)
          rv.IntVal = APInt(BitWidth, ((int64_t(*)())(intptr_t)FPtr)());
        else
          llvm_unreachable("Integer types > 64 bits not supported");
        return rv;
      }
      case Type::VoidTyID:
        rv.IntVal = APInt(32, ((int (*)())(intptr_t)FPtr)());
        return rv;
      case Type::FloatTyID:
        rv.FloatVal = ((float (*)())(intptr_t)FPtr)();
        return rv;
      case Type::DoubleTyID:
        rv.DoubleVal = ((double (*)())(intptr_t)FPtr)();
        return rv;
      case Type::X86_FP80TyID:
      case Type::FP128TyID:
      case Type::PPC_FP128TyID:
        llvm_unreachable("long double not supported yet");
      case Type::PointerTyID:
        return PTOGV(((void *(*)())(intptr_t)FPtr)());
    }
  }

  // Okay, this is not one of our quick and easy cases.  Because we don't have a
  // full FFI, we have to codegen a nullary stub function that just calls the
  // function we are interested in, passing in constants for all of the
  // arguments.  Make this function and return.

  // First, create the function.
  FunctionType *STy = FunctionType::get(RetTy, false);
  Function *Stub =
      Function::Create(STy, Function::InternalLinkage, "", F->getParent());

  // Insert a basic block.
  BasicBlock *StubBB = BasicBlock::Create(F->getContext(), "", Stub);

  // Convert all of the GenericValue arguments over to constants.  Note that we
  // currently don't support varargs.
  SmallVector<Value *, 8> Args;
  for (unsigned i = 0, e = ArgValues.size(); i != e; ++i) {
    Constant *C = nullptr;
    Type *ArgTy = FTy->getParamType(i);
    const GenericValue &AV = ArgValues[i];
    switch (ArgTy->getTypeID()) {
      default:
        llvm_unreachable("Unknown argument type for function call!");
      case Type::IntegerTyID:
        C = ConstantInt::get(F->getContext(), AV.IntVal);
        break;
      case Type::FloatTyID:
        C = ConstantFP::get(F->getContext(), APFloat(AV.FloatVal));
        break;
      case Type::DoubleTyID:
        C = ConstantFP::get(F->getContext(), APFloat(AV.DoubleVal));
        break;
      case Type::PPC_FP128TyID:
      case Type::X86_FP80TyID:
      case Type::FP128TyID:
        C = ConstantFP::get(F->getContext(),
                            APFloat(ArgTy->getFltSemantics(), AV.IntVal));
        break;
      case Type::PointerTyID:
        void *ArgPtr = GVTOP(AV);
        if (sizeof(void *) == 4)
          C = ConstantInt::get(Type::getInt32Ty(F->getContext()),
                               (int)(intptr_t)ArgPtr);
        else
          C = ConstantInt::get(Type::getInt64Ty(F->getContext()),
                               (intptr_t)ArgPtr);
        // Cast the integer to pointer
        C = ConstantExpr::getIntToPtr(C, ArgTy);
        break;
    }
    Args.push_back(C);
  }

  CallInst *TheCall = CallInst::Create(F, Args, "", StubBB);
  TheCall->setCallingConv(F->getCallingConv());
  TheCall->setTailCall();
  if (!TheCall->getType()->isVoidTy())
    // Return result of the call.
    ReturnInst::Create(F->getContext(), TheCall, StubBB);
  else
    ReturnInst::Create(F->getContext(), StubBB);  // Just return void.

  // Finally, call our nullary stub function.
  GenericValue Result = runFunction(Stub, std::vector<GenericValue>());
  // Erase it, since no other function can have a reference to it.
  Stub->eraseFromParent();
  // And return the result.
  return Result;
}

void JIT::RegisterJITEventListener(JITEventListener *L) {
  if (!L) return;
  MutexGuard locked(lock);
  EventListeners.push_back(L);
}
void JIT::UnregisterJITEventListener(JITEventListener *L) {
  if (!L) return;
  MutexGuard locked(lock);
  std::vector<JITEventListener *>::reverse_iterator I =
      std::find(EventListeners.rbegin(), EventListeners.rend(), L);
  if (I != EventListeners.rend()) {
    std::swap(*I, EventListeners.back());
    EventListeners.pop_back();
  }
}
void JIT::NotifyFunctionEmitted(
    const Function &F, void *Code, size_t Size,
    const JITEvent_EmittedFunctionDetails &Details) {
  MutexGuard locked(lock);
  for (unsigned I = 0, S = EventListeners.size(); I < S; ++I) {
    EventListeners[I]->NotifyFunctionEmitted(F, Code, Size, Details);
  }
}

void JIT::NotifyFreeingMachineCode(void *OldPtr) {
  MutexGuard locked(lock);
  for (unsigned I = 0, S = EventListeners.size(); I < S; ++I) {
    EventListeners[I]->NotifyFreeingMachineCode(OldPtr);
  }
}

/// runJITOnFunction - Run the FunctionPassManager full of
/// just-in-time compilation passes on F, hopefully filling in
/// GlobalAddress[F] with the address of F's machine code.
///
void JIT::runJITOnFunction(Function *F, MachineCodeInfo *MCI) {
  MutexGuard locked(lock);

  class MCIListener : public JITEventListener {
    MachineCodeInfo *const MCI;

   public:
    MCIListener(MachineCodeInfo *mci) : MCI(mci) {}
    void NotifyFunctionEmitted(const Function &, void *Code, size_t Size,
                               const EmittedFunctionDetails &) override {
      MCI->setAddress(Code);
      MCI->setSize(Size);
    }
  };
  MCIListener MCIL(MCI);
  if (MCI) RegisterJITEventListener(&MCIL);

  runJITOnFunctionUnlocked(F);

  if (MCI) UnregisterJITEventListener(&MCIL);
}

void JIT::runJITOnFunctionUnlocked(Function *F) {
  assert(!isAlreadyCodeGenerating && "Error: Recursive compilation detected!");

  jitTheFunctionUnlocked(F);

  // If the function referred to another function that had not yet been
  // read from bitcode, and we are jitting non-lazily, emit it now.
  while (!jitstate->getPendingFunctions().empty()) {
    Function *PF = jitstate->getPendingFunctions().back();
    jitstate->getPendingFunctions().pop_back();

    assert(!PF->hasAvailableExternallyLinkage() &&
           "Externally-defined function should not be in pending list.");

    jitTheFunctionUnlocked(PF);

    // Now that the function has been jitted, ask the JITEmitter to rewrite
    // the stub with real address of the function.
    updateFunctionStubUnlocked(PF);
  }
}

void JIT::jitTheFunctionUnlocked(Function *F) {
  isAlreadyCodeGenerating = true;
  jitstate->getPM().run(*F);
  isAlreadyCodeGenerating = false;

  // clear basic block addresses after this function is done
  getBasicBlockAddressMap().clear();
}

/// getPointerToFunction - This method is used to get the address of the
/// specified function, compiling it if necessary.
///
void *JIT::getPointerToFunction(Function *F) {
  if (void *Addr = getPointerToGlobalIfAvailable(F))
    return Addr;  // Check if function already code gen'd

  MutexGuard locked(lock);

  // Now that this thread owns the lock, make sure we read in the function if it
  // exists in this Module.
  std::string ErrorMsg;
  if (F->Materialize(&ErrorMsg)) {
    report_fatal_error("Error reading function '" + F->getName() +
                       "' from bitcode file: " + ErrorMsg);
  }

  // ... and check if another thread has already code gen'd the function.
  if (void *Addr = getPointerToGlobalIfAvailable(F)) return Addr;

  if (F->isDeclaration() || F->hasAvailableExternallyLinkage()) {
    bool AbortOnFailure = !F->hasExternalWeakLinkage();
    void *Addr = getPointerToNamedFunction(F->getName(), AbortOnFailure);
    addGlobalMapping(F, Addr);
    return Addr;
  }

  runJITOnFunctionUnlocked(F);

  void *Addr = getPointerToGlobalIfAvailable(F);
  assert(Addr && "Code generation didn't add function to GlobalAddress table!");
  return Addr;
}

void JIT::addPointerToBasicBlock(const BasicBlock *BB, void *Addr) {
  MutexGuard locked(lock);

  BasicBlockAddressMapTy::iterator I = getBasicBlockAddressMap().find(BB);
  if (I == getBasicBlockAddressMap().end()) {
    getBasicBlockAddressMap()[BB] = Addr;
  } else {
    // ignore repeats: some BBs can be split into few MBBs?
  }
}

void JIT::clearPointerToBasicBlock(const BasicBlock *BB) {
  MutexGuard locked(lock);
  getBasicBlockAddressMap().erase(BB);
}

void *JIT::getPointerToBasicBlock(BasicBlock *BB) {
  // make sure it's function is compiled by JIT
  (void)getPointerToFunction(BB->getParent());

  // resolve basic block address
  MutexGuard locked(lock);

  BasicBlockAddressMapTy::iterator I = getBasicBlockAddressMap().find(BB);
  if (I != getBasicBlockAddressMap().end()) {
    return I->second;
  } else {
    llvm_unreachable(
        "JIT does not have BB address for address-of-label, was"
        " it eliminated by optimizer?");
  }
}

void *JIT::getPointerToNamedFunction(const std::string &Name,
                                     bool AbortOnFailure) {
  if (!isSymbolSearchingDisabled()) {
    void *ptr = JMM->getPointerToNamedFunction(Name, false);
    if (ptr) return ptr;
  }

  /// If a LazyFunctionCreator is installed, use it to get/create the function.
  if (LazyFunctionCreator)
    if (void *RP = LazyFunctionCreator(Name)) return RP;

  if (AbortOnFailure) {
    report_fatal_error("Program used external function '" + Name +
                       "' which could not be resolved!");
  }
  return nullptr;
}

/// getOrEmitGlobalVariable - Return the address of the specified global
/// variable, possibly emitting it to memory if needed.  This is used by the
/// Emitter.
void *JIT::getOrEmitGlobalVariable(const GlobalVariable *GV) {
  MutexGuard locked(lock);

  void *Ptr = getPointerToGlobalIfAvailable(GV);
  if (Ptr) return Ptr;

  // If the global is external, just remember the address.
  if (GV->isDeclaration() || GV->hasAvailableExternallyLinkage()) {
#if HAVE___DSO_HANDLE
    if (GV->getName() == "__dso_handle") return (void *)&__dso_handle;
#endif
    Ptr = sys::DynamicLibrary::SearchForAddressOfSymbol(GV->getName());
    if (!Ptr) {
      report_fatal_error("Could not resolve external global address: " +
                         GV->getName());
    }
    addGlobalMapping(GV, Ptr);
  } else {
    // If the global hasn't been emitted to memory yet, allocate space and
    // emit it into memory.
    Ptr = getMemoryForGV(GV);
    addGlobalMapping(GV, Ptr);
    EmitGlobalVariable(GV);  // Initialize the variable.
  }
  return Ptr;
}

/// recompileAndRelinkFunction - This method is used to force a function
/// which has already been compiled, to be compiled again, possibly
/// after it has been modified. Then the entry to the old copy is overwritten
/// with a branch to the new copy. If there was no old copy, this acts
/// just like JIT::getPointerToFunction().
///
void *JIT::recompileAndRelinkFunction(Function *F) {
  void *OldAddr = getPointerToGlobalIfAvailable(F);

  // If it's not already compiled there is no reason to patch it up.
  if (!OldAddr) return getPointerToFunction(F);

  // Delete the old function mapping.
  addGlobalMapping(F, nullptr);

  // Recodegen the function
  runJITOnFunction(F);

  // Update state, forward the old function to the new function.
  void *Addr = getPointerToGlobalIfAvailable(F);
  assert(Addr && "Code generation didn't add function to GlobalAddress table!");
  TJI.replaceMachineCodeForFunction(OldAddr, Addr);
  return Addr;
}

/// getMemoryForGV - This method abstracts memory allocation of global
/// variable so that the JIT can allocate thread local variables depending
/// on the target.
///
char *JIT::getMemoryForGV(const GlobalVariable *GV) {
  char *Ptr;

  // GlobalVariable's which are not "constant" will cause trouble in a server
  // situation. It's returned in the same block of memory as code which may
  // not be writable.
  if (isGVCompilationDisabled() && !GV->isConstant()) {
    report_fatal_error("Compilation of non-internal GlobalValue is disabled!");
  }

  // Some applications require globals and code to live together, so they may
  // be allocated into the same buffer, but in general globals are allocated
  // through the memory manager which puts them near the code but not in the
  // same buffer.
  Type *GlobalType = GV->getType()->getElementType();
  size_t S = getDataLayout()->getTypeAllocSize(GlobalType);
  size_t A = getDataLayout()->getPreferredAlignment(GV);
  if (GV->isThreadLocal()) {
    MutexGuard locked(lock);
    Ptr = TJI.allocateThreadLocalMemory(S);
  } else if (TJI.allocateSeparateGVMemory()) {
    if (A <= 8) {
      Ptr = (char *)malloc(S);
    } else {
      // Allocate S+A bytes of memory, then use an aligned pointer within that
      // space.
      Ptr = (char *)malloc(S + A);
      unsigned MisAligned = ((intptr_t)Ptr & (A - 1));
      Ptr = Ptr + (MisAligned ? (A - MisAligned) : 0);
    }
  } else if (AllocateGVsWithCode) {
    Ptr = (char *)JCE->allocateSpace(S, A);
  } else {
    Ptr = (char *)JCE->allocateGlobal(S, A);
  }
  return Ptr;
}

void JIT::addPendingFunction(Function *F) {
  MutexGuard locked(lock);
  jitstate->getPendingFunctions().push_back(F);
}

JITEventListener::~JITEventListener() {}
