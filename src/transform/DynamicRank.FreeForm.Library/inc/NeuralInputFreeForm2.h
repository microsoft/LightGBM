/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#pragma once

#ifndef FREEFORM2_NEURAL_INPUT_FREEFORM2_H
#define FREEFORM2_NEURAL_INPUT_FREEFORM2_H

#include <NeuralInput.h>
#include <NeuralInputFactory.h>

#include <boost/foreach.hpp>
#include <boost/noncopyable.hpp>
#include <boost/shared_array.hpp>
#include <boost/shared_ptr.hpp>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "FreeForm2.h"
#include "FreeForm2CompilerFactory.h"
#include "FreeForm2Executable.h"
#include "NeuralInputFreeForm2_types.h"
#include "UnionBondInput_types.h"

namespace FreeForm2 {
class Program;

// Match with FillBond, Loader reguster. Also match the model .ini.
static const char *c_freeform2_tranform = "freeform2";
static const char *c_bulkinput_tranform = "bulkinput";

// a NeuralInput that compiles several FreeFormv2 expressions together into
// fewer functions that can be bulk evaluated. This is done to increase
// performance.
class BulkCompiledNeuralInput : public DynamicRank::BulkNeuralInput {
 public:
  BulkCompiledNeuralInput();

  BulkCompiledNeuralInput(std::vector<boost::shared_ptr<Program> > p_programs,
                          size_t p_numInputs,
                          const std::vector<UInt32> &p_features);

  void Evaluate(const UInt32 p_input[], float p_output[]) const override;

  void GetAllAssociatedFeatures(
      std::vector<UInt32> &p_associatedFeaturesList) const override;

  virtual size_t GetNumberOfNeuralInputs() const;

  // Get the size of this object, including internal and external memory.
  size_t GetSize() const override;

  // Bond Serialization.
  void FillBond(DynamicRank::UnionBondInput &p_data) const override;

  bool Equal(const DynamicRank::BulkNeuralInput *p_other) const override;

  explicit BulkCompiledNeuralInput(const DynamicRank::UnionBondInput &p_data);

  // BulkCompiledNeuralInput never load from config.ini since it's constructed
  // by ff2 compiler or reconstructed from UnionBondInput.
  static BulkCompiledNeuralInput *Load(DynamicRank::Config &p_config,
                                       const char *p_section,
                                       DynamicRank::IFeatureMap *p_featureMap,
                                       const char *p_transform);

 protected:
  // A std::vector of features used by the FreeForm expression.
  std::vector<UInt32> m_features;

  // Executable program.
  std::vector<boost::shared_ptr<Executable> > m_execs;

  // Directly executable function.
  std::vector<Executable::DirectEvalFun> m_funs;

  // Number of neural inputs.
  size_t m_numInputs;

 private:
  friend class NeuralInputFreeForm2;

  // Functions to do low-level serialization and deserialization of executable
  // code.
  static std::pair<boost::shared_array<unsigned char>, size_t> SerializeBlob(
      const std::vector<boost::shared_ptr<Executable> > &p_execs);
  static std::pair<boost::shared_array<unsigned char>, size_t>
  SerializePhoenixBlob(
      const std::vector<boost::shared_ptr<Executable> > &p_execs);
  static std::pair<boost::shared_array<unsigned char>, size_t>
  SerializeLlvmBlob(const std::vector<boost::shared_ptr<Executable> > &p_execs);

  static void DeserializeBlob(
      const unsigned char *p_buffer, size_t p_length,
      std::vector<boost::shared_ptr<Executable> > &p_execs,
      std::vector<Executable::DirectEvalFun> &p_funs);
};

// NeuralInputFreeForm2 wraps freeform2 expression evaluation in a neural
// input, which makes it suitable for use in neural nets and tree ensembles.
class NeuralInputFreeForm2 : public DynamicRank::NeuralInput {
 public:
  // Constructor, taking the input program, the name of the transform used
  // to identify this neural input, the feature map to use when compiling
  // the program.
  NeuralInputFreeForm2(const std::string &p_input, const char *p_transform,
                       DynamicRank::IFeatureMap &p_map);

  NeuralInputFreeForm2(const std::string &p_input, const char *p_transform,
                       DynamicRank::IFeatureMap &p_map,
                       boost::shared_ptr<Program> p_program);

  // Evaluate.
  double Evaluate(UInt32 p_input[]) const override;

  // Train.
  bool Train(double p_learningRate, double p_outputHigh, double p_outputLow,
             double p_outputDelta, UInt32 p_inputHigh[],
             UInt32 p_inputLow[]) override;

  // Get all associated features.
  void GetAllAssociatedFeatures(
      std::vector<UInt32> &p_associatedFeaturesList) const override;

  // Compile the freeform expression, making this input available for use.
  // The provided compiler will be used if not NULL, a compiler will be
  // created if not.
  virtual void Compile(Compiler *p_compiler);

  // Returns the Program referenced by this NeuralInput. Used for serialization.
  const Program &GetProgram() const;

  // Bond Serialization.
  void FillBond(DynamicRank::UnionBondInput &p_data) const override;

  bool Equal(const DynamicRank::NeuralInput *p_other) const override;

  explicit NeuralInputFreeForm2(const DynamicRank::UnionBondInput &p_data);

  // Interface to support batch serialization.
  // Just for special batch serializaton of freeform2.
  bool IsFreeForm2() const override;

  double GetMin() const override;

  double GetMax() const override;

  bool Save(FILE *p_out, size_t p_input,
            const DynamicRank::IFeatureMap &p_map) const override;

  // Function that loads a freeform2 neural input from a config
  // file.  Returns a pointer allocated via new, transferring
  // ownership (as per NeuralInputFactory requirements).
  static NeuralInputFreeForm2 *
  Load(DynamicRank::Config &p_config, const char *p_section,
      DynamicRank::IFeatureMap *p_featureMap, const char *p_transform);

  // Get the size of this object, including internal and external memory.
  size_t GetSize() const override;

 protected:
  // Gets a string representation of the input. This is used to log if an error
  // occurs while compiling.
  virtual std::string GetStringRepresentation() const;

  // Get the size of external memory.
  size_t GetExternalSize() const;

  // Load a freeform2 neural input string from a config file.
  static std::string LoadProgram(DynamicRank::Config &p_config,
                                 const char *p_section,
                                 const DynamicRank::IFeatureMap *p_featureMap,
                                 const char *p_transform);

  // Which transform was used to generate this input.
  const char *m_transform;

  // Compiled program.
  boost::shared_ptr<Program> m_program;

  // A std::vector of features used by the FreeForm expression.
  std::vector<UInt32> m_features;

  // Executable program.
  boost::shared_ptr<Executable> m_exec;

  // Directly executable function.
  Executable::DirectEvalFun m_fun;

  // Feature map used to compile feature.  This should be a shared
  // pointer, as we need it to live for the lifetime of this object.  TFS
  // task 55302 is open to fixing this issue.
  DynamicRank::IFeatureMap *m_map;

  NeuralInputFreeForm2();

 private:
  // Init all members.
  void Init();

  // Program input, concatenated onto a single line.
  std::string m_input;
};

// Class to hide Compile's implementation.
class NeuralInputCompiler : boost::noncopyable {
 public:
  static void Compile(const std::vector<NeuralInputFreeForm2 *> &p_inputs,
                      Compiler &p_compiler);

 private:
  // Make sure nobody can create instances of this class.
  NeuralInputCompiler();
};

// Class to load neural inputs from neural net/tree ensemble .ini files and then
// compile them.
template <class T>
class CompiledNeuralInputLoader
    : public DynamicRank::NeuralInputFactory::Loader {
 public:
  // Constructor, taking the name of the transform which this loader will
  // be responsible for.
  explicit CompiledNeuralInputLoader(const char *p_transform)
      : m_transform(p_transform) {}

  // Functor to create a NeuralInput given appropriate inputs.
  virtual DynamicRank::NeuralInput *operator()(
      DynamicRank::Config &p_config, const char *p_section,
      DynamicRank::IFeatureMap &p_featureMap) const {
    T *t = T::Load(p_config, p_section, &p_featureMap, m_transform);
    NeuralInputFreeForm2 *input = dynamic_cast<NeuralInputFreeForm2 *>(t);

    if (input != NULL) {
      // A stateful factory.
      m_loaded.push_back(input);
    }

    return input;
  }

  virtual DynamicRank::NeuralInput *FromBond(
      const DynamicRank::UnionBondInput &p_data) const {
    // To handle all the cases we let all the cases' constructor from bond
    // Take the DynamicRank::UnionBondInput as parameter.
    T *t = new T(p_data);
    DynamicRank::NeuralInput *t1 = dynamic_cast<DynamicRank::NeuralInput *>(t);
    if (!t1) {
      delete t;
      t = nullptr;
    }
    return t1;
  }

  virtual DynamicRank::BulkNeuralInput *FromBulkBond(
      const DynamicRank::UnionBondInput &p_data) const {
    // To handle all the cases we let all the cases' constructor from bond
    // Take the DynamicRank::UnionBondInput as parameter.
    T *t = new T(p_data);
    DynamicRank::BulkNeuralInput *t1 =
        dynamic_cast<DynamicRank::BulkNeuralInput *>(t);
    if (!t1) {
      delete t;
      t = nullptr;
    }
    return t1;
  }

  // Compile all freeform2 programs loaded through this factory.
  void Compile(Compiler &p_compiler) const {
    Compile(m_loaded, p_compiler);

    m_loaded.clear();
  }

  // Static function to compile a given set of inputs.
  static void Compile(const std::vector<NeuralInputFreeForm2 *> &p_inputs,
                      Compiler &p_compiler) {
    NeuralInputCompiler::Compile(p_inputs, p_compiler);
  }

 private:
  // The transform this loader is loading.
  const char *m_transform;

  // Vector of loaded compiled neural inputs, kept in order to allow compilation
  // after loading.  This is not great, but allows us to avoid individual
  // compilation of each input, which takes excessive amounts of
  // time.
  mutable std::vector<NeuralInputFreeForm2 *> m_loaded;
};

// A base class that groups a large amount of NeuralInputs and combines them
// into a few executable functions. They should have the same effect as running
// all the individual functions.
class CompiledBulkNeuralInputLoaderBase
    : public DynamicRank::BulkNeuralInputFactory {
 public:
  // The reduction factor is how many functions should be compiled together at
  // a time. A factor of 100 means that it will bulk compile in batches of 100.
  static const UInt32 c_DefaultReductionFactor = 100;

  // p_reductionFactor must be greater or equal to 1.
  explicit CompiledBulkNeuralInputLoaderBase(UInt32 p_reductionFactor);

  virtual std::unique_ptr<DynamicRank::BulkNeuralInput> ConvertToBulkInput(
      const std::vector<DynamicRank::BulkNeuralInputFactory::InputAndIndex>
          &p_inputs,
      DynamicRank::IFeatureMap &p_featureMap) const = 0;

 protected:
  // Load all neuralinput to a vector, extracted from ConvertToBulkInput for
  // BulkAggregatedNeuralInput.
  size_t LoadProgramsForBulkInput(
      std::vector<boost::shared_ptr<Program> > &p_programs,
      std::vector<UInt32> &p_features,
      const std::vector<DynamicRank::BulkNeuralInputFactory::InputAndIndex>
          &p_inputs,
      DynamicRank::IFeatureMap &p_featureMap) const;

 private:
  const UInt32 m_reductionFactor;
};

template <class T>
class CompiledBulkNeuralInputLoader : public CompiledBulkNeuralInputLoaderBase {
 public:
  // p_reductionFactor must be greater or equal to 1.
  explicit CompiledBulkNeuralInputLoader(UInt32 p_reductionFactor)
      : CompiledBulkNeuralInputLoaderBase(p_reductionFactor) {}

  virtual std::unique_ptr<DynamicRank::BulkNeuralInput> ConvertToBulkInput(
      const std::vector<DynamicRank::BulkNeuralInputFactory::InputAndIndex>
          &p_inputs,
      DynamicRank::IFeatureMap &p_featureMap) const {
    std::vector<UInt32> features;
    std::vector<boost::shared_ptr<Program> > programs;
    size_t convertedInputsSize =
        LoadProgramsForBulkInput(programs, features, p_inputs, p_featureMap);

    return std::unique_ptr<T>(new T(programs, convertedInputsSize, features));
  }
};

// This loader is just for FromBulkBond for BulkNeuralInput.
template <class T>
class FromBulkBondLoader : public DynamicRank::NeuralInputFactory::Loader {
 public:
  FromBulkBondLoader() {}

  virtual DynamicRank::NeuralInput *operator()(
      DynamicRank::Config &p_config, const char *p_section,
      DynamicRank::IFeatureMap &p_featureMap) const {
    // Not implemented.
    return nullptr;
  }

  virtual DynamicRank::NeuralInput *FromBond(
      const DynamicRank::UnionBondInput &p_data) const {
    // Not implemented.
    return nullptr;
  }

  virtual DynamicRank::BulkNeuralInput *FromBulkBond(
      const DynamicRank::UnionBondInput &p_data) const {
    // To handle all the cases we let all the cases' constructor from bond
    // Take the DynamicRank::UnionBondInput as parameter.
    T *t = new T(p_data);
    DynamicRank::BulkNeuralInput *t1 =
        dynamic_cast<DynamicRank::BulkNeuralInput *>(t);
    if (!t1) {
      delete t;
      t = nullptr;
    }
    return t1;
  }
};

// A DynamicRank::NeuralInputFactory with the NeuralInputFreeForm2 loader
//  and CompiledBulkNeuralInputLoader pre-registered.
// CompiledBulkNeuralInputLoader registered mainly for FromBond logic.
class CompiledNeuralInputFactory : public DynamicRank::NeuralInputFactory {
 public:
  CompiledNeuralInputFactory();

  const CompiledNeuralInputLoader<NeuralInputFreeForm2> &GetFreeForm2Loader()
      const;

 private:
  boost::shared_ptr<CompiledNeuralInputLoader<NeuralInputFreeForm2> >
      m_ff2Loader;

  // Register both loaders to help reconstruct tree from Bond.
  boost::shared_ptr<FromBulkBondLoader<BulkCompiledNeuralInput> > m_bulkLoader;
};
}  // namespace FreeForm2

#endif
