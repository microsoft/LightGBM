#pragma once

#include "basic_types.h"
#include "IFeatureMap.h"
#include "FeaSpecConfig.h"
#include <string>
#include <boost/noncopyable.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/scoped_array.hpp>
#include <stdio.h>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "MigratedApi.h"


class NeuralInputTest;


namespace DynamicRank
{

struct UnionBondInput;
struct FreeForm2CodeBondData;
struct NeuralInputBondData;
struct NeuralInputUnaryBondData;
struct NeuralInputLinearBondData;
struct NeuralInputLogLinearBondData;
struct NeuralInputRationalBondData;
struct NeuralInputBucketBondData;
struct NeuralInputTanhBondData;
struct NeuralInputTanhUnaryBondData;
struct NeuralInputUseAsFloatBondData;

// Base class representing an input value for a neural net
class NeuralInput : boost::noncopyable
{
protected:
    NeuralInput();
    
    // Get the size of external memory owned by this object.
    size_t GetExternalSize() const;

public:

    // Construct from Bond.
    explicit NeuralInput(const NeuralInputBondData& p_data);

    virtual ~NeuralInput();
    
    // Just for special batch serializaton of freeform2.
    virtual bool IsFreeForm2() const { return false; };

    virtual void BatchSerialize(const std::vector<const NeuralInput*>& /*p_inputs*/, FreeForm2CodeBondData& /*p_blob*/) const {};

    virtual void BatchUnSerialize(const std::vector<NeuralInput*>& /*p_inputs*/, const FreeForm2CodeBondData& /*p_blob*/) const {};

    // Fill the correct input name and data field.
    virtual void FillBond(UnionBondInput& p_data) const = 0;
    
    // Fill bond structure of NueralInput class. 
    void FillBondData(NeuralInputBondData& p_data) const;

    virtual double Evaluate(UInt32 input[]) const = 0;
    virtual double EvaluateInput(UInt32 input) const;

    // Evaluate an input for a document in all documents context.
    virtual double Evaluate(UInt32** p_featureVectorArray,
                            UInt32 p_currentDocument,
                            UInt32 p_documentCount) const;
    
    // Get the minimum and maximum possible outputs for this input node. 
    virtual double GetMin() const = 0;
    virtual double GetMax() const = 0;

    virtual UInt32 GetAssociatedFeature() const;
    virtual void GetAllAssociatedFeatures(std::vector<UInt32>& associatedFeaturesList) const = 0;

    virtual double GetSlope() const;
    virtual double GetIntercept() const;
    
    virtual bool Save(FILE *fpOutput, size_t nInputId,const IFeatureMap& p_featureMap) const;

    // Is same Input.
    virtual bool Equal(const NeuralInput* p_input) const = 0;
    
    // Compare the internal members are equal.
    // This is not virtual and just compare the base class members.
    bool EqualInternal(const NeuralInput* p_input) const;
    
    void SetSegments(const std::vector<std::string>& p_segments);

    virtual bool Train(double dblLearningRate, double outputHigh,
                       double outputLow, double dblOutputDelta,
                       UInt32 inputHigh[], UInt32 inputLow[]);

    // load Segments into m_segments
    void LoadSegments(DynamicRank::Config& p_config, const char *szSection);

    // Get the size of this object, including internal and external memory 
    // (memory accessed through pointers or objects contain pointers e.g. std::string, std::vector, etc.).
    virtual size_t GetSize() const = 0;

    // Get the list of segments this input belongs to. Empty means it belongs to base ranker.
    const std::vector<std::string>& GetSegments() const;
	
private:
    // Segments to which this input applies; a size zero vector means the input is part of the main L2 ranker
    boost::scoped_ptr<std::vector<std::string> > m_segments;

    friend class NeuralInputTest;
};


// An exception that is thrown when trying to bulk-optimize a list of NeuralInputs if at least
// one of them is not compatible with bulk optimization.
class BulkOptimizationUnsupported : public std::runtime_error
{
public:
    explicit BulkOptimizationUnsupported(const std::string& p_message);
};


// Base class that provides a replacement function to evaluate a set of neural inputs.
class BulkNeuralInput : boost::noncopyable
{
protected:
    BulkNeuralInput();

public:
    
    virtual ~BulkNeuralInput();

    // Fill the correct input name and data field.
    virtual void FillBond(UnionBondInput& p_data) const = 0;
    
    virtual bool Equal(const BulkNeuralInput* p_other) const = 0;

    // Add all the features referenced by this BulkNeuralInput object into the
    // INeuralNetFetures reference.
    /* virtual void AddNeuralNetFeatures(INeuralNetFeatures& p_neuralNetFeatures) const = 0; */

    // Evaluates the neural inputs and places the result of their evaluation in the
    // corresponding indices in the p_output array.
    virtual void Evaluate(const UInt32 p_input[], float p_output[]) const = 0;

    // Evaluates the neural inputs in all documents context and places the result
    //     of their evalution in the corresponding indices in the p_output array.
    virtual void Evaluate(UInt32** p_featureVectorArray,
                          UInt32 p_currentDocument,
                          UInt32 p_documentCount,
                          float p_output[]) const;
    
    // Populates p_associatedFeaturesList with all the feature indices that this
    // BulkNeuralInput object requires.
    virtual void GetAllAssociatedFeatures(std::vector<UInt32>& p_associatedFeaturesList) const = 0;
    
    // Get the size of this object, including internal and external memory 
    // (memory accessed through pointers or objects contain pointers e.g. std::string, std::vector, etc.).
    virtual size_t GetSize() const = 0;
};


// Base class representing an input value for a neural net
// that is transformed by a function that can be expressed as a function
// with only 1 integer input
class NeuralInputUnary : public NeuralInput
{
public:

    // Construct from Bond.
    explicit NeuralInputUnary(const NeuralInputUnaryBondData& p_data);

    // Fill the correct input name and data field.
    void FillBond(UnionBondInput& /*p_data*/) const {};

    // Fill bond structure of NeuralInputUnary class. 
    void FillBondData(NeuralInputUnaryBondData& p_data) const;
        
    // Copy all members.
    void CopyFrom(const NeuralInputUnary& p_neuralInputUnary);
    
    static bool ReadAssociatedFeature(DynamicRank::Config& p_config,
                                      const char *szSection,
                                      IFeatureMap& p_featureMap,
                                      UInt32 *piFeature);

    UInt32 GetAssociatedFeature() const;
    void GetAllAssociatedFeatures(std::vector<UInt32>& associatedFeaturesList) const;
    double Evaluate(UInt32 input[]) const;
    
    bool Save(FILE *fpOutput, size_t nInputId,const IFeatureMap& p_featureMap) const;
    
    // Check objects are equal.
    virtual bool Equal(const NeuralInput* p_input) const;
    
    // Get the size of this object, including internal and external memory 
    // (memory accessed through pointers or objects contain pointers e.g. std::string, std::vector, etc.).
    size_t GetSize() const;
    
    // Get feature.
    UInt32 GetFeature() const;
    
protected:

    // Default constructor for unit tests.
    NeuralInputUnary();
    
    NeuralInputUnary(int iFeature);
    
    NeuralInputUnary(int iFeature, IFeatureMap& p_featureMap);

    // Get the size of external memory owned by this object.
    size_t GetExternalSize() const;

    // Index of this input in the input vector
    UInt32 m_iFeature;    
};


// Linear input, using only a slope and an intercept
class NeuralInputLinear : public NeuralInputUnary
{
protected:
    double m_slope;
    double m_intercept;

    // Default constructor.	
    NeuralInputLinear();
	
    NeuralInputLinear(int id, double slope, double intercept);

    // Get the size of external memory owned by this object.
    size_t GetExternalSize() const;

public:

    // Construct from Bond.
    explicit NeuralInputLinear(const NeuralInputLinearBondData& p_data);
    
    // Fill the correct input name and data field.
    void FillBond(UnionBondInput& p_data) const;

    // Fill bond structure of NeuralInputLinear class. 
    void FillBondData(NeuralInputLinearBondData& p_data) const;
        
    // Get member variables.
    double GetSlope() const;
    double GetIntercept() const;

    double GetMin() const;
    double GetMax() const;
    double EvaluateInput(UInt32 input) const;
    bool Save(FILE *fpOutput, size_t nInputId, const IFeatureMap& p_featureMap) const;

    // Same input.
    virtual bool Equal(const NeuralInput* p_input) const;
    
    static NeuralInputLinear *Load(
		DynamicRank::Config& p_config,
        const char *szSection,
        IFeatureMap& p_featureMap);

    // Get the size of this object, including internal and external memory 
    // (memory accessed through pointers or objects contain pointers e.g. std::string, std::vector, etc.).
    size_t GetSize() const;
    
private:

    friend class NeuralInputTest;    
};


// Maps an input x to x/(c + x) where c is a damping factor
class NeuralInputRational : public NeuralInputUnary
{
protected:
    double m_dblDampingFactor;
    NeuralInputRational(int p_id, double p_dblDampingFactor);

    // Get the size of external memory owned by this object.
    size_t GetExternalSize() const;

public:

    // Construct from Bond.
    explicit NeuralInputRational(const NeuralInputRationalBondData& p_data);
    
    // Fill the correct input name and data field.
    void FillBond(UnionBondInput& p_data) const;

    // Fill bond structure of NeuralInputRational class. 
    void FillBondData(NeuralInputRationalBondData& p_data) const;
        
    // Get member variables.
    double GetDampingFactor() const;

    double GetMin() const;
    double GetMax() const;
    double EvaluateInput(UInt32 input) const;
    bool Save(FILE *fpOutput, size_t nInputId, const IFeatureMap& p_featureMap) const;

    // Same input.
    virtual bool Equal(const NeuralInput* p_input) const;
    
    static NeuralInputRational *Load(
		DynamicRank::Config& p_config,
        const char *szSection,
        IFeatureMap& p_featureMap);

    // Get the size of this object, including internal and external memory 
    // (memory accessed through pointers or objects contain pointers e.g. std::string, std::vector, etc.).
    size_t GetSize() const;
    
private:

    friend class NeuralInputTest;
};


// LogLinear input, applying the log and then transforming 
// using slope and intercept.
class NeuralInputLogLinear : public NeuralInputLinear
{
protected:
    NeuralInputLogLinear(int id, double slope, double intercept);

    // Get the size of external memory owned by this object.
    size_t GetExternalSize() const;

public:

    // Construct from Bond.
    explicit NeuralInputLogLinear(const NeuralInputLogLinearBondData& p_data);
    
    // Fill the correct input name and data field.
    void FillBond(UnionBondInput& p_data) const;

    // Fill bond structure of NeuralInputLogLinear class. 
    void FillBondData(NeuralInputLogLinearBondData& p_data) const;
        
    double EvaluateInput(UInt32 input) const;
    bool Save(FILE *fpOutput, size_t nInputId, const IFeatureMap& p_featureMap) const;

    // Same input.
    virtual bool Equal(const NeuralInput* p_input) const;
    

    static NeuralInputLogLinear *Load(
		DynamicRank::Config& p_config,
        const char *szSection,
        IFeatureMap& p_featureMap);

    // Get the size of this object, including internal and external memory 
    // (memory accessed through pointers or objects contain pointers e.g. std::string, std::vector, etc.).
    size_t GetSize() const;
    
private:

    friend class NeuralInputTest;
};


// Bucket Input, transforming the input to 0 or 1 depending on whether or input
// falls within defined bucket
class NeuralInputBucket : public NeuralInputUnary
{
protected:
    bool m_fMinInclusive;
    bool m_fMaxInclusive;

    UInt32 m_nMinValue;
    UInt32 m_nMaxValue;

    // Get the size of external memory owned by this object.
    size_t GetExternalSize() const;

public:

    NeuralInputBucket(int p_id, double p_min, bool p_mininclusive, double p_max, bool p_maxinclusive);

    // Construct from Bond.
    explicit NeuralInputBucket(const NeuralInputBucketBondData& p_data);
    
    // Fill the correct input name and data field.
    void FillBond(UnionBondInput& p_data) const;

    // Fill bond structure of NeuralInputBucket class. 
    void FillBondData(NeuralInputBucketBondData& p_data) const;
    
    double GetMin() const;
    double GetMax() const;
    
    // Get member variables.
    bool GetMinInclusive() const;
    bool GetMaxInclusive() const;
    UInt32 GetMinValue() const;
    UInt32 GetMaxValue() const;

    double EvaluateInput(UInt32 input) const;
    bool Save(FILE *fpOutput, size_t nInputId, const IFeatureMap& p_featureMap) const;
    
    // Same input.
    virtual bool Equal(const NeuralInput* p_input) const;
    
    static NeuralInputBucket *Load(
		DynamicRank::Config& p_config,
        const char *szSection,
        IFeatureMap& p_featureMap);

    // Get the size of this object, including internal and external memory 
    // (memory accessed through pointers or objects contain pointers e.g. std::string, std::vector, etc.).
    size_t GetSize() const;
};


// A caching wrapper around an underlying input
class NeuralInputCached : public NeuralInputUnary
{
protected:
    size_t m_cacheSize;
    boost::scoped_array<double> m_resultCache;
    boost::scoped_ptr<NeuralInputUnary> m_input;
    
    NeuralInputCached(size_t nCacheSize, NeuralInputUnary *pChild);
    
    // Get the size of external memory owned by this object.
    size_t GetExternalSize() const;

public:
    ~NeuralInputCached();

    // Fill the correct input name and data field.
    void FillBond(UnionBondInput& p_data) const;
            
    // Get the wrapped input.
    const NeuralInputUnary* GetBaseInput() const;

    double GetMin() const;
    double GetMax() const;
    double EvaluateInput(UInt32 input) const;
    bool Save(FILE *fpOutput, size_t nInputId,const IFeatureMap& p_featureMap) const;
    
    // Check objects are equal.
    virtual bool Equal(const NeuralInput* p_input) const;
    
    bool Train(double dblLearningRate, double outputHigh, 
               double outputLow, double dblOutputDelta, 
               UInt32 inputHigh[], UInt32 inputLow[]);

    static NeuralInputUnary *Load(size_t nCacheSize, 
                                  NeuralInputUnary *pChild);

    // Get the size of this object, including internal and external memory 
    // (memory accessed through pointers or objects contain pointers e.g. std::string, std::vector, etc.).
    size_t GetSize() const;
    
private:

    friend class NeuralInputTest;    
};


class NeuralInputTanh : public NeuralInput
{
public:

    // Construct from Bond.
    explicit NeuralInputTanh(const NeuralInputTanhBondData& p_data);
    
    // Fill the correct input name and data field.
    void FillBond(UnionBondInput& p_data) const;
    
    // Fill bond structure of NeuralInputTanh class. 
    void FillBondData(NeuralInputTanhBondData& p_data) const;

    double GetMin() const;
    double GetMax() const;
    double Evaluate(UInt32 input[]) const;
    bool Save(FILE *fpOutput, size_t nInputId, const IFeatureMap& p_featureMap) const;
    bool Train(double dblLearningRate, double outputHigh, 
                       double outputLow, double dblOutputDelta, 
                       UInt32 inputHigh[], UInt32 inputLow[]);

    static NeuralInputTanh *Load(
		DynamicRank::Config& p_config,
        const char *szSection,
        IFeatureMap& p_featureMap);

    // Check objects are equal.
    virtual bool Equal(const NeuralInput* p_input) const;
    
    void GetAllAssociatedFeatures(std::vector<UInt32>& associatedFeaturesList) const;

    // Get the size of this object, including internal and external memory 
    // (memory accessed through pointers or objects contain pointers e.g. std::string, std::vector, etc.).
    size_t GetSize() const;

protected:

    // Get the size of external memory owned by this object.
    size_t GetExternalSize() const;

private:

    NeuralInputTanh();
    
    static const int c_maxInputs=30;

    size_t m_cInputs;
    
    bool m_locked;

    // Index of this input in the input vector.
    UInt32 m_rgId[c_maxInputs];
    
    double m_rgWeights[c_maxInputs];
    
    double m_threshold; 
    
    friend class NeuralInputTest;  
};


class NeuralInputTanhUnary : public NeuralInputUnary
{
public:

    // Construct from Bond.
    explicit NeuralInputTanhUnary(const NeuralInputTanhUnaryBondData& p_data);
    
    // Fill the correct input name and data field.
    void FillBond(UnionBondInput& p_data) const;
    
    // Fill bond structure of NeuralInputTanhUnary class. 
    void FillBondData(NeuralInputTanhUnaryBondData& p_data) const;
    
    double GetMin() const;
    double GetMax() const;
    double EvaluateInput(UInt32 input) const;
    bool Save(FILE *fpOutput, size_t nInputId, const IFeatureMap& p_featureMap) const;
    
    // Check objects are equal.
    virtual bool Equal(const NeuralInput* p_input) const;
    
    bool Train(double dblLearningRate, double outputHigher, 
                       double outputLower, double dblOutputDelta, 
                       UInt32 inputHigh[], UInt32 inputLow[]);

    static NeuralInputTanhUnary *Load(
		DynamicRank::Config& p_config,
        const char *szSection,
        IFeatureMap& p_featureMap);

    // Get the size of this object, including internal and external memory 
    // (memory accessed through pointers or objects contain pointers e.g. std::string, std::vector, etc.).
    size_t GetSize() const;
    
    // Get member variables.
    double GetWeight() const;
    double GetThreshold() const;
    
protected:

    NeuralInputTanhUnary(UInt32 iFeature, double dblWeights, 
                         double dblThreshold, bool fLocked);

    // Get the size of external memory owned by this object.
    size_t GetExternalSize() const;

private:

    bool m_fLocked;

    double m_dblWeights;
    
    double m_dblThreshold;    
    
    friend class NeuralInputTest;      
};


class NeuralInputUseAsFloat : public NeuralInputUnary
{
protected:

    // Default constructor.
    NeuralInputUseAsFloat();
    
    NeuralInputUseAsFloat(UInt32 iFeature);

    // Get the size of external memory owned by this object.
    size_t GetExternalSize() const;

public:

    // Construct from Bond.
    explicit NeuralInputUseAsFloat(const NeuralInputUseAsFloatBondData& p_data);
    
    // Fill the correct input name and data field.
    void FillBond(UnionBondInput& p_data) const;

    // Fill bond structure of NeuralInputUseAsFloat class. 
    void FillBondData(NeuralInputUseAsFloatBondData& p_data) const;
        
    // Copy all members.
    void CopyFrom(const NeuralInputUseAsFloat& p_neuralInputUseAsFloat);    
    
    double EvaluateInput(UInt32 input) const;
    bool Save(FILE *fpOutput, size_t nInputId, const IFeatureMap& p_featureMap) const;
    
    // Check objects are equal.
    virtual bool Equal(const NeuralInput* p_input) const;
    
    double GetMin() const;
    double GetMax() const;


    static NeuralInputUseAsFloat *Load(
		DynamicRank::Config& p_config,
        const char *szSection,
        IFeatureMap& p_featureMap);

    // Get the size of this object, including internal and external memory 
    // (memory accessed through pointers or objects contain pointers e.g. std::string, std::vector, etc.).
    size_t GetSize() const;
 
private:

    friend class NeuralInputTest;
};
}
