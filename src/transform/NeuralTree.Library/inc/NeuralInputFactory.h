#pragma once

#include <boost/shared_ptr.hpp>
#include "NeuralInput.h"
#include <map>
#include <boost/utility.hpp>

namespace DynamicRank
{

class BulkNeuralInput;
class NeuralNet;
class NeuralInput;
class IFeatureMap;
struct UnionBondInput;

// NeuralInputFactory loads neural inputs from config files using a 
// registration paradigm, in order to support different types of inputs for
// different situations.
class NeuralInputFactory : private boost::noncopyable
{
public:
    // Shortcut for configuration type.
    typedef DynamicRank::Config IConfiguration;

    // Class that knows how to load a particular transform.
    class Loader
    {
    public:
        // Shortcut definition for shared pointer of this type.
        typedef boost::shared_ptr<Loader> Ptr;

        virtual ~Loader();

        // Functor to create a NeuralInput given appropriate inputs.
        virtual NeuralInput* operator()(IConfiguration& p_config, 
                                        const char* p_section,
                                        IFeatureMap& p_featureMap) const = 0;

        // Use for all NeuralInputs.
        virtual NeuralInput* FromBond(const UnionBondInput& p_data) const = 0;

        // Used for BulkNeuralInput.
        //virtual BulkNeuralInput* FromBulkBond(const UnionBondInput& p_data) const = 0;
    };


    // Create a neural input factory that knows how to create a variety of NeuralInput classes.
    NeuralInputFactory();

    virtual ~NeuralInputFactory();

    typedef NeuralInput* (*LoadFunction)(IConfiguration& p_config, 
                                         const char* p_section,
                                         IFeatureMap& p_featureMap);


    // Add registration for a particular transform, throwing exceptions if
    // registration fails, or if p_loader is NULL.
    void AddTransform(const char* p_transform, LoadFunction p_loader, bool p_replace = false);

    // Add registration for a particular transform, throwing exceptions if
    // registration fails, or if p_loader is NULL.
    void AddTransform(const char* p_transform, Loader::Ptr p_loader, bool p_replace = false);

    NeuralInput* FromBond(const UnionBondInput& p_data) const;

    //BulkNeuralInput* FromBulkBond(const UnionBondInput& p_data) const;

    // Remove all transforms from the neural input factory.
    void ClearTransforms();

    // Load a transform, returning NULL if loading fails.
    NeuralInput* Load(const char* p_transform, 
                      IConfiguration& p_config, 
                      const char* p_section,
                      IFeatureMap& p_featureMap) const;

    // Read the transform name for an input and load using Load function above.
    NeuralInput* Load(IConfiguration& p_config, 
                      int p_ID,
                      IFeatureMap& p_featureMap) const;

    // Small template class to adapt functions returning subtypes of 
    // NeuralInput to return NeuralInput.
    template<class InputType, 
             InputType* (*fun)(IConfiguration& p_config, 
                               const char* p_section,
                               IFeatureMap& p_featureMap)>
    static NeuralInput* LoadAdapt(IConfiguration& p_config, 
                                  const char* p_section,
                                  IFeatureMap& p_featureMap)
    {
        return fun(p_config, p_section, p_featureMap);
    }

private:

    // Map of neural input transform names to loading functions.
    typedef std::map<std::string, Loader::Ptr> TransformMap;
    TransformMap m_transform;
};

// BulkNeuralInputFactory converts a set of NeuralInput objects into a
// BulkNeuralInput that produces the same effect as evaluating all the
// inputs separately.
class BulkNeuralInputFactory : public boost::noncopyable
{
public:
    // A mapping from NeuralInput to offset in the output array.
    typedef std::pair<NeuralInput*, size_t> InputAndIndex;

    // Create an instance of BulkNeuralInput from a list of NeuralInput objects and their
    // offset in the output array. Throws an exception if at least one of the NeuralInput
    // objects does not support bulk optimization.
    virtual std::unique_ptr<BulkNeuralInput> ConvertToBulkInput(const std::vector<InputAndIndex>& p_inputs,
                                                                IFeatureMap& p_featureMap) const = 0;

};

}
