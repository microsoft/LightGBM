#pragma once

#ifndef FREEFORM2_UTILS_H
#define FREEFORM2_UTILS_H

#include <boost/noncopyable.hpp>
#include <basic_types.h>
#include "FreeForm2.h"
#include <INeuralNetFeatures.h>
#include <set>
#include <vector>
#include <iosfwd>

namespace DynamicRank
{
    class IFeatureMap;
}

namespace FreeForm2
{
    // Utility class that populates a vector of features used using the
    // INeuralNetFeatures interface.
    class VectorFromNeuralNetFeatures 
        : public DynamicRank::INeuralNetFeatures, boost::noncopyable
    {
    public:
        VectorFromNeuralNetFeatures(std::vector<UInt32>& p_associatedFeaturesList)
            : m_associatedFeaturesList(p_associatedFeaturesList)
        {
        }


        virtual void ProcessFeature(UInt32 p_featureIndex) override;

        virtual void ProcessFeature(UInt32 p_featureIndex, const std::vector<std::string>& p_segments) override;

    private:
        std::vector<UInt32>& m_associatedFeaturesList;
    };


    // Utility class that populates a vector of features used using the
    // INeuralNetFeatures interface.
    class SetFromNeuralNetFeatures 
        : public DynamicRank::INeuralNetFeatures, boost::noncopyable
    {
    public:
        SetFromNeuralNetFeatures(std::set<UInt32>& p_associatedFeaturesList)
            : m_associatedFeaturesList(p_associatedFeaturesList)
        {
        }


        virtual void ProcessFeature(UInt32 p_featureIndex) override;

        virtual void ProcessFeature(UInt32 p_featureIndex, const std::vector<std::string>& p_segments) override;

    private:
        std::set<UInt32>& m_associatedFeaturesList;
    };


    // Print a SIZED_STRING to an output stream.
    std::ostream& operator<<(std::ostream& p_out, SIZED_STRING p_str);

    // Log errors after a crash, suitable for use as a structured exception
    // handling test.  Note that this function always returns false, as it does
    // not handle exceptions, it simply logs information regarding that exception.
    void LogHardwareException(DWORD p_exceptionCode, 
                              const Executable::FeatureType p_features[], 
                              const DynamicRank::IFeatureMap& p_map, 
                              const char* p_sourceFile, 
                              unsigned int p_sourceLine);

    // Indicates whether the given name is composed of alphanumeric
    // characters and '-' only.
    bool IsSimpleName(SIZED_STRING p_name);

    // Write a sparse vector to the output stream. This algorithm uses a simple
    // run-length encoding to compress ranges of 0's.
    void WriteCompressedVectorRLE(const UInt32* p_data, size_t p_numElements, std::ostream& p_out);

    // Read a sparse integer vector incoded using the above method.
    void ReadCompressedVectorRLE(UInt32* p_data, size_t p_numElements, std::istream& p_in);
}

#endif

