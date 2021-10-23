#pragma once

#ifndef FREEFORM2_INC_FEATURES_H
#define FREEFORM2_INC_FEATURES_H

#include <string>

namespace FreeForm2
{
    class TypeImpl;

    // The feature information struct provides a namespace for feature-related
    // data declarations.
    struct FeatureInformation
    {
        // This enum lists the type of features supported by the feature 
        // compiler.
        enum FeatureType
        {
            MetaStreamFeature,
            DerivedFeature,
            AggregatedDerivedFeature,
            AbInitioFeature
        };
    };

    // This namespace declares the names of required external data members for
    // metastream features.
    namespace RequiredMetaStreamData
    {
        // The number of query paths in the current query.
        struct NumQueryPaths
        {
            static const std::string& GetName();
            static const TypeImpl& GetType();
        };

        // The number of words in the query.
        struct QueryLength
        {
            static const std::string& GetName();
            static const TypeImpl& GetType();
        };

        // The index of word candidates per term in a specific query path.
        struct QueryPathCandidates
        {
            static const std::string& GetName();
        };

        // The stream data over which a metastream feature operates.
        struct Stream
        {
            static const std::string& GetName();
            static const TypeImpl& GetType();
        };

        // The number of tuples of interest per type.
        struct TupleOfInterestCount
        {
            static const std::string& GetName();
        };

        // The tuples of interest.
        struct TuplesOfInterest
        {
            static const std::string& GetName();
        };

        // Duplicate term information.
        struct UnsafeDuplicateTermInformation
        {
            static const std::string& GetName();
        };
    }

    // This namespace declares the external data members required for 
    // compilation of derived features.
    namespace RequiredDerivedFeatureData
    {
        // The data member representing the stream ID.
        struct StreamID
        {
            static const std::string& GetName();
            static const TypeImpl& GetType();
        };
    }
}

#endif
