#pragma once

#include <basic_types.h>
#include <boost/static_assert.hpp>
#include <ExtractorConfig.h>

namespace FreeForm2
{
    namespace BFB
    {
        // Masks for the first word of a BFB stream definition.
        static const UInt32 c_streamTypeMask = 0x80000000;
        static const UInt32 c_streamTupleCountMask = 0x7FFFFFFF;

        // Values of m_decodeType in the various tuple types.
        static const UInt64 c_decodeSmallTuple = 0;
        static const UInt64 c_decodeLargeTuple = 1;
        static const UInt64 c_decodeExtraLargeTuple = 2;
        static const UInt64 c_decodeMetadataToken = 3;

        // Values of m_bodyBlockFlag in the various tuple types.
        static const UInt64 c_inBothStreams = 0;
        static const UInt64 c_inBodyStream = 1;
        static const UInt64 c_inBodyBlockStream = 2;

#pragma pack(push, 1)
        union SmallTuple
        {
            struct
            {
                UInt16 m_decodeType : 2;
                UInt16 m_bodyBlockFlag : 2;
                UInt16 m_relativeOffset : 5;
                UInt16 m_wordAttribute : 3;
                UInt16 m_candidateID : 2;
                UInt16 m_wordID : 2;
            } m_tuple;
            struct
            {
                UInt16 m_data1;
            } m_data;
        };
        BOOST_STATIC_ASSERT(sizeof(SmallTuple) * 8 == 16);

        // Large tuples are specified as 32-bit integers.
        union LargeTuple
        {
            struct
            {
                UInt32 m_decodeType : 2;
                UInt32 m_bodyBlockFlag : 2;
                UInt32 m_relativeOffset : 14;
                UInt32 m_wordLength : 4;
                UInt32 m_wordAttribute : 3;
                UInt32 m_candidateID : 2;
                UInt32 m_wordID : 5;
            } m_tuple;
            struct
            {
                UInt32 m_data1;
            } m_data;
        };
        BOOST_STATIC_ASSERT(sizeof(LargeTuple) * 8 == 32);

        // Extra large tuples are specified as 48-bit integers.
        union ExtraLargeTuple
        {
            // Because there is no 48-bit integer type, the m_tuple member of
            // ExtraLargeTuple is actually the size of a 64-bit integer, though
            // only the lower 48-bits are used.
            struct 
            {
                UInt64 m_decodeType : 2;
                UInt64 m_bodyBlockFlag : 2;
                UInt64 m_relativeOffset : 19;
                UInt64 m_wordLength : 5;
                UInt64 m_wordAttribute : 3;
                UInt64 m_candidateID : 4;
                UInt64 m_wordID : 5;
                UInt64 m_reserved : 8;
            } m_tuple;
            struct
            {
                UInt32 m_data1;
                UInt16 m_data2;
            } m_data;
        };
        BOOST_STATIC_ASSERT(sizeof(ExtraLargeTuple) * 8 == 64);

        // Metadata tokens currently only act as body block headers.
        struct MetadataToken
        {
            UInt32 m_decodeType : 2;
            UInt32 m_bodyBlockType : 3;
            UInt32 m_bodyBlockLength : 27;
        };
        BOOST_STATIC_ASSERT(sizeof(MetadataToken) * 8 == 32);

        // BFB tuple formats.
        union Tuple
        {
            SmallTuple m_smallTuple;
            LargeTuple m_largeTuple;
            ExtraLargeTuple m_xlargeTuple;
            MetadataToken m_metaTuple;
            UInt64 m_value;
        };
        BOOST_STATIC_ASSERT(sizeof(Tuple) * 8 == 64);

        // BFB InterestingTuple data.
        union InterestingTupleData
        {
            struct
            {
                UInt32 m_tupleID : 3;
                UInt32 m_tupleIndex : 3;
                UInt32 m_firstWord : 4;
                UInt32 m_lastWord : 4;
                UInt32 m_tupleWeight : 18;
            } m_struct;
            UInt32 m_value;
        };
        BOOST_STATIC_ASSERT(sizeof(InterestingTupleData) * 8 == 32);

        // BFB QueryPath data format.
        union QueryPathData
        {
            struct
            {
                UInt32 m_pathIndex : 3;
                UInt32 m_candidate0 : 2;
                UInt32 m_candidate1 : 2;
                UInt32 m_candidate2 : 2;
                UInt32 m_candidate3 : 2;
                UInt32 m_candidate4 : 2;
                UInt32 m_candidate5 : 2;
                UInt32 m_candidate6 : 2;
                UInt32 m_candidate7 : 2;
                UInt32 m_candidate8 : 2;
                UInt32 m_candidate9 : 2;
                UInt32 m_pathWeight : 9;
            } m_struct;
            UInt32 m_value;
        };
        BOOST_STATIC_ASSERT(sizeof(QueryPathData) * 8 == 32);

        // Phrase normalize: a fixed-point 48-bit decimal number.
        union PhraseNormalizer
        {
            UInt64 m_value;
            struct
            {
                UInt32 m_value1;
                UInt16 m_value2;
            } m_data;
        };
        BOOST_STATIC_ASSERT(sizeof(PhraseNormalizer) * 8 == 64);

        // The number of UInt32s required to hold the configuration bitmask.
        static const size_t c_numberOfConfigBlocks 
            = (ExtractorConfig::ExtractorConfigCount - 1) / (sizeof(UInt32) * 8) + 1;

        struct PerStreamFeatures
        {
            UInt32 m_wordFound;
            UInt32 m_wordsFound;
            UInt32 m_bm25f;
            UInt32 m_bm25fNorm;
            UInt32 m_originalQueryBM25F;
            UInt32 m_originalQueryBM25FNorm;
            UInt32 m_proxBM25F;
            UInt32 m_proxBM25FNorm;
            UInt32 m_perStreamLMScore;
            UInt32 m_parametersPresent[c_numberOfConfigBlocks];
        };
        BOOST_STATIC_ASSERT(sizeof(PerStreamFeatures) == sizeof(UInt32) * (9 + c_numberOfConfigBlocks));

        union QueryDataBits
        {
            UInt32 m_value;
            struct
            {
                UInt32 m_extractWordCandidateDate : 1;
                UInt32 m_termWeightEnabled : 1;
                UInt32 m_alterationWeightEnabled : 1;
                UInt32 m_calculationMethod : 2;
                UInt32 m_newFeatureFlags : 5;
                UInt32 m_pbsTupleTypes : 6;
                UInt32 m_unused : 16;
            } m_data;
        };
        static_assert(sizeof(QueryDataBits) == sizeof(UInt32), "QueryDataBits has bad size");

        // Custom data structure to hold the max size for each feature
        struct FeatureDefinition
        {
            UInt32 m_nameIndex;
            unsigned char m_size[4];
        };
#pragma pack(pop)
    }
}