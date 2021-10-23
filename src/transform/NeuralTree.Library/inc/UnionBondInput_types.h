
//------------------------------------------------------------------------------
// <auto-generated>
//     This code was generated by a tool.
// 
//     Tool     : bondc, Version=3.0.1, Build=bond-git.retail.0
//     Template : Microsoft.Bond.Rules.dll#Rules_BOND_CPP.tt
//     File     : UnionBondInput_types.h
//
//     Changes to this file may cause incorrect behavior and will be lost when
//     the code is regenerated.
// </auto-generated>
//------------------------------------------------------------------------------

#pragma once


#include <bond/core/bond_version.h>
#if BOND_MAJOR_VERSION_MIN_SUPPORTED > 3 \
    || (BOND_MAJOR_VERSION_MIN_SUPPORTED == 3 && BOND_MINOR_VERSION_MIN_SUPPORTED > 1)
#error This file was generated by an older Bond compiler which is \
       incompatible with current Bond library. Please regenerate \
       with the latest Bond compiler.
#endif

#include <bond/core/config.h>
#include <bond/core/containers.h>
#include <bond/core/nullable.h>
#include "NeuralInput_types.h"
//#include "NeuralInputBSpline_types.h"
//#include "NeuralInputFreeForm_types.h"
//#include "NeuralInputTree_types.h"
//#include "NeuralInputFreeForm_types.h"
//#include "NeuralInputMultiple_types.h
//#include "NeuralInputBM252_types.h"
//#include "NeuralInputNgramBM25_types.h"
#include "NeuralInputFreeForm2_types.h"

namespace DynamicRank
{

// UnionBondInput 
struct UnionBondInput 
{
    // 1: optional string m_inputType
    // Used to decide which input is. linear loglinear rational bspline bsplinebasis bucket decisiontree tanh freeform floatdata aggregatedfreeform   <---Note: Research this new case. Could be evil. sumbucket sumcomparisons sumbucketcomparison sumlinear sumloglinear sumgreater sumdivisor bm25f2 logbm25f2 ngrambm25f logngrambm25f perfbm25f perflogbm25f freeform2 <--- This input factory will be in dynamicranker.freeform.library. bulkinput <-- For bulky compiled input.
    std::string m_inputType;

    // 2: optional nullable<DynamicRank.NeuralInputLinearBondData> m_linear
    bond::nullable< ::DynamicRank::NeuralInputLinearBondData> m_linear;

    // 3: optional nullable<DynamicRank.NeuralInputLogLinearBondData> m_loglinear
    bond::nullable< ::DynamicRank::NeuralInputLogLinearBondData> m_loglinear;

    // 4: optional nullable<DynamicRank.NeuralInputRationalBondData> m_rational
    bond::nullable< ::DynamicRank::NeuralInputRationalBondData> m_rational;

    //// 5: optional nullable<DynamicRank.NeuralInputBSplineBondData> m_bspline
    //bond::nullable< ::DynamicRank::NeuralInputBSplineBondData> m_bspline;

    //// 6: optional nullable<DynamicRank.NeuralInputBSplineBasisFunctionBondData> m_bsplinebasis
    //bond::nullable< ::DynamicRank::NeuralInputBSplineBasisFunctionBondData> m_bsplinebasis;

    // 7: optional nullable<DynamicRank.NeuralInputBucketBondData> m_bucket
    bond::nullable< ::DynamicRank::NeuralInputBucketBondData> m_bucket;

    //// 8: optional nullable<DynamicRank.NeuralInputTreeBondData> m_decisiontree
    //bond::nullable< ::DynamicRank::NeuralInputTreeBondData> m_decisiontree;

    // 10: optional nullable<DynamicRank.UnionNeuralInputTanhBondData> m_tanh
    bond::nullable< ::DynamicRank::UnionNeuralInputTanhBondData> m_tanh;

    //// 11: optional nullable<DynamicRank.NeuralInputFreeFormBondData> m_freeform
    //bond::nullable< ::DynamicRank::NeuralInputFreeFormBondData> m_freeform;

    // 12: optional nullable<DynamicRank.NeuralInputUseAsFloatBondData> m_floatdata
    bond::nullable< ::DynamicRank::NeuralInputUseAsFloatBondData> m_floatdata;

    //// 13: optional nullable<DynamicRank.NeuralInputFreeFormBondData> m_aggregatedfreeform
    //bond::nullable< ::DynamicRank::NeuralInputFreeFormBondData> m_aggregatedfreeform;

    //// 14: optional nullable<DynamicRank.NeuralInputSumBucketBondData> m_sumbucket
    //bond::nullable< ::DynamicRank::NeuralInputSumBucketBondData> m_sumbucket;

    //// 16: optional nullable<DynamicRank.NeuralInputSumComparisonsBondData> m_sumcomparisons
    //bond::nullable< ::DynamicRank::NeuralInputSumComparisonsBondData> m_sumcomparisons;

    //// 17: optional nullable<DynamicRank.NeuralInputSumBucketComparisonBondData> m_sumbucketcomparisons
    //bond::nullable< ::DynamicRank::NeuralInputSumBucketComparisonBondData> m_sumbucketcomparisons;

    //// 18: optional nullable<DynamicRank.NeuralInputSumLinearBondData> m_sumlinear
    //bond::nullable< ::DynamicRank::NeuralInputSumLinearBondData> m_sumlinear;

    //// 19: optional nullable<DynamicRank.NeuralInputSumLogLinearBondData> m_sumloglinear
    //bond::nullable< ::DynamicRank::NeuralInputSumLogLinearBondData> m_sumloglinear;

    //// 20: optional nullable<DynamicRank.NeuralInputSumGreaterBondData> m_sumgreater
    //bond::nullable< ::DynamicRank::NeuralInputSumGreaterBondData> m_sumgreater;

    //// 21: optional nullable<DynamicRank.NeuralInputSumDivisorBondData> m_sumdivisor
    //bond::nullable< ::DynamicRank::NeuralInputSumDivisorBondData> m_sumdivisor;

    //// 22: optional nullable<DynamicRank.NeuralInputLinearBM25BondData> m_bm25f2
    //bond::nullable< ::DynamicRank::NeuralInputLinearBM25BondData> m_bm25f2;

    //// 23: optional nullable<DynamicRank.NeuralInputLogLinearBM25BondData> m_logbm25f2
    //bond::nullable< ::DynamicRank::NeuralInputLogLinearBM25BondData> m_logbm25f2;

    //// 24: optional nullable<DynamicRank.NeuralInputLinearNgramBM25BondData> m_ngrambm25f
    //bond::nullable< ::DynamicRank::NeuralInputLinearNgramBM25BondData> m_ngrambm25f;

    //// 25: optional nullable<DynamicRank.NeuralInputLogLinearNgramBM25BondData> m_logngrambm25f
    //bond::nullable< ::DynamicRank::NeuralInputLogLinearNgramBM25BondData> m_logngrambm25f;

    //// 26: optional nullable<DynamicRank.NeuralInputLinearBM25BondData> m_perfbm25f
    //bond::nullable< ::DynamicRank::NeuralInputLinearBM25BondData> m_perfbm25f;

    //// 27: optional nullable<DynamicRank.NeuralInputLogLinearBM25BondData> m_perflogbm25f
    //bond::nullable< ::DynamicRank::NeuralInputLogLinearBM25BondData> m_perflogbm25f;

    // 28: optional nullable<DynamicRank.NeuralInputFreeForm2BondData> m_freeform2
    bond::nullable< ::DynamicRank::NeuralInputFreeForm2BondData> m_freeform2;

    // 29: optional nullable<DynamicRank.BulkNeuralInputBondData> m_bulkinput
    bond::nullable< ::DynamicRank::BulkNeuralInputBondData> m_bulkinput;

    UnionBondInput()
    {
    }


    // Compiler generated copy ctor OK
#ifndef BOND_NO_CXX11_DEFAULTED_FUNCTIONS
    UnionBondInput(const UnionBondInput& /*_bond_rhs*/) = default;
#endif


#ifndef BOND_NO_CXX11_RVALUE_REFERENCES
    UnionBondInput(UnionBondInput&& _bond_rhs) BOND_NOEXCEPT_IF((true
        && std::is_nothrow_move_constructible< std::string >::value
        && std::is_nothrow_move_constructible< bond::nullable< ::DynamicRank::NeuralInputLinearBondData> >::value
        && std::is_nothrow_move_constructible< bond::nullable< ::DynamicRank::NeuralInputLogLinearBondData> >::value
        && std::is_nothrow_move_constructible< bond::nullable< ::DynamicRank::NeuralInputRationalBondData> >::value
        && std::is_nothrow_move_constructible< bond::nullable< ::DynamicRank::NeuralInputBucketBondData> >::value
        && std::is_nothrow_move_constructible< bond::nullable< ::DynamicRank::UnionNeuralInputTanhBondData> >::value
        && std::is_nothrow_move_constructible< bond::nullable< ::DynamicRank::NeuralInputUseAsFloatBondData> >::value
        && std::is_nothrow_move_constructible< bond::nullable< ::DynamicRank::NeuralInputFreeForm2BondData> >::value
        && std::is_nothrow_move_constructible< bond::nullable< ::DynamicRank::BulkNeuralInputBondData> >::value
        ))
        : m_inputType(std::move(_bond_rhs.m_inputType)),
          m_linear(std::move(_bond_rhs.m_linear)),
          m_loglinear(std::move(_bond_rhs.m_loglinear)),
          m_rational(std::move(_bond_rhs.m_rational)),
          m_bucket(std::move(_bond_rhs.m_bucket)),
          m_tanh(std::move(_bond_rhs.m_tanh)),
          m_floatdata(std::move(_bond_rhs.m_floatdata)),
          m_freeform2(std::move(_bond_rhs.m_freeform2)),
          m_bulkinput(std::move(_bond_rhs.m_bulkinput))
    {
    }
#endif


    template<typename Allocator>
    explicit
    UnionBondInput(Allocator* _bond_allocator)
        : m_inputType(*_bond_allocator),
          m_linear(*_bond_allocator),
          m_loglinear(*_bond_allocator),
          m_rational(*_bond_allocator),
          m_bucket(*_bond_allocator),
          m_tanh(*_bond_allocator),
          m_floatdata(*_bond_allocator),
          m_freeform2(*_bond_allocator),
          m_bulkinput(*_bond_allocator)
    {
    }


    // Compiler generated operator= OK
#ifndef BOND_NO_CXX11_DEFAULTED_FUNCTIONS
    UnionBondInput& operator=(const UnionBondInput& _bond_rhs) = default;
#endif


#ifndef BOND_NO_CXX11_RVALUE_REFERENCES
    UnionBondInput& operator=(UnionBondInput&& _bond_rhs)
    {
        UnionBondInput(std::move(_bond_rhs)).swap(*this);
        return *this;
    }
#endif


    bool operator==(const UnionBondInput& _bond_other) const
    {
        return true
            && (m_inputType == _bond_other.m_inputType)
            && (m_linear == _bond_other.m_linear)
            && (m_loglinear == _bond_other.m_loglinear)
            && (m_rational == _bond_other.m_rational)
            && (m_bucket == _bond_other.m_bucket)
            && (m_tanh == _bond_other.m_tanh)
            && (m_floatdata == _bond_other.m_floatdata)
            && (m_freeform2 == _bond_other.m_freeform2)
            && (m_bulkinput == _bond_other.m_bulkinput);
    }


    bool operator!=(const UnionBondInput& _bond_other) const
    {
        return !(*this == _bond_other);
    }


    void swap(UnionBondInput& _bond_other)
    {
        using std::swap;
        swap(m_inputType, _bond_other.m_inputType);
        swap(m_linear, _bond_other.m_linear);
        swap(m_loglinear, _bond_other.m_loglinear);
        swap(m_rational, _bond_other.m_rational);
        swap(m_bucket, _bond_other.m_bucket);
        swap(m_tanh, _bond_other.m_tanh);
        swap(m_floatdata, _bond_other.m_floatdata);
        swap(m_freeform2, _bond_other.m_freeform2);
        swap(m_bulkinput, _bond_other.m_bulkinput);
    }


    struct Schema;


protected:
    void InitMetadata(const char* /*_bond_name*/, const char* /*_bond_full_name*/)
    {
    }
};


inline void swap(UnionBondInput& _bond_left, UnionBondInput& _bond_right)
{
    _bond_left.swap(_bond_right);
}
} // namespace DynamicRank
