// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2020 Everton Constantino <everton.constantino@ibm.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/

#include "main.h"

#include <Eigen/Core>

using namespace Eigen;

#define GET(i,j) (StorageOrder == RowMajor ? (i)*stride + (j) : (i) + (j)*stride)
#define SCATTER(i,j,k) (StorageOrder == RowMajor ? ((i)+(k))*stride + (j) : (i) + ((j)+(k))*stride)

template<typename Scalar, typename Packet>
void compare(const Packet& a, const Packet& b)
{
    int pktsz = internal::packet_traits<Scalar>::size;
    Scalar *buffA = new Scalar[pktsz];
    Scalar *buffB = new Scalar[pktsz];

    internal::pstoreu<Scalar, Packet>(buffA, a);
    internal::pstoreu<Scalar, Packet>(buffB, b);

    for(int i = 0; i < pktsz; i++)
    {
        VERIFY_IS_EQUAL(buffA[i], buffB[i]);
    }

    delete[] buffA;
    delete[] buffB;
}

template<typename Scalar, int StorageOrder, int n>
struct PacketBlockSet
{
    typedef typename internal::packet_traits<Scalar>::type Packet;

    void setPacketBlock(internal::PacketBlock<Packet,n>& block, Scalar value)
    {
        for(int idx = 0; idx < n; idx++)
        {
            block.packet[idx] = internal::pset1<Packet>(value);
        }
    }

    void comparePacketBlock(Scalar *data, int i, int j, int stride, internal::PacketBlock<Packet, n>& block)
    {
        for(int idx = 0; idx < n; idx++)
        {
            Packet line = internal::ploadu<Packet>(data + SCATTER(i,j,idx));
            compare<Scalar, Packet>(block.packet[idx], line);
        }
    }
};

template<typename Scalar, int StorageOrder, int BlockSize>
void run_bdmp_spec_1()
{
    typedef internal::blas_data_mapper<Scalar, int, StorageOrder> BlasDataMapper;
    int packetSize = internal::packet_traits<Scalar>::size;
    int minSize = std::max<int>(packetSize, BlockSize);
    typedef typename internal::packet_traits<Scalar>::type Packet;

    int szm = internal::random<int>(minSize,500), szn = internal::random<int>(minSize,500);
    int stride = StorageOrder == RowMajor ? szn : szm;
    Scalar *d = new Scalar[szn*szm];

    // Initializing with random entries
    for(int i = 0; i < szm*szn; i++)
    {
        d[i] = internal::random<Scalar>(static_cast<Scalar>(3), static_cast<Scalar>(10));
    }

    BlasDataMapper bdm(d, stride);

    // Testing operator()
    for(int i = 0; i < szm; i++)
    {
        for(int j = 0; j < szn; j++)
        {
            VERIFY_IS_EQUAL(d[GET(i,j)], bdm(i,j));
        }
    }

    // Testing getSubMapper and getLinearMapper
    int i0 = internal::random<int>(0,szm-2);
    int j0 = internal::random<int>(0,szn-2);
    for(int i = i0; i < szm; i++)
    {
        for(int j = j0; j < szn; j++)
        {
            const BlasDataMapper& bdmSM = bdm.getSubMapper(i0,j0);
            const internal::BlasLinearMapper<Scalar, int, 0>& bdmLM = bdm.getLinearMapper(i0,j0);

            Scalar v = bdmSM(i - i0, j - j0);
            Scalar vd = d[GET(i,j)];
            VERIFY_IS_EQUAL(vd, v);
            VERIFY_IS_EQUAL(vd, bdmLM(GET(i-i0, j-j0)));
        }
    }

    // Testing loadPacket
    for(int i = 0; i < szm - minSize; i++)
    {
        for(int j = 0; j < szn - minSize; j++)
        {
            Packet pktBDM = bdm.template loadPacket<Packet>(i,j);
            Packet pktD = internal::ploadu<Packet>(d + GET(i,j));

            compare<Scalar, Packet>(pktBDM, pktD);
        }
    }

    // Testing gatherPacket
    Scalar *buff = new Scalar[packetSize];
    for(int i = 0; i < szm - minSize; i++)
    {
        for(int j = 0; j < szn - minSize; j++)
        {
            Packet p = bdm.template gatherPacket<Packet>(i,j);
            internal::pstoreu<Scalar, Packet>(buff, p);

            for(int k = 0; k < packetSize; k++)
            {
                VERIFY_IS_EQUAL(d[SCATTER(i,j,k)], buff[k]);
            }

        }
    }
    delete[] buff;

    // Testing scatterPacket
    for(int i = 0; i < szm - minSize; i++)
    {
        for(int j = 0; j < szn - minSize; j++)
        {
            Packet p = internal::pset1<Packet>(static_cast<Scalar>(1));
            bdm.template scatterPacket<Packet>(i,j,p);
            for(int k = 0; k < packetSize; k++)
            {
                VERIFY_IS_EQUAL(d[SCATTER(i,j,k)], static_cast<Scalar>(1));
            }
        }
    }

    //Testing storePacketBlock
    internal::PacketBlock<Packet, BlockSize> block;

    PacketBlockSet<Scalar, StorageOrder, BlockSize> pbs;
    pbs.setPacketBlock(block, static_cast<Scalar>(2));

    for(int i = 0; i < szm - minSize; i++)
    {
        for(int j = 0; j < szn - minSize; j++)
        {
            bdm.template storePacketBlock<Packet, BlockSize>(i, j, block);

            pbs.comparePacketBlock(d, i, j, stride, block);
        }
    }

    delete[] d;
}

template<typename Scalar>
void run_test()
{
    run_bdmp_spec_1<Scalar, RowMajor, 1>();
    run_bdmp_spec_1<Scalar, ColMajor, 1>();
    run_bdmp_spec_1<Scalar, RowMajor, 2>();
    run_bdmp_spec_1<Scalar, ColMajor, 2>();
    run_bdmp_spec_1<Scalar, RowMajor, 4>();
    run_bdmp_spec_1<Scalar, ColMajor, 4>();
    run_bdmp_spec_1<Scalar, RowMajor, 8>();
    run_bdmp_spec_1<Scalar, ColMajor, 8>();
    run_bdmp_spec_1<Scalar, RowMajor, 16>();
    run_bdmp_spec_1<Scalar, ColMajor, 16>();
}

EIGEN_DECLARE_TEST(blasutil)
{
    for(int i = 0; i < g_repeat; i++)
    {
        CALL_SUBTEST_1(run_test<int8_t>());
        CALL_SUBTEST_2(run_test<int16_t>());
        CALL_SUBTEST_3(run_test<int32_t>());
        CALL_SUBTEST_4(run_test<int64_t>());
        CALL_SUBTEST_5(run_test<float_t>());
        CALL_SUBTEST_6(run_test<double_t>());
    }
}