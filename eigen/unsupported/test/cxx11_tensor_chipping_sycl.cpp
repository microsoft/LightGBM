// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016
// Mehdi Goli    Codeplay Software Ltd.
// Ralph Potter  Codeplay Software Ltd.
// Luke Iwanski  Codeplay Software Ltd.
// Contact: <eigen@codeplay.com>
// Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_NO_COMPLEX

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int64_t
#define EIGEN_USE_SYCL

#include "main.h"

#include <Eigen/CXX11/Tensor>

using Eigen::Tensor;

template <typename DataType, int DataLayout, typename IndexType>
static void test_static_chip_sycl(const Eigen::SyclDevice& sycl_device)
{
  IndexType sizeDim1 = 2;
  IndexType sizeDim2 = 3;
  IndexType sizeDim3 = 5;
  IndexType sizeDim4 = 7;
  IndexType sizeDim5 = 11;

  array<IndexType, 5> tensorRange = {{sizeDim1, sizeDim2, sizeDim3, sizeDim4, sizeDim5}};
  array<IndexType, 4> chip1TensorRange = {{sizeDim2, sizeDim3, sizeDim4, sizeDim5}};

  Tensor<DataType, 5, DataLayout,IndexType> tensor(tensorRange);
  Tensor<DataType, 4, DataLayout,IndexType> chip1(chip1TensorRange);

  tensor.setRandom();

  const size_t tensorBuffSize =tensor.size()*sizeof(DataType);
  const size_t chip1TensorBuffSize =chip1.size()*sizeof(DataType);
  DataType* gpu_data_tensor  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));
  DataType* gpu_data_chip1  = static_cast<DataType*>(sycl_device.allocate(chip1TensorBuffSize));

  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_tensor(gpu_data_tensor, tensorRange);
  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_chip1(gpu_data_chip1, chip1TensorRange);

  sycl_device.memcpyHostToDevice(gpu_data_tensor, tensor.data(), tensorBuffSize);
  gpu_chip1.device(sycl_device)=gpu_tensor.template chip<0l>(1l);
  sycl_device.memcpyDeviceToHost(chip1.data(), gpu_data_chip1, chip1TensorBuffSize);

  VERIFY_IS_EQUAL(chip1.dimension(0), sizeDim2);
  VERIFY_IS_EQUAL(chip1.dimension(1), sizeDim3);
  VERIFY_IS_EQUAL(chip1.dimension(2), sizeDim4);
  VERIFY_IS_EQUAL(chip1.dimension(3), sizeDim5);

  for (IndexType i = 0; i < sizeDim2; ++i) {
    for (IndexType j = 0; j < sizeDim3; ++j) {
      for (IndexType k = 0; k < sizeDim4; ++k) {
        for (IndexType l = 0; l < sizeDim5; ++l) {
          VERIFY_IS_EQUAL(chip1(i,j,k,l), tensor(1l,i,j,k,l));
        }
      }
    }
  }

  array<IndexType, 4> chip2TensorRange = {{sizeDim1, sizeDim3, sizeDim4, sizeDim5}};
  Tensor<DataType, 4, DataLayout,IndexType> chip2(chip2TensorRange);
  const size_t chip2TensorBuffSize =chip2.size()*sizeof(DataType);
  DataType* gpu_data_chip2  = static_cast<DataType*>(sycl_device.allocate(chip2TensorBuffSize));
  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_chip2(gpu_data_chip2, chip2TensorRange);

  gpu_chip2.device(sycl_device)=gpu_tensor.template chip<1l>(1l);
  sycl_device.memcpyDeviceToHost(chip2.data(), gpu_data_chip2, chip2TensorBuffSize);

  VERIFY_IS_EQUAL(chip2.dimension(0), sizeDim1);
  VERIFY_IS_EQUAL(chip2.dimension(1), sizeDim3);
  VERIFY_IS_EQUAL(chip2.dimension(2), sizeDim4);
  VERIFY_IS_EQUAL(chip2.dimension(3), sizeDim5);

  for (IndexType i = 0; i < sizeDim1; ++i) {
    for (IndexType j = 0; j < sizeDim3; ++j) {
      for (IndexType k = 0; k < sizeDim4; ++k) {
        for (IndexType l = 0; l < sizeDim5; ++l) {
          VERIFY_IS_EQUAL(chip2(i,j,k,l), tensor(i,1l,j,k,l));
        }
      }
    }
  }

  array<IndexType, 4> chip3TensorRange = {{sizeDim1, sizeDim2, sizeDim4, sizeDim5}};
  Tensor<DataType, 4, DataLayout,IndexType> chip3(chip3TensorRange);
  const size_t chip3TensorBuffSize =chip3.size()*sizeof(DataType);
  DataType* gpu_data_chip3  = static_cast<DataType*>(sycl_device.allocate(chip3TensorBuffSize));
  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_chip3(gpu_data_chip3, chip3TensorRange);

  gpu_chip3.device(sycl_device)=gpu_tensor.template chip<2l>(2l);
  sycl_device.memcpyDeviceToHost(chip3.data(), gpu_data_chip3, chip3TensorBuffSize);

  VERIFY_IS_EQUAL(chip3.dimension(0), sizeDim1);
  VERIFY_IS_EQUAL(chip3.dimension(1), sizeDim2);
  VERIFY_IS_EQUAL(chip3.dimension(2), sizeDim4);
  VERIFY_IS_EQUAL(chip3.dimension(3), sizeDim5);

  for (IndexType i = 0; i < sizeDim1; ++i) {
    for (IndexType j = 0; j < sizeDim2; ++j) {
      for (IndexType k = 0; k < sizeDim4; ++k) {
        for (IndexType l = 0; l < sizeDim5; ++l) {
          VERIFY_IS_EQUAL(chip3(i,j,k,l), tensor(i,j,2l,k,l));
        }
      }
    }
  }

  array<IndexType, 4> chip4TensorRange = {{sizeDim1, sizeDim2, sizeDim3, sizeDim5}};
  Tensor<DataType, 4, DataLayout,IndexType> chip4(chip4TensorRange);
  const size_t chip4TensorBuffSize =chip4.size()*sizeof(DataType);
  DataType* gpu_data_chip4  = static_cast<DataType*>(sycl_device.allocate(chip4TensorBuffSize));
  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_chip4(gpu_data_chip4, chip4TensorRange);

  gpu_chip4.device(sycl_device)=gpu_tensor.template chip<3l>(5l);
  sycl_device.memcpyDeviceToHost(chip4.data(), gpu_data_chip4, chip4TensorBuffSize);

  VERIFY_IS_EQUAL(chip4.dimension(0), sizeDim1);
  VERIFY_IS_EQUAL(chip4.dimension(1), sizeDim2);
  VERIFY_IS_EQUAL(chip4.dimension(2), sizeDim3);
  VERIFY_IS_EQUAL(chip4.dimension(3), sizeDim5);

  for (IndexType i = 0; i < sizeDim1; ++i) {
    for (IndexType j = 0; j < sizeDim2; ++j) {
      for (IndexType k = 0; k < sizeDim3; ++k) {
        for (IndexType l = 0; l < sizeDim5; ++l) {
          VERIFY_IS_EQUAL(chip4(i,j,k,l), tensor(i,j,k,5l,l));
        }
      }
    }
  }


  array<IndexType, 4> chip5TensorRange = {{sizeDim1, sizeDim2, sizeDim3, sizeDim4}};
  Tensor<DataType, 4, DataLayout,IndexType> chip5(chip5TensorRange);
  const size_t chip5TensorBuffSize =chip5.size()*sizeof(DataType);
  DataType* gpu_data_chip5  = static_cast<DataType*>(sycl_device.allocate(chip5TensorBuffSize));
  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_chip5(gpu_data_chip5, chip5TensorRange);

  gpu_chip5.device(sycl_device)=gpu_tensor.template chip<4l>(7l);
  sycl_device.memcpyDeviceToHost(chip5.data(), gpu_data_chip5, chip5TensorBuffSize);

  VERIFY_IS_EQUAL(chip5.dimension(0), sizeDim1);
  VERIFY_IS_EQUAL(chip5.dimension(1), sizeDim2);
  VERIFY_IS_EQUAL(chip5.dimension(2), sizeDim3);
  VERIFY_IS_EQUAL(chip5.dimension(3), sizeDim4);

  for (IndexType i = 0; i < sizeDim1; ++i) {
    for (IndexType j = 0; j < sizeDim2; ++j) {
      for (IndexType k = 0; k < sizeDim3; ++k) {
        for (IndexType l = 0; l < sizeDim4; ++l) {
          VERIFY_IS_EQUAL(chip5(i,j,k,l), tensor(i,j,k,l,7l));
        }
      }
    }
  }

  sycl_device.deallocate(gpu_data_tensor);
  sycl_device.deallocate(gpu_data_chip1);
  sycl_device.deallocate(gpu_data_chip2);
  sycl_device.deallocate(gpu_data_chip3);
  sycl_device.deallocate(gpu_data_chip4);
  sycl_device.deallocate(gpu_data_chip5);
}

template <typename DataType, int DataLayout, typename IndexType>
static void test_dynamic_chip_sycl(const Eigen::SyclDevice& sycl_device)
{
  IndexType sizeDim1 = 2;
  IndexType sizeDim2 = 3;
  IndexType sizeDim3 = 5;
  IndexType sizeDim4 = 7;
  IndexType sizeDim5 = 11;

  array<IndexType, 5> tensorRange = {{sizeDim1, sizeDim2, sizeDim3, sizeDim4, sizeDim5}};
  array<IndexType, 4> chip1TensorRange = {{sizeDim2, sizeDim3, sizeDim4, sizeDim5}};

  Tensor<DataType, 5, DataLayout,IndexType> tensor(tensorRange);
  Tensor<DataType, 4, DataLayout,IndexType> chip1(chip1TensorRange);

  tensor.setRandom();

  const size_t tensorBuffSize =tensor.size()*sizeof(DataType);
  const size_t chip1TensorBuffSize =chip1.size()*sizeof(DataType);
  DataType* gpu_data_tensor  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));
  DataType* gpu_data_chip1  = static_cast<DataType*>(sycl_device.allocate(chip1TensorBuffSize));

  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_tensor(gpu_data_tensor, tensorRange);
  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_chip1(gpu_data_chip1, chip1TensorRange);

  sycl_device.memcpyHostToDevice(gpu_data_tensor, tensor.data(), tensorBuffSize);
  gpu_chip1.device(sycl_device)=gpu_tensor.chip(1l,0l);
  sycl_device.memcpyDeviceToHost(chip1.data(), gpu_data_chip1, chip1TensorBuffSize);

  VERIFY_IS_EQUAL(chip1.dimension(0), sizeDim2);
  VERIFY_IS_EQUAL(chip1.dimension(1), sizeDim3);
  VERIFY_IS_EQUAL(chip1.dimension(2), sizeDim4);
  VERIFY_IS_EQUAL(chip1.dimension(3), sizeDim5);

  for (IndexType i = 0; i < sizeDim2; ++i) {
    for (IndexType j = 0; j < sizeDim3; ++j) {
      for (IndexType k = 0; k < sizeDim4; ++k) {
        for (IndexType l = 0; l < sizeDim5; ++l) {
          VERIFY_IS_EQUAL(chip1(i,j,k,l), tensor(1l,i,j,k,l));
        }
      }
    }
  }

  array<IndexType, 4> chip2TensorRange = {{sizeDim1, sizeDim3, sizeDim4, sizeDim5}};
  Tensor<DataType, 4, DataLayout,IndexType> chip2(chip2TensorRange);
  const size_t chip2TensorBuffSize =chip2.size()*sizeof(DataType);
  DataType* gpu_data_chip2  = static_cast<DataType*>(sycl_device.allocate(chip2TensorBuffSize));
  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_chip2(gpu_data_chip2, chip2TensorRange);

  gpu_chip2.device(sycl_device)=gpu_tensor.chip(1l,1l);
  sycl_device.memcpyDeviceToHost(chip2.data(), gpu_data_chip2, chip2TensorBuffSize);

  VERIFY_IS_EQUAL(chip2.dimension(0), sizeDim1);
  VERIFY_IS_EQUAL(chip2.dimension(1), sizeDim3);
  VERIFY_IS_EQUAL(chip2.dimension(2), sizeDim4);
  VERIFY_IS_EQUAL(chip2.dimension(3), sizeDim5);

  for (IndexType i = 0; i < sizeDim1; ++i) {
    for (IndexType j = 0; j < sizeDim3; ++j) {
      for (IndexType k = 0; k < sizeDim4; ++k) {
        for (IndexType l = 0; l < sizeDim5; ++l) {
          VERIFY_IS_EQUAL(chip2(i,j,k,l), tensor(i,1l,j,k,l));
        }
      }
    }
  }

  array<IndexType, 4> chip3TensorRange = {{sizeDim1, sizeDim2, sizeDim4, sizeDim5}};
  Tensor<DataType, 4, DataLayout,IndexType> chip3(chip3TensorRange);
  const size_t chip3TensorBuffSize =chip3.size()*sizeof(DataType);
  DataType* gpu_data_chip3  = static_cast<DataType*>(sycl_device.allocate(chip3TensorBuffSize));
  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_chip3(gpu_data_chip3, chip3TensorRange);

  gpu_chip3.device(sycl_device)=gpu_tensor.chip(2l,2l);
  sycl_device.memcpyDeviceToHost(chip3.data(), gpu_data_chip3, chip3TensorBuffSize);

  VERIFY_IS_EQUAL(chip3.dimension(0), sizeDim1);
  VERIFY_IS_EQUAL(chip3.dimension(1), sizeDim2);
  VERIFY_IS_EQUAL(chip3.dimension(2), sizeDim4);
  VERIFY_IS_EQUAL(chip3.dimension(3), sizeDim5);

  for (IndexType i = 0; i < sizeDim1; ++i) {
    for (IndexType j = 0; j < sizeDim2; ++j) {
      for (IndexType k = 0; k < sizeDim4; ++k) {
        for (IndexType l = 0; l < sizeDim5; ++l) {
          VERIFY_IS_EQUAL(chip3(i,j,k,l), tensor(i,j,2l,k,l));
        }
      }
    }
  }

  array<IndexType, 4> chip4TensorRange = {{sizeDim1, sizeDim2, sizeDim3, sizeDim5}};
  Tensor<DataType, 4, DataLayout,IndexType> chip4(chip4TensorRange);
  const size_t chip4TensorBuffSize =chip4.size()*sizeof(DataType);
  DataType* gpu_data_chip4  = static_cast<DataType*>(sycl_device.allocate(chip4TensorBuffSize));
  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_chip4(gpu_data_chip4, chip4TensorRange);

  gpu_chip4.device(sycl_device)=gpu_tensor.chip(5l,3l);
  sycl_device.memcpyDeviceToHost(chip4.data(), gpu_data_chip4, chip4TensorBuffSize);

  VERIFY_IS_EQUAL(chip4.dimension(0), sizeDim1);
  VERIFY_IS_EQUAL(chip4.dimension(1), sizeDim2);
  VERIFY_IS_EQUAL(chip4.dimension(2), sizeDim3);
  VERIFY_IS_EQUAL(chip4.dimension(3), sizeDim5);

  for (IndexType i = 0; i < sizeDim1; ++i) {
    for (IndexType j = 0; j < sizeDim2; ++j) {
      for (IndexType k = 0; k < sizeDim3; ++k) {
        for (IndexType l = 0; l < sizeDim5; ++l) {
          VERIFY_IS_EQUAL(chip4(i,j,k,l), tensor(i,j,k,5l,l));
        }
      }
    }
  }


  array<IndexType, 4> chip5TensorRange = {{sizeDim1, sizeDim2, sizeDim3, sizeDim4}};
  Tensor<DataType, 4, DataLayout,IndexType> chip5(chip5TensorRange);
  const size_t chip5TensorBuffSize =chip5.size()*sizeof(DataType);
  DataType* gpu_data_chip5  = static_cast<DataType*>(sycl_device.allocate(chip5TensorBuffSize));
  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_chip5(gpu_data_chip5, chip5TensorRange);

  gpu_chip5.device(sycl_device)=gpu_tensor.chip(7l,4l);
  sycl_device.memcpyDeviceToHost(chip5.data(), gpu_data_chip5, chip5TensorBuffSize);

  VERIFY_IS_EQUAL(chip5.dimension(0), sizeDim1);
  VERIFY_IS_EQUAL(chip5.dimension(1), sizeDim2);
  VERIFY_IS_EQUAL(chip5.dimension(2), sizeDim3);
  VERIFY_IS_EQUAL(chip5.dimension(3), sizeDim4);

  for (IndexType i = 0; i < sizeDim1; ++i) {
    for (IndexType j = 0; j < sizeDim2; ++j) {
      for (IndexType k = 0; k < sizeDim3; ++k) {
        for (IndexType l = 0; l < sizeDim4; ++l) {
          VERIFY_IS_EQUAL(chip5(i,j,k,l), tensor(i,j,k,l,7l));
        }
      }
    }
  }
  sycl_device.deallocate(gpu_data_tensor);
  sycl_device.deallocate(gpu_data_chip1);
  sycl_device.deallocate(gpu_data_chip2);
  sycl_device.deallocate(gpu_data_chip3);
  sycl_device.deallocate(gpu_data_chip4);
  sycl_device.deallocate(gpu_data_chip5);
}

template <typename DataType, int DataLayout, typename IndexType>
static void test_chip_in_expr(const Eigen::SyclDevice& sycl_device) {

  IndexType sizeDim1 = 2;
  IndexType sizeDim2 = 3;
  IndexType sizeDim3 = 5;
  IndexType sizeDim4 = 7;
  IndexType sizeDim5 = 11;

  array<IndexType, 5> tensorRange = {{sizeDim1, sizeDim2, sizeDim3, sizeDim4, sizeDim5}};
  array<IndexType, 4> chip1TensorRange = {{sizeDim2, sizeDim3, sizeDim4, sizeDim5}};

  Tensor<DataType, 5, DataLayout,IndexType> tensor(tensorRange);

  Tensor<DataType, 4, DataLayout,IndexType> chip1(chip1TensorRange);
  Tensor<DataType, 4, DataLayout,IndexType> tensor1(chip1TensorRange);
  tensor.setRandom();
  tensor1.setRandom();

  const size_t tensorBuffSize =tensor.size()*sizeof(DataType);
  const size_t chip1TensorBuffSize =chip1.size()*sizeof(DataType);
  DataType* gpu_data_tensor  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));
  DataType* gpu_data_chip1  = static_cast<DataType*>(sycl_device.allocate(chip1TensorBuffSize));
  DataType* gpu_data_tensor1  = static_cast<DataType*>(sycl_device.allocate(chip1TensorBuffSize));

  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_tensor(gpu_data_tensor, tensorRange);
  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_chip1(gpu_data_chip1, chip1TensorRange);
  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_tensor1(gpu_data_tensor1, chip1TensorRange);


  sycl_device.memcpyHostToDevice(gpu_data_tensor, tensor.data(), tensorBuffSize);
  sycl_device.memcpyHostToDevice(gpu_data_tensor1, tensor1.data(), chip1TensorBuffSize);
  gpu_chip1.device(sycl_device)=gpu_tensor.template chip<0l>(0l) + gpu_tensor1;
  sycl_device.memcpyDeviceToHost(chip1.data(), gpu_data_chip1, chip1TensorBuffSize);

  for (int i = 0; i < sizeDim2; ++i) {
    for (int j = 0; j < sizeDim3; ++j) {
      for (int k = 0; k < sizeDim4; ++k) {
        for (int l = 0; l < sizeDim5; ++l) {
          float expected = tensor(0l,i,j,k,l) + tensor1(i,j,k,l);
          VERIFY_IS_EQUAL(chip1(i,j,k,l), expected);
        }
      }
    }
  }

  array<IndexType, 3> chip2TensorRange = {{sizeDim2, sizeDim4, sizeDim5}};
  Tensor<DataType, 3, DataLayout,IndexType> tensor2(chip2TensorRange);
  Tensor<DataType, 3, DataLayout,IndexType> chip2(chip2TensorRange);
  tensor2.setRandom();
  const size_t chip2TensorBuffSize =tensor2.size()*sizeof(DataType);
  DataType* gpu_data_tensor2  = static_cast<DataType*>(sycl_device.allocate(chip2TensorBuffSize));
  DataType* gpu_data_chip2  = static_cast<DataType*>(sycl_device.allocate(chip2TensorBuffSize));
  TensorMap<Tensor<DataType, 3, DataLayout,IndexType>> gpu_tensor2(gpu_data_tensor2, chip2TensorRange);
  TensorMap<Tensor<DataType, 3, DataLayout,IndexType>> gpu_chip2(gpu_data_chip2, chip2TensorRange);

  sycl_device.memcpyHostToDevice(gpu_data_tensor2, tensor2.data(), chip2TensorBuffSize);
  gpu_chip2.device(sycl_device)=gpu_tensor.template chip<0l>(0l).template chip<1l>(2l) + gpu_tensor2;
  sycl_device.memcpyDeviceToHost(chip2.data(), gpu_data_chip2, chip2TensorBuffSize);

  for (int i = 0; i < sizeDim2; ++i) {
    for (int j = 0; j < sizeDim4; ++j) {
      for (int k = 0; k < sizeDim5; ++k) {
        float expected = tensor(0l,i,2l,j,k) + tensor2(i,j,k);
        VERIFY_IS_EQUAL(chip2(i,j,k), expected);
      }
    }
  }
  sycl_device.deallocate(gpu_data_tensor);
  sycl_device.deallocate(gpu_data_tensor1);
  sycl_device.deallocate(gpu_data_chip1);
  sycl_device.deallocate(gpu_data_tensor2);
  sycl_device.deallocate(gpu_data_chip2);
}

template <typename DataType, int DataLayout, typename IndexType>
static void test_chip_as_lvalue_sycl(const Eigen::SyclDevice& sycl_device)
{

  IndexType sizeDim1 = 2;
  IndexType sizeDim2 = 3;
  IndexType sizeDim3 = 5;
  IndexType sizeDim4 = 7;
  IndexType sizeDim5 = 11;

  array<IndexType, 5> tensorRange = {{sizeDim1, sizeDim2, sizeDim3, sizeDim4, sizeDim5}};
  array<IndexType, 4> input2TensorRange = {{sizeDim2, sizeDim3, sizeDim4, sizeDim5}};

  Tensor<DataType, 5, DataLayout,IndexType> tensor(tensorRange);
  Tensor<DataType, 5, DataLayout,IndexType> input1(tensorRange);
  Tensor<DataType, 4, DataLayout,IndexType> input2(input2TensorRange);
  input1.setRandom();
  input2.setRandom();


  const size_t tensorBuffSize =tensor.size()*sizeof(DataType);
  const size_t input2TensorBuffSize =input2.size()*sizeof(DataType);
  std::cout << tensorBuffSize << " , "<<  input2TensorBuffSize << std::endl;
  DataType* gpu_data_tensor  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));
  DataType* gpu_data_input1  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));
  DataType* gpu_data_input2  = static_cast<DataType*>(sycl_device.allocate(input2TensorBuffSize));

  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_tensor(gpu_data_tensor, tensorRange);
  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_input1(gpu_data_input1, tensorRange);
  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_input2(gpu_data_input2, input2TensorRange);

  sycl_device.memcpyHostToDevice(gpu_data_input1, input1.data(), tensorBuffSize);
  gpu_tensor.device(sycl_device)=gpu_input1;
  sycl_device.memcpyHostToDevice(gpu_data_input2, input2.data(), input2TensorBuffSize);
  gpu_tensor.template chip<0l>(1l).device(sycl_device)=gpu_input2;
  sycl_device.memcpyDeviceToHost(tensor.data(), gpu_data_tensor, tensorBuffSize);

  for (int i = 0; i < sizeDim1; ++i) {
    for (int j = 0; j < sizeDim2; ++j) {
      for (int k = 0; k < sizeDim3; ++k) {
        for (int l = 0; l < sizeDim4; ++l) {
          for (int m = 0; m < sizeDim5; ++m) {
            if (i != 1) {
              VERIFY_IS_EQUAL(tensor(i,j,k,l,m), input1(i,j,k,l,m));
            } else {
              VERIFY_IS_EQUAL(tensor(i,j,k,l,m), input2(j,k,l,m));
            }
          }
        }
      }
    }
  }

  gpu_tensor.device(sycl_device)=gpu_input1;
  array<IndexType, 4> input3TensorRange = {{sizeDim1, sizeDim3, sizeDim4, sizeDim5}};
  Tensor<DataType, 4, DataLayout,IndexType> input3(input3TensorRange);
  input3.setRandom();

  const size_t input3TensorBuffSize =input3.size()*sizeof(DataType);
  DataType* gpu_data_input3  = static_cast<DataType*>(sycl_device.allocate(input3TensorBuffSize));
  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_input3(gpu_data_input3, input3TensorRange);

  sycl_device.memcpyHostToDevice(gpu_data_input3, input3.data(), input3TensorBuffSize);
  gpu_tensor.template chip<1l>(1l).device(sycl_device)=gpu_input3;
  sycl_device.memcpyDeviceToHost(tensor.data(), gpu_data_tensor, tensorBuffSize);

  for (int i = 0; i < sizeDim1; ++i) {
    for (int j = 0; j < sizeDim2; ++j) {
      for (int k = 0; k <sizeDim3; ++k) {
        for (int l = 0; l < sizeDim4; ++l) {
          for (int m = 0; m < sizeDim5; ++m) {
            if (j != 1) {
              VERIFY_IS_EQUAL(tensor(i,j,k,l,m), input1(i,j,k,l,m));
            } else {
              VERIFY_IS_EQUAL(tensor(i,j,k,l,m), input3(i,k,l,m));
            }
          }
        }
      }
    }
  }

  gpu_tensor.device(sycl_device)=gpu_input1;
  array<IndexType, 4> input4TensorRange = {{sizeDim1, sizeDim2, sizeDim4, sizeDim5}};
  Tensor<DataType, 4, DataLayout,IndexType> input4(input4TensorRange);
  input4.setRandom();

  const size_t input4TensorBuffSize =input4.size()*sizeof(DataType);
  DataType* gpu_data_input4  = static_cast<DataType*>(sycl_device.allocate(input4TensorBuffSize));
  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_input4(gpu_data_input4, input4TensorRange);

  sycl_device.memcpyHostToDevice(gpu_data_input4, input4.data(), input4TensorBuffSize);
  gpu_tensor.template chip<2l>(3l).device(sycl_device)=gpu_input4;
  sycl_device.memcpyDeviceToHost(tensor.data(), gpu_data_tensor, tensorBuffSize);

  for (int i = 0; i < sizeDim1; ++i) {
    for (int j = 0; j < sizeDim2; ++j) {
      for (int k = 0; k <sizeDim3; ++k) {
        for (int l = 0; l < sizeDim4; ++l) {
          for (int m = 0; m < sizeDim5; ++m) {
            if (k != 3) {
              VERIFY_IS_EQUAL(tensor(i,j,k,l,m), input1(i,j,k,l,m));
            } else {
              VERIFY_IS_EQUAL(tensor(i,j,k,l,m), input4(i,j,l,m));
            }
          }
        }
      }
    }
  }

  gpu_tensor.device(sycl_device)=gpu_input1;
  array<IndexType, 4> input5TensorRange = {{sizeDim1, sizeDim2, sizeDim3, sizeDim5}};
  Tensor<DataType, 4, DataLayout,IndexType> input5(input5TensorRange);
  input5.setRandom();

  const size_t input5TensorBuffSize =input5.size()*sizeof(DataType);
  DataType* gpu_data_input5  = static_cast<DataType*>(sycl_device.allocate(input5TensorBuffSize));
  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_input5(gpu_data_input5, input5TensorRange);

  sycl_device.memcpyHostToDevice(gpu_data_input5, input5.data(), input5TensorBuffSize);
  gpu_tensor.template chip<3l>(4l).device(sycl_device)=gpu_input5;
  sycl_device.memcpyDeviceToHost(tensor.data(), gpu_data_tensor, tensorBuffSize);

  for (int i = 0; i < sizeDim1; ++i) {
    for (int j = 0; j < sizeDim2; ++j) {
      for (int k = 0; k <sizeDim3; ++k) {
        for (int l = 0; l < sizeDim4; ++l) {
          for (int m = 0; m < sizeDim5; ++m) {
            if (l != 4) {
              VERIFY_IS_EQUAL(tensor(i,j,k,l,m), input1(i,j,k,l,m));
            } else {
              VERIFY_IS_EQUAL(tensor(i,j,k,l,m), input5(i,j,k,m));
            }
          }
        }
      }
    }
  }
  gpu_tensor.device(sycl_device)=gpu_input1;
  array<IndexType, 4> input6TensorRange = {{sizeDim1, sizeDim2, sizeDim3, sizeDim4}};
  Tensor<DataType, 4, DataLayout,IndexType> input6(input6TensorRange);
  input6.setRandom();

  const size_t input6TensorBuffSize =input6.size()*sizeof(DataType);
  DataType* gpu_data_input6  = static_cast<DataType*>(sycl_device.allocate(input6TensorBuffSize));
  TensorMap<Tensor<DataType, 4, DataLayout,IndexType>> gpu_input6(gpu_data_input6, input6TensorRange);

  sycl_device.memcpyHostToDevice(gpu_data_input6, input6.data(), input6TensorBuffSize);
  gpu_tensor.template chip<4l>(5l).device(sycl_device)=gpu_input6;
  sycl_device.memcpyDeviceToHost(tensor.data(), gpu_data_tensor, tensorBuffSize);

  for (int i = 0; i < sizeDim1; ++i) {
    for (int j = 0; j < sizeDim2; ++j) {
      for (int k = 0; k <sizeDim3; ++k) {
        for (int l = 0; l < sizeDim4; ++l) {
          for (int m = 0; m < sizeDim5; ++m) {
            if (m != 5) {
              VERIFY_IS_EQUAL(tensor(i,j,k,l,m), input1(i,j,k,l,m));
            } else {
              VERIFY_IS_EQUAL(tensor(i,j,k,l,m), input6(i,j,k,l));
            }
          }
        }
      }
    }
  }


  gpu_tensor.device(sycl_device)=gpu_input1;
  Tensor<DataType, 5, DataLayout,IndexType> input7(tensorRange);
  input7.setRandom();

  DataType* gpu_data_input7  = static_cast<DataType*>(sycl_device.allocate(tensorBuffSize));
  TensorMap<Tensor<DataType, 5, DataLayout,IndexType>> gpu_input7(gpu_data_input7, tensorRange);

  sycl_device.memcpyHostToDevice(gpu_data_input7, input7.data(), tensorBuffSize);
  gpu_tensor.chip(0l,0l).device(sycl_device)=gpu_input7.chip(0l,0l);
  sycl_device.memcpyDeviceToHost(tensor.data(), gpu_data_tensor, tensorBuffSize);

  for (int i = 0; i < sizeDim1; ++i) {
    for (int j = 0; j < sizeDim2; ++j) {
      for (int k = 0; k <sizeDim3; ++k) {
        for (int l = 0; l < sizeDim4; ++l) {
          for (int m = 0; m < sizeDim5; ++m) {
            if (i != 0) {
              VERIFY_IS_EQUAL(tensor(i,j,k,l,m), input1(i,j,k,l,m));
            } else {
              VERIFY_IS_EQUAL(tensor(i,j,k,l,m), input7(i,j,k,l,m));
            }
          }
        }
      }
    }
  }
  sycl_device.deallocate(gpu_data_tensor);
  sycl_device.deallocate(gpu_data_input1);
  sycl_device.deallocate(gpu_data_input2);
  sycl_device.deallocate(gpu_data_input3);
  sycl_device.deallocate(gpu_data_input4);
  sycl_device.deallocate(gpu_data_input5);
  sycl_device.deallocate(gpu_data_input6);
  sycl_device.deallocate(gpu_data_input7);

}

template<typename DataType, typename dev_Selector> void sycl_chipping_test_per_device(dev_Selector s){
  QueueInterface queueInterface(s);
  auto sycl_device = Eigen::SyclDevice(&queueInterface);
 /* test_static_chip_sycl<DataType, RowMajor, int64_t>(sycl_device);
  test_static_chip_sycl<DataType, ColMajor, int64_t>(sycl_device);
  test_dynamic_chip_sycl<DataType, RowMajor, int64_t>(sycl_device);
  test_dynamic_chip_sycl<DataType, ColMajor, int64_t>(sycl_device);
  test_chip_in_expr<DataType, RowMajor, int64_t>(sycl_device);
  test_chip_in_expr<DataType, ColMajor, int64_t>(sycl_device);*/
  test_chip_as_lvalue_sycl<DataType, RowMajor, int64_t>(sycl_device);
 // test_chip_as_lvalue_sycl<DataType, ColMajor, int64_t>(sycl_device);
}
EIGEN_DECLARE_TEST(cxx11_tensor_chipping_sycl)
{
  for (const auto& device :Eigen::get_sycl_supported_devices()) {
    CALL_SUBTEST(sycl_chipping_test_per_device<float>(device));
  }
}
