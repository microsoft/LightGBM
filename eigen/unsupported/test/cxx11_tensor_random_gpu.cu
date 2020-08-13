// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_NO_COMPLEX

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU

#include "main.h"
#include <Eigen/CXX11/Tensor>

#include <Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h>

void test_gpu_random_uniform()
{
  Tensor<float, 2> out(72,97);
  out.setZero();

  std::size_t out_bytes = out.size() * sizeof(float);

  float* d_out;
  gpuMalloc((void**)(&d_out), out_bytes);

  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<float, 2> > gpu_out(d_out, 72,97);

  gpu_out.device(gpu_device) = gpu_out.random();

  assert(gpuMemcpyAsync(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);

  // For now we just check this code doesn't crash.
  // TODO: come up with a valid test of randomness
}


void test_gpu_random_normal()
{
  Tensor<float, 2> out(72,97);
  out.setZero();

  std::size_t out_bytes = out.size() * sizeof(float);

  float* d_out;
  gpuMalloc((void**)(&d_out), out_bytes);

  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<float, 2> > gpu_out(d_out, 72,97);

  Eigen::internal::NormalRandomGenerator<float> gen(true);
  gpu_out.device(gpu_device) = gpu_out.random(gen);

  assert(gpuMemcpyAsync(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);
}

static void test_complex()
{
  Tensor<std::complex<float>, 1> vec(6);
  vec.setRandom();

  // Fixme: we should check that the generated numbers follow a uniform
  // distribution instead.
  for (int i = 1; i < 6; ++i) {
    VERIFY_IS_NOT_EQUAL(vec(i), vec(i-1));
  }
}


EIGEN_DECLARE_TEST(cxx11_tensor_random_gpu)
{
  CALL_SUBTEST(test_gpu_random_uniform());
  CALL_SUBTEST(test_gpu_random_normal());
  CALL_SUBTEST(test_complex());
}
