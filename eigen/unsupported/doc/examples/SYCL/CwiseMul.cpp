#include <iostream>
#define EIGEN_USE_SYCL
#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::array;
using Eigen::SyclDevice;
using Eigen::Tensor;
using Eigen::TensorMap;

int main()
{
  using DataType = float;
  using IndexType = int64_t;
  constexpr auto DataLayout = Eigen::RowMajor;

  auto devices = Eigen::get_sycl_supported_devices();
  const auto device_selector = *devices.begin();
  Eigen::QueueInterface queueInterface(device_selector);
  auto sycl_device = Eigen::SyclDevice(&queueInterface);
  
  // create the tensors to be used in the operation
  IndexType sizeDim1 = 3;
  IndexType sizeDim2 = 3;
  IndexType sizeDim3 = 3;
  array<IndexType, 3> tensorRange = {{sizeDim1, sizeDim2, sizeDim3}};

  // initialize the tensors with the data we want manipulate to
  Tensor<DataType, 3,DataLayout, IndexType> in1(tensorRange);
  Tensor<DataType, 3,DataLayout, IndexType> in2(tensorRange);
  Tensor<DataType, 3,DataLayout, IndexType> out(tensorRange);

  // set up some random data in the tensors to be multiplied
  in1 = in1.random();
  in2 = in2.random();

  // allocate memory for the tensors
  DataType * gpu_in1_data  = static_cast<DataType*>(sycl_device.allocate(in1.size()*sizeof(DataType)));
  DataType * gpu_in2_data  = static_cast<DataType*>(sycl_device.allocate(in2.size()*sizeof(DataType)));
  DataType * gpu_out_data =  static_cast<DataType*>(sycl_device.allocate(out.size()*sizeof(DataType)));

  // 
  TensorMap<Tensor<DataType, 3, DataLayout, IndexType>> gpu_in1(gpu_in1_data, tensorRange);
  TensorMap<Tensor<DataType, 3, DataLayout, IndexType>> gpu_in2(gpu_in2_data, tensorRange);
  TensorMap<Tensor<DataType, 3, DataLayout, IndexType>> gpu_out(gpu_out_data, tensorRange);

  // copy the memory to the device and do the c=a*b calculation
  sycl_device.memcpyHostToDevice(gpu_in1_data, in1.data(),(in1.size())*sizeof(DataType));
  sycl_device.memcpyHostToDevice(gpu_in2_data, in2.data(),(in2.size())*sizeof(DataType));
  gpu_out.device(sycl_device) = gpu_in1 * gpu_in2;
  sycl_device.memcpyDeviceToHost(out.data(), gpu_out_data,(out.size())*sizeof(DataType));
  sycl_device.synchronize();

  // print out the results
   for (IndexType i = 0; i < sizeDim1; ++i) {
    for (IndexType j = 0; j < sizeDim2; ++j) {
      for (IndexType k = 0; k < sizeDim3; ++k) {
        std::cout << "device_out" << "(" << i << ", " << j << ", " << k << ") : " << out(i,j,k) 
                  << " vs host_out" << "(" << i << ", " << j << ", " << k << ") : " << in1(i,j,k) * in2(i,j,k) << "\n";
      }
    }
  }
  printf("c=a*b Done\n");
}
