#include <cuda.h>
#include <iostream>
#include <stdexcept>

__global__ void kernel()
{
  printf("Hello, world!\n");
}

int main()
{
  if(cuInit(0) != CUDA_SUCCESS)
  {
    std::cerr << "CUDA error after cuDeviceGetAttribute()" << std::endl;
    std::terminate();
  }

  int pi = 0;
  if(cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS, 0) != CUDA_SUCCESS)
  {
    throw std::runtime_error("CUDA error after cuDeviceGetAttribute().");
  }

  if(!pi)
  {
    throw std::runtime_error("CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS is unsupported.");
  }

  std::int32_t flag = 0;
  if(auto error = cudaHostRegister(&flag, sizeof(std::int32_t), cudaHostRegisterDefault))
  {
    throw std::runtime_error("CUDA error after cudaHostRegister(): " + std::string(cudaGetErrorString(error)));
  }

  void* d_flag = nullptr;
  if(auto error = cudaHostGetDevicePointer(&d_flag, &flag, 0))
  {
    throw std::runtime_error("CUDA error after cudaHostGetDevicePointer(): " + std::string(cudaGetErrorString(error)));
  }

  cudaStream_t stream{};
  if(auto error = cudaStreamCreate(&stream))
  {
    throw std::runtime_error("CUDA error after cudaStreamCreate(): " + std::string(cudaGetErrorString(error)));
  }

  if(cuStreamWaitValue32(stream, reinterpret_cast<CUdeviceptr>(d_flag), 1, CU_STREAM_WAIT_VALUE_EQ) != CUDA_SUCCESS)
  {
    throw std::runtime_error("CUDA error after cuStreamWaitValue32().");
  }

  // launch the kernel
  kernel<<<1,1,0,stream>>>();

  // release the kernel
  flag = true;

  if(auto error = cudaStreamSynchronize(stream))
  {
    throw std::runtime_error("CUDA error after cudaStreamSynchronize(): " + std::string(cudaGetErrorString(error)));
  }

  if(auto error = cudaStreamDestroy(stream))
  {
    throw std::runtime_error("CUDA error after cudaStreamDestroy(): " + std::string(cudaGetErrorString(error)));
  }

  std::cout << "OK" << std::endl;

  return 0;
}

