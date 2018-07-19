#include <cuda.h>
#include <iostream>
#include <cstdint>
#include <memory>
#include <thread>
#include <chrono>

__global__ void dummy_kernel() {}

class signaller
{
  public:
    signaller()
      : flag_(new std::int32_t(0)),
        d_flag_(get_device_pointer(flag_.get())),
        event_(make_event())
    {}

    ~signaller()
    {
      if(auto error = cudaEventDestroy(event_))
      {
        std::cerr << "CUDA error after cudaEventDestroy(): " + std::string(cudaGetErrorString(error));
        std::terminate();
      }

      if(auto error = cudaHostUnregister(flag_.get()))
      {
        std::cerr << "CUDA error after cudaHostUnregister(): " + std::string(cudaGetErrorString(error));
        std::terminate();
      }
    }

    void signal()
    {
      *flag_ = true;
    }

    cudaEvent_t dependency() const
    {
      return event_;
    }

  private:
    static std::int32_t* get_device_pointer(std::int32_t* ptr)
    {
      if(auto error = cudaHostRegister(ptr, sizeof(std::int32_t), cudaHostRegisterDefault))
      {
        throw std::runtime_error("CUDA error after cudaHostRegister(): " + std::string(cudaGetErrorString(error)));
      }

      std::int32_t* d_ptr = nullptr;
      if(auto error = cudaHostGetDevicePointer(&d_ptr, ptr, 0))
      {
        throw std::runtime_error("CUDA error after cudaHostGetDevicePointer(): " + std::string(cudaGetErrorString(error)));
      }

      return d_ptr;
    }

    cudaEvent_t make_event()
    {
      // create a new stream
      cudaStream_t stream{};
      if(auto error = cudaStreamCreate(&stream))
      {
        throw std::runtime_error("CUDA error after cudaStreamCreate(): " + std::string(cudaGetErrorString(error)));
      }

      // make the stream wait on the flag
      if(cuStreamWaitValue32(stream, reinterpret_cast<CUdeviceptr>(d_flag_), 1, CU_STREAM_WAIT_VALUE_EQ) != CUDA_SUCCESS)
      {
        throw std::runtime_error("CUDA error after cuStreamWaitValue32().");
      }

      // launch a dummy kernel
      dummy_kernel<<<1,1,0,stream>>>();

      // create an event
      cudaEvent_t event{};
      if(auto error = cudaEventCreateWithFlags(&event, cudaEventDisableTiming))
      {
        throw std::runtime_error("CUDA error after cudaEventCreateWithFlags(): " + std::string(cudaGetErrorString(error)));
      }

      // record it
      if(auto error = cudaEventRecord(event, stream))
      {
        throw std::runtime_error("CUDA error after cudaEventRecord(): " + std::string(cudaGetErrorString(error)));
      }

      // destroy the stream
      if(auto error = cudaStreamDestroy(stream))
      {
        throw std::runtime_error("CUDA error after cudaStreamDestroy(): " + std::string(cudaGetErrorString(error)));
      }

      return event;
    }

    std::unique_ptr<int32_t> flag_;
    std::int32_t* d_flag_;
    cudaEvent_t event_;
};


__global__ void hello_world()
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

  ::signaller signaller;

  cudaStream_t stream{};
  if(auto error = cudaStreamCreate(&stream))
  {
    throw std::runtime_error("CUDA error after cudaStreamCreate(): " + std::string(cudaGetErrorString(error)));
  }

  // make our stream wait on the signal's dependency
  if(auto error = cudaStreamWaitEvent(stream, signaller.dependency(), 0))
  {
    throw std::runtime_error("CUDA error after cudaStreamWaitEvent(): " + std::string(cudaGetErrorString(error)));
  }

  // launch a kernel on our stream dependent on the gate being released 
  hello_world<<<1,1,0,stream>>>();

  // wait for a couple seconds before signaling
  std::cout << "Sleeping before signaling kernel..." << std::endl;
  std::this_thread::sleep_for(std::chrono::seconds(2));
  signaller.signal();

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

