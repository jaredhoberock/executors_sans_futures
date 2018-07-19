#include <iostream>
#include "oneway_cuda_executor.hpp"

int main()
{
  // start with an executor dependent on nothing
  oneway_cuda_executor ex_a;

  // launch task a
  ex_a.execute([] __host__ __device__ ()
  {
    printf("Hello, world from task a!\n");
  });

  // create a new executor dependent on task a
  auto ex_b = ex_a.require(depend_on(ex_a.query_last_event())); 

  // launch task b dependent on task a
  ex_b.execute([] __host__ __device__ ()
  {
    printf("Hello, world from task b!\n");
  });

  // wait on ex_b's last event
  // XXX having to go out of band to synchronize is undesirable
  if(auto error = cudaEventSynchronize(ex_b.query_last_event()))
  {
    throw std::runtime_error("CUDA error after cudaEventSynchronize(): " + std::string(cudaGetErrorString(error)));
  }

  std::cout << "OK" << std::endl;

  return 0;
}

