// nvcc -std=c++14 --expt-extended-lambda demo_host_then_gpu.cu -lcuda
#include <iostream>
#include "execution.hpp"
#include "oneway_cuda_executor.hpp"
#include "new_thread_executor.hpp"

int main()
{
  // demonstrate a GPU task dependent on a host task
  new_thread_executor host;
  oneway_cuda_executor gpu;

  // create a signaller
  auto signaller = gpu.query(execution::signaller_factory)();

  // get the signaller's dependency
  auto task_a = signaller.dependency();

  // launch task A on the host and signal the gpu when complete
  host.execute([signaller = std::move(signaller)]() mutable
  {
    std::cout << "Hello, world from task A running on the host!" << std::endl;

    // signal that task A is complete
    signaller.signal();
  });

  // launch task B on the GPU which depends on task A
  gpu.require(execution::depend_on(task_a)).execute([] __host__ __device__
  {
    printf("Hello, world from task B running on the gpu!\n");
  });

  // wait on gpu's most recent task
  // require blocking and use a no-op task
  gpu.require(execution::blocking_always).execute([] __host__ __device__ {});

  std::cout << "OK" << std::endl;

  return 0;
}

