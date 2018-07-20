#include "oneway_cuda_executor.hpp"
#include "new_thread_executor.hpp"
#include <iostream>

int main()
{
  // demonstrate a CUDA task dependent on a task running in a new thread
  new_thread_executor host;
  oneway_cuda_executor gpu;

  // create a signaller
  auto signaller = gpu.query(signaller_factory)();

  // get the signaller's dependency
  auto task_a = signaller.dependency();

  // launch task A on the host and signal the gpu when complete
  host.execute([signaller = std::move(signaller)]() mutable
  {
    std::cout << "Hello, world from task A running on the host!" << std::endl;

    // signal that task A is complete
    signaller.signal();
  });

  // launch task B on the GPU and depend on task A
  gpu.require(depend_on(task_a)).execute([] __host__ __device__
  {
    printf("Hello, world from task B running on the gpu!\n");
  });

  // wait on gpu's most recent task
  // require blocking and use a no-op task
  gpu.require(blocking_always).execute([] __host__ __device__ {});

  std::cout << "OK" << std::endl;

  return 0;
}

