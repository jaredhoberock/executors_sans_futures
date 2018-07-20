// nvcc -std=c++14 --expt-extended-lambda demo_gpu_then_host.cu -lcuda
#include <iostream>
#include "execution.hpp"
#include "oneway_cuda_executor.hpp"
#include "new_thread_executor.hpp"

int main()
{
  // demonstrate a host task dependent on a task running on the GPU
  oneway_cuda_executor gpu;
  new_thread_executor host;

  // create a signaller
  auto signaller = host.query(signaller_factory)();

  // get the signaller's dependency
  auto task_a = signaller.dependency();

  // launch task A on the gpu and signal the host when complete
  // XXX attempting to pass the signaler to a GPU task is illegal
  // XXX even if it were legal, it would be inefficient to do so because
  // XXX the most efficient place to signal is at a join point after the body of the task completes.
  // XXX what if we had an optional property to call a function when the most recent task completes?
  // XXX what agent would that function be invoked on, and what would its execution guarantees be?
  gpu.execute([signaller = std::move(signaller)]() __host__ __device__ mutable
  {
    printf("Hello, world from task A running on the gpu!\n");

    // signal that task A is complete
    signaller.signal();
  });

  // launch task B on the host which depends on task A
  host.require(depend_on(task_a)).execute([]
  {
  std::cout << "Hello, world from task B running on the host!" << std::endl;
  });

  // wait on host's most recent task
  // require blocking and use a no-op task
  // XXX burning an agent to block is no good
  // XXX do we need some sort of wait on last task property?
  host.require(blocking_always).execute([]{});

  std::cout << "OK" << std::endl;

  return 0;
}

