// nvcc -std=c++14 --expt-extended-lambda demo_gpu_then_gpu.cu -lcuda
#include <iostream>
#include "execution.hpp"
#include "oneway_cuda_executor.hpp"

int main()
{
  // start with an executor dependent on nothing
  oneway_cuda_executor ex_a;

  // execute task a
  ex_a.execute([] __host__ __device__ ()
  {
    printf("Hello, world from task a!\n");
  });

  // create a new executor dependent on task a
  auto task_a = ex_a.query(dependency_id);
  auto ex_b = ex_a.require(depend_on(task_a));

  // execute task b dependent on task a
  ex_b.execute([] __host__ __device__ ()
  {
    printf("Hello, world from task b!\n");
  });

  // wait on ex_b's most recent task
  // require blocking and use a no-op task
  ex_b.require(blocking_always).execute([] __host__ __device__ {});

  std::cout << "OK" << std::endl;

  return 0;
}

