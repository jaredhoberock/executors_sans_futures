#include <iostream>
#include <utility>
#include <type_traits>
#include <cassert>


namespace detail
{


class cuda_stream
{
  public:
    cuda_stream()
      : stream_(make_cudaStream_t())
    {}

    cuda_stream(const cuda_stream&) = delete;

    cuda_stream(cuda_stream&& other)
      : stream_{}
    {
      std::swap(stream_, other.stream_);
    }

    ~cuda_stream()
    {
      if(stream_)
      {
        if(auto error = cudaStreamDestroy(stream_))
        {
          std::cerr << "CUDA error after cudaStreamDestroy(): " + std::string(cudaGetErrorString(error)) << std::endl;
          std::terminate();
        }
      }
    }

    cudaStream_t native_handle() const
    {
      return stream_;
    }

    void wait_on(cudaEvent_t event) const
    {
      assert(event);
      if(auto error = cudaStreamWaitEvent(stream_, event, 0))
      {
        throw std::runtime_error("CUDA error after cudaStreamWaitEvent(): " + std::string(cudaGetErrorString(error)));
      }
    }

    void record(cudaEvent_t event) const
    {
      if(auto error = cudaEventRecord(event, stream_))
      {
        throw std::runtime_error("CUDA error after cudaEventRecord(): " + std::string(cudaGetErrorString(error)));
      }
    }

  private:
    static cudaStream_t make_cudaStream_t()
    {
      cudaStream_t result{};

      if(auto error = cudaStreamCreate(&result))
      {
        throw std::runtime_error("CUDA error after cudaStreamCreate(): " + std::string(cudaGetErrorString(error)));
      }

      return result;
    }

    cudaStream_t stream_;
};


class cuda_event
{
  public:
    cuda_event()
      : event_(make_ready_cudaEvent_t())
    {}

    cuda_event(const cuda_event& other)
      : cuda_event()
    {
      // create a new stream so we can create a new event
      cuda_stream stream;

      // make stream wait on other
      assert(other.native_handle());
      stream.wait_on(other.native_handle());

      // record our cudaEvent_t on stream
      stream.record(event_);
    }

    cuda_event(cuda_event&& other)
      : event_{}
    {
      std::swap(event_, other.event_);
    }

    ~cuda_event()
    {
      if(event_)
      {
        if(auto error = cudaEventDestroy(event_))
        {
          std::cerr << "CUDA error after cudaEventDestroy(): " + std::string(cudaGetErrorString(error)) << std::endl;
          std::terminate();
        }
      }
    }

    cudaEvent_t native_handle() const
    {
      return event_;
    }

  private:
    static cudaEvent_t make_ready_cudaEvent_t()
    {
      cudaEvent_t result{};

      if(auto error = cudaEventCreateWithFlags(&result, cudaEventDisableTiming))
      {
        throw std::runtime_error("CUDA error after cudaEventCreateWithFlags(): " + std::string(cudaGetErrorString(error)));
      }

      return result;
    }

    cudaEvent_t event_;
};


template<class Function>
__global__ void bulk_kernel(Function f)
{
  f(static_cast<size_t>(blockDim.x * blockIdx.x + threadIdx.x));
}


template<class Function>
__global__ void singular_kernel(Function f)
{
  f();
}


} // end detail


class depend_on_t
{
  public:
    constexpr depend_on_t()
      : depend_on_t(cudaEvent_t{})
    {}

    constexpr depend_on_t(cudaEvent_t e)
      : event_(e)
    {}

    constexpr depend_on_t operator()(cudaEvent_t e) const
    {
      return depend_on_t{e};
    }

    constexpr cudaEvent_t value() const
    {
      return event_;
    }

  private:
    cudaEvent_t event_;
};


constexpr depend_on_t depend_on{};

class dependency_id_t {};

constexpr dependency_id_t dependency_id{};

class blocking_never_t {};
constexpr blocking_never_t blocking_never{};

class blocking_always_t {};
constexpr blocking_always_t blocking_always{};

class oneway_cuda_executor
{
  public:
    oneway_cuda_executor()
      : oneway_cuda_executor(false, cudaEvent_t{})
    {}

    oneway_cuda_executor(const oneway_cuda_executor&) = default;

    cudaEvent_t query(dependency_id_t) const
    {
      return dependency_id_.native_handle();
    }

    oneway_cuda_executor require(depend_on_t d) const
    {
      // XXX shouldn't the result executor depend on d AND dependency_id_?
      return oneway_cuda_executor(blocking_, d.value());
    }

    oneway_cuda_executor require(blocking_never_t) const
    {
      return oneway_cuda_executor(false, dependency_id_.native_handle());
    }

    oneway_cuda_executor require(blocking_always_t) const
    {
      return oneway_cuda_executor(true, dependency_id_.native_handle());
    }

    template<class Function>
    void execute(Function f) const
    {
      // create a new stream
      detail::cuda_stream stream;

      // make stream wait on our external dependency, if it exists
      if(external_dependency_) stream.wait_on(external_dependency_);

      // launch the singular kernel
      detail::singular_kernel<<<1,1,0,stream.native_handle()>>>(f);

      // record an event corresponding to this launch
      stream.record(dependency_id_.native_handle());

      // block if we are required to
      if(blocking_)
      {
        wait();
      }
    }

  private:
    oneway_cuda_executor(bool blocking, cudaEvent_t external_dependency)
      : blocking_(blocking), external_dependency_(external_dependency)
    {}

    void wait() const
    {
      if(auto error = cudaEventSynchronize(dependency_id_.native_handle()))
      {
        throw std::runtime_error("CUDA error after cudaEventSynchronize(): " + std::string(cudaGetErrorString(error)));
      }
    }

    bool blocking_;
    cudaEvent_t external_dependency_;
    detail::cuda_event dependency_id_; // this is an RAII object because it is owned by *this
};

static_assert(std::is_copy_constructible<oneway_cuda_executor>::value, "oneway_cuda_executor is not copy constructible.");

