#include <iostream>
#include <utility>
#include <type_traits>


class _cuda_stream
{
  public:
    _cuda_stream()
      : stream_(make_cudaStream_t())
    {}

    _cuda_stream(const _cuda_stream&) = delete;

    _cuda_stream(_cuda_stream&& other)
      : stream_{}
    {
      std::swap(stream_, other.stream_);
    }

    ~_cuda_stream()
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


class _cuda_event
{
  public:
    _cuda_event()
      : event_(make_ready_cudaEvent_t())
    {}

    _cuda_event(const _cuda_event& other)
      : event_{}
    {
      // create a new stream so we can create a new event
      _cuda_stream stream;

      // make stream wait on other
      stream.wait_on(other.native_handle());

      // record our cudaEvent_t on stream
      stream.record(event_);
    }

    _cuda_event(_cuda_event&& other)
      : event_{}
    {
      std::swap(event_, other.event_);
    }

    ~_cuda_event()
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
__global__ void _bulk_kernel(Function f)
{
  f(static_cast<size_t>(blockDim.x * blockIdx.x + threadIdx.x));
}


template<class Function>
__global__ void _singular_kernel(Function f)
{
  f();
}


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


constexpr depend_on_t depend_on {};

class dependency_id_t {};

constexpr dependency_id_t dependency_id {};


class oneway_cuda_executor
{
  public:
    oneway_cuda_executor()
      : oneway_cuda_executor(cudaEvent_t{})
    {}

    oneway_cuda_executor(const oneway_cuda_executor&) = default;

    oneway_cuda_executor require(depend_on_t d) const
    {
      return oneway_cuda_executor(d.value());
    }

    cudaEvent_t query(dependency_id_t) const
    {
      return dependency_id_.native_handle();
    }

    template<class Function>
    void execute(Function f) const
    {
      // create a new stream
      _cuda_stream stream;

      // make stream wait on our external dependency, if it exists
      if(external_dependency_) stream.wait_on(external_dependency_);

      // launch the singular kernel
      _singular_kernel<<<1,1,0,stream.native_handle()>>>(f);

      // record an event corresponding to this launch
      stream.record(dependency_id_.native_handle());
    }

  private:
    oneway_cuda_executor(cudaEvent_t external_dependency)
      : external_dependency_(external_dependency)
    {}

    cudaEvent_t external_dependency_;

    // this is an RAII object because it is "owned" by *this
    _cuda_event dependency_id_;
};

static_assert(std::is_copy_constructible<oneway_cuda_executor>::value, "oneway_cuda_executor is not copy constructible.");

