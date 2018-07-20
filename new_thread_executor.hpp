#include <thread>
#include <utility>
#include "execution.hpp"

class new_thread_executor
{
  public:
    template<class Function>
    void execute(Function&& f) const
    {
      std::thread(std::forward<Function>(f)).detach();
    }

    bool operator==(const new_thread_executor&) const
    {
      return true;
    }

    bool operator!=(const new_thread_executor&) const
    {
      return false;
    }
};

