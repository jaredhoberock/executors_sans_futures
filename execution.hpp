#pragma once

namespace execution
{


template<class DependencyId>
class depend_on_t
{
  public:
    constexpr depend_on_t(DependencyId id)
      : id_(id)
    {}

    constexpr DependencyId value() const
    {
      return id_;
    }

  private:
    DependencyId id_;
};


template<>
class depend_on_t<void>
{
  public:
    template<class DependencyId>
    constexpr depend_on_t<DependencyId> operator()(DependencyId id) const
    {
      return depend_on_t<DependencyId>{id};
    }
};


constexpr depend_on_t<void> depend_on{};

class dependency_id_t {};

constexpr dependency_id_t dependency_id{};

class blocking_never_t {};
constexpr blocking_never_t blocking_never{};

class blocking_always_t {};
constexpr blocking_always_t blocking_always{};


class signaller_factory_t {};
constexpr signaller_factory_t signaller_factory{};


} // end execution

