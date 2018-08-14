# Summary

This repository is an exploration of additional executor properties which could enable task dependencies without requiring the existence of future types in P0443.

If successful, these properties would eliminate the need for these four execution functions from P0443's executor API:

  1. `.twoway_execute()`
  2. `.then_execute()`
  3. `.bulk_twoway_execute()`
  4. `.bulk_then_execute()`

It would also remove P0443's dependency on a `Future` concept and defer related discussion to future work.

# Basic Idea

The basic idea is to allow executor types to optionally expose properties which describe an incoming and outgoing dependency of execution agents they create.

For example, a client may require that an agent's execution depends on some incoming dependency ID:

    execution::require(ex, execution::depends_on(id)).execute(func);

In this example, `ex` guarantees that `func` will not be invoked until `id` is complete. If `ex` cannot support dependency IDs, then the expression is ill-formed.

Symmetrically, a client may query for the dependency ID corresponding to the completion of the last task created by an executor:

    auto id = execution::query(ex, execution::dependency_id);

This ID may be used as the incoming dependency of a subsequent task.

# Heterogeneous Task Chaining

The type of an executor's dependency ID is executor-defined, and in general
executor types will not recognize the dependency IDs of other executor types.
In order to enable the chaining of tasks created by heterogeneous types of
executors, we require the ability to construct an incomplete dependency ID and
signal its completion by a client of that executor. Such a signal would
complete the dependency and release an already submitted task for active
execution.

    executor_a ex_a;
    executor_b ex_b;

    // create a signaller for ex_b
    auto signaller = execution::query(ex_b, execution::signaller_factory)();

    // get the signaller's dependency ID
    auto task_a = signaller.dependency_id();

    // launch task a on ex_a and signal when complete
    ex_a.execute([signaller = std::move(signaller)]() mutable
    {
      std::cout << "Hello, world from task a on ex_a!" << std::endl;

      // signal that task a is complete
      signaller.signal();
    });

    // launch task b on ex_b dependent on task a
    execution::require(ex_b, execution::depends_on(task_a)).execute([]
    {
      std::cout << "Hello, world from task b on ex_b!" << std::endl;
    });

# Impact on P0443

1. Introduce new material
   1. `DependencyId` type requirements
   2. Requirable & preferable `execution::depends_on` property
   3. Queryable `execution::dependency_id` property
   4. Queryable `execution::signaller_factory` property
4. Eliminate old material
   1. `Future`
   2. `executor_future_t`
   3. `then_execute`
   4. `twoway_execute`
   5. `bulk_then_execute`
   6. `bulk_twoway_execute`
   7. Others?

# Known Problems

See [demo program](demo_gpu_then_host.cu).

