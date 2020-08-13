// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_USE_THREADS

#include <iostream>
#include <unordered_set>

#include "main.h"
#include <Eigen/CXX11/ThreadPool>

struct Counter {
  Counter() = default;

  void inc() {
    // Check that mutation happens only in a thread that created this counter.
    VERIFY_IS_EQUAL(std::this_thread::get_id(), created_by);
    counter_value++;
  }
  int value() { return counter_value; }

  std::thread::id created_by;
  int counter_value = 0;
};

struct InitCounter {
  void operator()(Counter& counter) {
    counter.created_by = std::this_thread::get_id();
  }
};

void test_simple_thread_local() {
  int num_threads = internal::random<int>(4, 32);
  Eigen::ThreadPool thread_pool(num_threads);
  Eigen::ThreadLocal<Counter, InitCounter> counter(num_threads, InitCounter());

  int num_tasks = 3 * num_threads;
  Eigen::Barrier barrier(num_tasks);

  for (int i = 0; i < num_tasks; ++i) {
    thread_pool.Schedule([&counter, &barrier]() {
      Counter& local = counter.local();
      local.inc();

      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      barrier.Notify();
    });
  }

  barrier.Wait();

  counter.ForEach(
      [](std::thread::id, Counter& cnt) { VERIFY_IS_EQUAL(cnt.value(), 3); });
}

void test_zero_sized_thread_local() {
  Eigen::ThreadLocal<Counter, InitCounter> counter(0, InitCounter());

  Counter& local = counter.local();
  local.inc();

  int total = 0;
  counter.ForEach([&total](std::thread::id, Counter& cnt) {
    total += cnt.value();
    VERIFY_IS_EQUAL(cnt.value(), 1);
  });

  VERIFY_IS_EQUAL(total, 1);
}

// All thread local values fits into the lock-free storage.
void test_large_number_of_tasks_no_spill() {
  int num_threads = internal::random<int>(4, 32);
  Eigen::ThreadPool thread_pool(num_threads);
  Eigen::ThreadLocal<Counter, InitCounter> counter(num_threads, InitCounter());

  int num_tasks = 10000;
  Eigen::Barrier barrier(num_tasks);

  for (int i = 0; i < num_tasks; ++i) {
    thread_pool.Schedule([&counter, &barrier]() {
      Counter& local = counter.local();
      local.inc();
      barrier.Notify();
    });
  }

  barrier.Wait();

  int total = 0;
  std::unordered_set<std::thread::id> unique_threads;

  counter.ForEach([&](std::thread::id id, Counter& cnt) {
    total += cnt.value();
    unique_threads.insert(id);
  });

  VERIFY_IS_EQUAL(total, num_tasks);
  // Not all threads in a pool might be woken up to execute submitted tasks.
  // Also thread_pool.Schedule() might use current thread if queue is full.
  VERIFY_IS_EQUAL(
      unique_threads.size() <= (static_cast<size_t>(num_threads + 1)), true);
}

// Lock free thread local storage is too small to fit all the unique threads,
// and it spills to a map guarded by a mutex.
void test_large_number_of_tasks_with_spill() {
  int num_threads = internal::random<int>(4, 32);
  Eigen::ThreadPool thread_pool(num_threads);
  Eigen::ThreadLocal<Counter, InitCounter> counter(1, InitCounter());

  int num_tasks = 10000;
  Eigen::Barrier barrier(num_tasks);

  for (int i = 0; i < num_tasks; ++i) {
    thread_pool.Schedule([&counter, &barrier]() {
      Counter& local = counter.local();
      local.inc();
      barrier.Notify();
    });
  }

  barrier.Wait();

  int total = 0;
  std::unordered_set<std::thread::id> unique_threads;

  counter.ForEach([&](std::thread::id id, Counter& cnt) {
    total += cnt.value();
    unique_threads.insert(id);
  });

  VERIFY_IS_EQUAL(total, num_tasks);
  // Not all threads in a pool might be woken up to execute submitted tasks.
  // Also thread_pool.Schedule() might use current thread if queue is full.
  VERIFY_IS_EQUAL(
      unique_threads.size() <= (static_cast<size_t>(num_threads + 1)), true);
}

EIGEN_DECLARE_TEST(cxx11_tensor_thread_local) {
  CALL_SUBTEST(test_simple_thread_local());
  CALL_SUBTEST(test_zero_sized_thread_local());
  CALL_SUBTEST(test_large_number_of_tasks_no_spill());
  CALL_SUBTEST(test_large_number_of_tasks_with_spill());
}
