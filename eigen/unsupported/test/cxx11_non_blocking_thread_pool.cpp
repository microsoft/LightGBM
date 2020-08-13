// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Dmitry Vyukov <dvyukov@google.com>
// Copyright (C) 2016 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_USE_THREADS
#include "main.h"
#include "Eigen/CXX11/ThreadPool"
#include "Eigen/CXX11/Tensor"

static void test_create_destroy_empty_pool()
{
  // Just create and destroy the pool. This will wind up and tear down worker
  // threads. Ensure there are no issues in that logic.
  for (int i = 0; i < 16; ++i) {
    ThreadPool tp(i);
  }
}


static void test_parallelism(bool allow_spinning)
{
  // Test we never-ever fail to match available tasks with idle threads.
  const int kThreads = 16;  // code below expects that this is a multiple of 4
  ThreadPool tp(kThreads, allow_spinning);
  VERIFY_IS_EQUAL(tp.NumThreads(), kThreads);
  VERIFY_IS_EQUAL(tp.CurrentThreadId(), -1);
  for (int iter = 0; iter < 100; ++iter) {
    std::atomic<int> running(0);
    std::atomic<int> done(0);
    std::atomic<int> phase(0);
    // Schedule kThreads tasks and ensure that they all are running.
    for (int i = 0; i < kThreads; ++i) {
      tp.Schedule([&]() {
        const int thread_id = tp.CurrentThreadId();
        VERIFY_GE(thread_id, 0);
        VERIFY_LE(thread_id, kThreads - 1);
        running++;
        while (phase < 1) {
        }
        done++;
      });
    }
    while (running != kThreads) {
    }
    running = 0;
    phase = 1;
    // Now, while the previous tasks exit, schedule another kThreads tasks and
    // ensure that they are running.
    for (int i = 0; i < kThreads; ++i) {
      tp.Schedule([&, i]() {
        running++;
        while (phase < 2) {
        }
        // When all tasks are running, half of tasks exit, quarter of tasks
        // continue running and quarter of tasks schedule another 2 tasks each.
        // Concurrently main thread schedules another quarter of tasks.
        // This gives us another kThreads tasks and we ensure that they all
        // are running.
        if (i < kThreads / 2) {
        } else if (i < 3 * kThreads / 4) {
          running++;
          while (phase < 3) {
          }
          done++;
        } else {
          for (int j = 0; j < 2; ++j) {
            tp.Schedule([&]() {
              running++;
              while (phase < 3) {
              }
              done++;
            });
          }
        }
        done++;
      });
    }
    while (running != kThreads) {
    }
    running = 0;
    phase = 2;
    for (int i = 0; i < kThreads / 4; ++i) {
      tp.Schedule([&]() {
        running++;
        while (phase < 3) {
        }
        done++;
      });
    }
    while (running != kThreads) {
    }
    phase = 3;
    while (done != 3 * kThreads) {
    }
  }
}


static void test_cancel()
{
  ThreadPool tp(2);

  // Schedule a large number of closure that each sleeps for one second. This
  // will keep the thread pool busy for much longer than the default test timeout.
  for (int i = 0; i < 1000; ++i) {
    tp.Schedule([]() { EIGEN_SLEEP(2000); });
  }

  // Cancel the processing of all the closures that are still pending.
  tp.Cancel();
}

static void test_pool_partitions() {
  const int kThreads = 2;
  ThreadPool tp(kThreads);

  // Assign each thread to its own partition, so that stealing other work only
  // occurs globally when a thread is idle.
  std::vector<std::pair<unsigned, unsigned>> steal_partitions(kThreads);
  for (int i = 0; i < kThreads; ++i) {
    steal_partitions[i] = std::make_pair(i, i + 1);
  }
  tp.SetStealPartitions(steal_partitions);

  std::atomic<int> running(0);
  std::atomic<int> done(0);
  std::atomic<int> phase(0);

  // Schedule kThreads tasks and ensure that they all are running.
  for (int i = 0; i < kThreads; ++i) {
    tp.Schedule([&]() {
      const int thread_id = tp.CurrentThreadId();
      VERIFY_GE(thread_id, 0);
      VERIFY_LE(thread_id, kThreads - 1);
      ++running;
      while (phase < 1) {
      }
      ++done;
    });
  }
  while (running != kThreads) {
  }
  // Schedule each closure to only run on thread 'i' and verify that it does.
  for (int i = 0; i < kThreads; ++i) {
    tp.ScheduleWithHint(
        [&, i]() {
          ++running;
          const int thread_id = tp.CurrentThreadId();
          VERIFY_IS_EQUAL(thread_id, i);
          while (phase < 2) {
          }
          ++done;
        },
        i, i + 1);
  }
  running = 0;
  phase = 1;
  while (running != kThreads) {
  }
  running = 0;
  phase = 2;
}


EIGEN_DECLARE_TEST(cxx11_non_blocking_thread_pool)
{
  CALL_SUBTEST(test_create_destroy_empty_pool());
  CALL_SUBTEST(test_parallelism(true));
  CALL_SUBTEST(test_parallelism(false));
  CALL_SUBTEST(test_cancel());
  CALL_SUBTEST(test_pool_partitions());
}
