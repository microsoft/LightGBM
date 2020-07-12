/*
 * yamc_scoped_lock.hpp
 *
 * MIT License
 *
 * Copyright (c) 2018 yohhoy
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef YAMC_SCOPED_LOCK_HPP_
#define YAMC_SCOPED_LOCK_HPP_

#include <mutex>
#include <tuple>
#include <type_traits>


/*
 * std::scoped_lock in C++17 Standard Library
 *
 * - yamc::scoped_lock<MutexTypes...>
 */
namespace yamc {

template <typename... MutexTypes>
class scoped_lock {
  template <std::size_t I = 0>
  typename std::enable_if<I == sizeof...(MutexTypes), void>::type
  invoke_unlock() {}

  template <std::size_t I = 0>
  typename std::enable_if<I < sizeof...(MutexTypes), void>::type
  invoke_unlock()
  {
    std::get<I>(pm_).unlock();
    invoke_unlock<I + 1>();
  }

public:
  explicit scoped_lock(MutexTypes&... m)
    : pm_(m...)
  {
    std::lock(m...);
  }

  explicit scoped_lock(std::adopt_lock_t, MutexTypes&... m)
    : pm_(m...)
  {
  }

  ~scoped_lock()
  {
    invoke_unlock<>();
  }

  scoped_lock(const scoped_lock&) = delete;
  scoped_lock& operator=(const scoped_lock&) = delete;

private:
  std::tuple<MutexTypes&...> pm_;
};


template <>
class scoped_lock<> {
public:
  explicit scoped_lock() = default;
  explicit scoped_lock(std::adopt_lock_t) {}
  ~scoped_lock() = default;

  scoped_lock(const scoped_lock&) = delete;
  scoped_lock& operator=(const scoped_lock&) = delete;
};


template <typename Mutex>
class scoped_lock<Mutex> {
public:
  using mutex_type = Mutex;

  explicit scoped_lock(Mutex& m)
    : m_(m)
  {
    m.lock();
  }

  explicit scoped_lock(std::adopt_lock_t, Mutex& m)
    : m_(m)
  {
  }

  ~scoped_lock()
  {
    m_.unlock();
  }

  scoped_lock(const scoped_lock&) = delete;
  scoped_lock& operator=(const scoped_lock&) = delete;

private:
  Mutex& m_;
};


} // namespace yamc

#endif
