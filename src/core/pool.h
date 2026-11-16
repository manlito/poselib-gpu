// A small persistent thread pool with one operation: run fn(i) for i in [0, n), items handed out one
// at a time. In WebAssembly (single-threaded build) it runs inline.
#pragma once

#include <atomic>
#include <cstddef>
#include <functional>

#ifndef __EMSCRIPTEN__
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>
#endif

namespace jpo {

class ThreadPool {
 public:
#ifdef __EMSCRIPTEN__
  explicit ThreadPool(int) {}
  int size() const { return 1; }
  void parallel_for(size_t n, const std::function<void(size_t)>& fn) {
    for (size_t i = 0; i < n; ++i) fn(i);
  }
#else
  explicit ThreadPool(int threads) {
    int n = threads > 0 ? threads : static_cast<int>(std::thread::hardware_concurrency());
    if (n < 1) n = 1;
    size_ = n;
    for (int i = 1; i < n; ++i) workers_.emplace_back([this] { worker(); });
  }
  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
      ++generation_;
    }
    cv_.notify_all();
    for (auto& t : workers_) t.join();
  }
  int size() const { return size_; }

  void parallel_for(size_t n, const std::function<void(size_t)>& fn) {
    if (n == 0) return;
    if (n == 1 || workers_.empty()) {
      for (size_t i = 0; i < n; ++i) fn(i);
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mu_);
      fn_ = &fn;
      n_ = n;
      next_.store(0);
      pending_ = workers_.size();
      ++generation_;
    }
    cv_.notify_all();
    drain();
    std::unique_lock<std::mutex> lock(mu_);
    done_.wait(lock, [this] { return pending_ == 0; });
    fn_ = nullptr;
  }

 private:
  void drain() {
    for (;;) {
      const size_t i = next_.fetch_add(1);
      if (i >= n_) break;
      (*fn_)(i);
    }
  }
  void worker() {
    size_t seen = 0;
    for (;;) {
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&] { return generation_ != seen; });
        seen = generation_;
        if (stop_) return;
      }
      drain();
      {
        std::lock_guard<std::mutex> lock(mu_);
        --pending_;
      }
      done_.notify_one();
    }
  }

  int size_ = 1;
  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable cv_, done_;
  const std::function<void(size_t)>* fn_ = nullptr;
  size_t n_ = 0;
  std::atomic<size_t> next_{0};
  size_t pending_ = 0;
  size_t generation_ = 0;
  bool stop_ = false;
#endif
};

}  // namespace jpo
