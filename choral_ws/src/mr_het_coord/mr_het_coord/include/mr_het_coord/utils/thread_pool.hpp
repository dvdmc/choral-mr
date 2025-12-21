#ifndef MR_HET_COORD_THREAD_POOL_HPP
#define MR_HET_COORD_THREAD_POOL_HPP

#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
using namespace std;

class ThreadPool {
 public:
  ThreadPool(size_t num_threads = thread::hardware_concurrency(),
             size_t max_tasks = 200)
      : max_tasks_(max_tasks) {
    for (size_t i = 0; i < num_threads; ++i) {
      threads_.emplace_back([this] {
        while (true) {
          function<void()> task;

          {
            unique_lock<mutex> lock(queue_mutex_);

            cv_.wait(lock, [this] { return !tasks_.empty() || stop_; });

            if (stop_ && tasks_.empty()) {
              return;
            }

            task = move(tasks_.front());
            tasks_.pop();
            ++active_tasks_;  // Increment active task counter

            space_cv_.notify_one();
          }

          task();  // Execute the task

          {
            unique_lock<mutex> lock(queue_mutex_);
            --active_tasks_;  // Decrement active task counter
            if (active_tasks_ == 0 && tasks_.empty()) {
              done_cv_.notify_all();  // Notify when all tasks are done
            }
          }
        }
      });
    }
  }

  ~ThreadPool() {
    {
      unique_lock<mutex> lock(queue_mutex_);
      stop_ = true;
    }

    cv_.notify_all();

    for (auto& thread : threads_) {
      thread.join();
    }
  }

  void enqueue(function<void()> task) {
    {
      unique_lock<mutex> lock(queue_mutex_);
      space_cv_.wait(lock, [this] { return tasks_.size() < max_tasks_; });
      tasks_.emplace(move(task));
    }
    cv_.notify_one();
  }

  void wait()  // Wait for all tasks to complete
  {
    unique_lock<mutex> lock(queue_mutex_);
    done_cv_.wait(lock,
                  [this] { return tasks_.empty() && active_tasks_ == 0; });
  }

 private:
  vector<thread> threads_;
  queue<function<void()>> tasks_;
  mutex queue_mutex_;
  condition_variable cv_;
  condition_variable
      done_cv_;  // Condition variable for waiting until all tasks are done
  condition_variable space_cv_;  // Condition variable for waiting until there
                                 // is space in the queue
  bool stop_ = false;
  size_t active_tasks_ = 0;  // Counter to track active tasks
  size_t max_tasks_;
};

#endif