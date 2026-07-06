#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>

extern "C" {
#include <rockchip/mpp_frame.h>
}

namespace rkai {

struct DecoderConfig {
  std::string input_path;
  std::size_t output_queue_capacity = 8;
  bool verbose = true;
};

struct DecodedFrame {
  int64_t frame_id = 0;
  int64_t pts = 0;
  int width = 0;
  int height = 0;
  int hor_stride = 0;
  int ver_stride = 0;
  int format = 0;
  int dma_fd = -1;
  MppFrame frame = nullptr;
  std::shared_ptr<void> owner;

  DecodedFrame() = default;
  DecodedFrame(const DecodedFrame&) = delete;
  DecodedFrame& operator=(const DecodedFrame&) = delete;
  DecodedFrame(DecodedFrame&& other) noexcept;
  DecodedFrame& operator=(DecodedFrame&& other) noexcept;
  ~DecodedFrame();
};

class StopFlag {
 public:
  void request_stop();
  bool stop_requested() const;

 private:
  std::atomic<bool> stopped_{false};
};

template <typename T>
class BlockingQueue {
 public:
  explicit BlockingQueue(std::size_t capacity) : capacity_(capacity) {}

  bool push(T item) {
    std::unique_lock<std::mutex> lock(mutex_);
    not_full_.wait(lock, [&] { return closed_ || queue_.size() < capacity_; });
    if (closed_) {
      return false;
    }
    queue_.push_back(std::move(item));
    not_empty_.notify_one();
    return true;
  }

  bool pop(T& item) {
    std::unique_lock<std::mutex> lock(mutex_);
    not_empty_.wait(lock, [&] { return closed_ || !queue_.empty(); });
    if (queue_.empty()) {
      return false;
    }
    item = std::move(queue_.front());
    queue_.pop_front();
    not_full_.notify_one();
    return true;
  }

  void close() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      closed_ = true;
    }
    not_empty_.notify_all();
    not_full_.notify_all();
  }

 private:
  std::size_t capacity_ = 0;
  std::mutex mutex_;
  std::condition_variable not_empty_;
  std::condition_variable not_full_;
  std::deque<T> queue_;
  bool closed_ = false;
};

using DecodedFramePtr = std::shared_ptr<DecodedFrame>;
using DecodedFrameQueue = BlockingQueue<DecodedFramePtr>;

void decoder_thread(const DecoderConfig& config, StopFlag& stop, DecodedFrameQueue& output);

}  // namespace rkai