#pragma once

#include "decoder_thread.hpp"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>

namespace rkai {

struct OledDisplayConfig {
  bool enabled = false;
  bool verbose = false;
  std::string i2c_device = "/dev/i2c-5";
  int i2c_address = 0x3c;
  std::size_t queue_capacity = 2;
  int min_refresh_interval_ms = 120;
};

struct OledPlateEvent {
  std::string plate_text;
  float plate_score = 0.0f;
  int track_id = -1;
  int64_t frame_id = 0;
};

struct OledPlateQueueStats {
  int64_t pushed = 0;
  int64_t dropped = 0;
};

class OledPlateQueue {
 public:
  explicit OledPlateQueue(std::size_t capacity);

  bool try_push(OledPlateEvent event);
  bool pop(OledPlateEvent* event);
  void close();
  OledPlateQueueStats stats() const;

 private:
  std::size_t capacity_ = 0;
  mutable std::mutex mutex_;
  std::condition_variable not_empty_;
  std::deque<OledPlateEvent> queue_;
  bool closed_ = false;
  OledPlateQueueStats stats_;
};

void oled_display_thread(const OledDisplayConfig& config, StopFlag& stop, OledPlateQueue& input);

}  // namespace rkai