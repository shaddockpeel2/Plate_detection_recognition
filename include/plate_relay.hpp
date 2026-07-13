#pragma once

#include "decoder_thread.hpp"
#include "pipeline_types.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace rkai {

struct PlateRelayGateConfig {
  bool enabled = false;
  bool verbose = false;
  std::string whitelist_path;
  float min_detect_score = 0.50f;
  float min_plate_score = 0.90f;
  int plate_cooldown_sec = 30;
};

struct RelayControllerConfig {
  bool enabled = false;
  bool verbose = false;
  std::string device = "/dev/ttyS0";
  int baud_rate = 9600;
  int slave_address = 1;
  int coil_address = 0;
  int pulse_ms = 2000;
  int io_timeout_ms = 500;
};

struct RelayEvent {
  std::string plate_text;
  float plate_score = 0.0f;
  int track_id = -1;
  int64_t frame_id = 0;
};

struct RelayEventQueueStats {
  int64_t pushed = 0;
  int64_t dropped = 0;
};

class RelayEventQueue {
 public:
  explicit RelayEventQueue(std::size_t capacity);

  bool try_push(RelayEvent event);
  bool pop_for(RelayEvent* event, int timeout_ms);
  void set_accepting(bool accepting);
  bool is_closed() const;
  void close();
  RelayEventQueueStats stats() const;

 private:
  std::size_t capacity_ = 0;
  mutable std::mutex mutex_;
  std::condition_variable not_empty_;
  std::deque<RelayEvent> queue_;
  bool accepting_ = false;
  bool closed_ = false;
  RelayEventQueueStats stats_;
};

class PlateRelayGate {
 public:
  explicit PlateRelayGate(PlateRelayGateConfig config = PlateRelayGateConfig{});

  bool init();
  bool select_event(const Detection& detection, int64_t frame_id, RelayEvent* event);
  void commit_event(const RelayEvent& event);
  const std::string& last_error() const;

 private:
  bool is_in_cooldown(const std::string& plate, std::chrono::steady_clock::time_point now) const;

  PlateRelayGateConfig config_;
  std::unordered_set<std::string> whitelist_;
  std::unordered_map<std::string, std::chrono::steady_clock::time_point> last_triggered_;
  std::string last_error_;
  bool ready_ = false;
};

void relay_controller_thread(const RelayControllerConfig& config, StopFlag& stop, RelayEventQueue& input);
std::string normalize_plate_text(std::string value);

}  // namespace rkai