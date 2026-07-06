#pragma once

#include "decoder_thread.hpp"
#include "pipeline_types.hpp"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace rkai {

struct UploadEventConfig {
  bool enabled = false;
  std::string device_id = "rk3588-001";
  float min_detect_score = 0.50f;
  float min_plate_score = 0.80f;
  int track_cooldown_frames = 90;
  int plate_cooldown_frames = 1800;
  bool verbose = false;
};

struct UploadEventThreadConfig {
  bool enabled = false;
  bool verbose = false;
  std::string snapshot_dir = "./upload_snapshots";
  std::string upload_url;
  std::string public_base_url;
  int jpeg_quality = 85;
  long http_timeout_ms = 3000;
  bool mqtt_enabled = false;
  std::string mqtt_host;
  int mqtt_port = 1883;
  std::string mqtt_username;
  std::string mqtt_password;
  std::string mqtt_client_id = "rk3588-001";
  std::string mqtt_topic = "devices/rk3588-001/plate/events";
  std::string mqtt_heartbeat_topic = "devices/rk3588-001/status/heartbeat";
  std::string mqtt_error_topic = "devices/rk3588-001/status/error";
  int mqtt_heartbeat_interval_sec = 30;
  long mqtt_timeout_ms = 3000;
};

struct UploadEventQueueStats {
  int64_t pushed = 0;
  int64_t dropped = 0;
};

struct UploadEvent {
  std::string device_id;
  std::string event_id;
  int64_t frame_id = 0;
  int64_t pts = 0;
  int image_width = 0;
  int image_height = 0;
  int track_id = -1;
  int class_id = 0;
  float detect_score = 0.0f;
  std::string plate_text;
  float plate_score = 0.0f;
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  std::string image_path;
  std::string image_url;
  DecodedFramePtr frame;
};

class UploadEventGate {
 public:
  explicit UploadEventGate(UploadEventConfig config = UploadEventConfig{});

  std::vector<UploadEvent> select_events(const DecodedFramePtr& frame, const std::vector<Detection>& detections);
  void log_event(const UploadEvent& event) const;

 private:
  bool should_emit(const DecodedFrame& frame, const Detection& detection) const;
  bool is_in_cooldown(const DecodedFrame& frame, const Detection& detection) const;
  void remember(const UploadEvent& event);
  UploadEvent make_event(const DecodedFramePtr& frame, const Detection& detection) const;

  UploadEventConfig config_;
  std::unordered_map<int, int64_t> last_track_frame_;
  std::unordered_map<std::string, int64_t> last_plate_frame_;
};

class UploadEventQueue {
 public:
  explicit UploadEventQueue(std::size_t capacity);

  bool try_push(UploadEvent event);
  bool pop(UploadEvent& event);
  void close();
  UploadEventQueueStats stats() const;

 private:
  std::size_t capacity_ = 0;
  mutable std::mutex mutex_;
  std::condition_variable not_empty_;
  std::deque<UploadEvent> queue_;
  bool closed_ = false;
  UploadEventQueueStats stats_;
};

void upload_event_thread(const UploadEventThreadConfig& config, StopFlag& stop, UploadEventQueue& input);

}  // namespace rkai