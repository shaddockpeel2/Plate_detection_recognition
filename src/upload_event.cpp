#include "upload_event.hpp"

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <netdb.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

extern "C" {
#include <rockchip/mpp_buffer.h>
}

namespace rkai {
namespace {

struct HttpUrl {
  std::string host;
  std::string port = "80";
  std::string path = "/";
};

float clamp_non_negative(float value) {
  return std::max(0.0f, value);
}

int clamp_int(float value, int lo, int hi) {
  return std::max(lo, std::min(hi, static_cast<int>(value + 0.5f)));
}

std::string trim_trailing_slash(std::string value) {
  while (!value.empty() && value[value.size() - 1] == '/') {
    value.resize(value.size() - 1);
  }
  return value;
}

std::string origin_from_url(const std::string& url) {
  const std::string::size_type scheme = url.find("://");
  if (scheme == std::string::npos) {
    return std::string();
  }
  const std::string::size_type host_start = scheme + 3;
  const std::string::size_type path_start = url.find('/', host_start);
  if (path_start == std::string::npos) {
    return url;
  }
  return url.substr(0, path_start);
}

bool parse_http_url(const std::string& url, HttpUrl* parsed) {
  if (parsed == nullptr) {
    return false;
  }
  const std::string prefix = "http://";
  if (url.compare(0, prefix.size(), prefix) != 0) {
    return false;
  }

  const std::string::size_type host_start = prefix.size();
  const std::string::size_type path_start = url.find('/', host_start);
  const std::string host_port = path_start == std::string::npos ? url.substr(host_start) : url.substr(host_start, path_start - host_start);
  parsed->path = path_start == std::string::npos ? "/" : url.substr(path_start);

  const std::string::size_type colon = host_port.rfind(':');
  if (colon == std::string::npos) {
    parsed->host = host_port;
    parsed->port = "80";
  } else {
    parsed->host = host_port.substr(0, colon);
    parsed->port = host_port.substr(colon + 1);
  }
  return !parsed->host.empty() && !parsed->port.empty() && !parsed->path.empty();
}

std::string extract_json_string(const std::string& json, const std::string& key) {
  const std::string marker = "\"" + key + "\"";
  const std::string::size_type key_pos = json.find(marker);
  if (key_pos == std::string::npos) {
    return std::string();
  }
  const std::string::size_type colon_pos = json.find(':', key_pos + marker.size());
  if (colon_pos == std::string::npos) {
    return std::string();
  }
  const std::string::size_type quote_start = json.find('"', colon_pos + 1);
  if (quote_start == std::string::npos) {
    return std::string();
  }
  const std::string::size_type quote_end = json.find('"', quote_start + 1);
  if (quote_end == std::string::npos) {
    return std::string();
  }
  return json.substr(quote_start + 1, quote_end - quote_start - 1);
}

std::string make_snapshot_name(const UploadEvent& event) {
  std::ostringstream name;
  name << event.device_id << '_' << event.frame_id << '_' << event.track_id << ".jpg";
  return name.str();
}

bool ensure_directory(const std::string& path) {
  if (path.empty()) {
    return false;
  }

  std::string current;
  for (size_t i = 0; i < path.size(); ++i) {
    current.push_back(path[i]);
    if (path[i] != '/' && i + 1 != path.size()) {
      continue;
    }
    if (current.empty() || current == "/") {
      continue;
    }
    if (mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) {
      return false;
    }
  }
  return true;
}

bool read_file(const std::string& path, std::string* content) {
  if (content == nullptr) {
    return false;
  }
  std::ifstream file(path.c_str(), std::ios::binary);
  if (!file) {
    return false;
  }
  std::ostringstream buffer;
  buffer << file.rdbuf();
  *content = buffer.str();
  return true;
}

bool send_all(int fd, const std::string& data) {
  const char* cursor = data.data();
  size_t remaining = data.size();
  while (remaining > 0) {
    const ssize_t sent = send(fd, cursor, remaining, 0);
    if (sent <= 0) {
      return false;
    }
    cursor += sent;
    remaining -= static_cast<size_t>(sent);
  }
  return true;
}

bool recv_all(int fd, std::string* response) {
  if (response == nullptr) {
    return false;
  }
  char buffer[4096];
  for (;;) {
    const ssize_t n = recv(fd, buffer, sizeof(buffer), 0);
    if (n > 0) {
      response->append(buffer, static_cast<size_t>(n));
      continue;
    }
    return n == 0;
  }
}

int connect_tcp(const std::string& host, const std::string& port, long timeout_ms) {
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* result = nullptr;
  if (getaddrinfo(host.c_str(), port.c_str(), &hints, &result) != 0 || result == nullptr) {
    return -1;
  }

  int fd = -1;
  for (addrinfo* it = result; it != nullptr; it = it->ai_next) {
    fd = socket(it->ai_family, it->ai_socktype, it->ai_protocol);
    if (fd < 0) {
      continue;
    }
    timeval timeout{};
    timeout.tv_sec = static_cast<long>(timeout_ms / 1000);
    timeout.tv_usec = static_cast<long>((timeout_ms % 1000) * 1000);
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    if (connect(fd, it->ai_addr, it->ai_addrlen) == 0) {
      break;
    }
    close(fd);
    fd = -1;
  }
  freeaddrinfo(result);
  return fd;
}

void append_mqtt_string(std::string* packet, const std::string& value) {
  if (packet == nullptr || value.size() > 65535) {
    return;
  }
  packet->push_back(static_cast<char>((value.size() >> 8) & 0xff));
  packet->push_back(static_cast<char>(value.size() & 0xff));
  packet->append(value);
}

void append_mqtt_remaining_length(std::string* packet, size_t length) {
  do {
    uint8_t encoded = static_cast<uint8_t>(length % 128);
    length /= 128;
    if (length > 0) {
      encoded |= 128;
    }
    packet->push_back(static_cast<char>(encoded));
  } while (length > 0);
}

std::string mqtt_packet(uint8_t packet_type, const std::string& payload) {
  std::string packet;
  packet.push_back(static_cast<char>(packet_type));
  append_mqtt_remaining_length(&packet, payload.size());
  packet.append(payload);
  return packet;
}

std::string json_escape(const std::string& value) {
  std::string out;
  out.reserve(value.size() + 8);
  for (char c : value) {
    switch (c) {
      case '\\': out += "\\\\"; break;
      case '"': out += "\\\""; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default: out.push_back(c); break;
    }
  }
  return out;
}

int64_t now_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

std::string make_mqtt_payload(const UploadEvent& event) {
  std::ostringstream json;
  json << "{"
       << "\"device_id\":\"" << json_escape(event.device_id) << "\","
       << "\"event_id\":\"" << json_escape(event.event_id) << "\","
       << "\"frame_id\":" << event.frame_id << ","
       << "\"timestamp_ms\":" << now_ms() << ","
       << "\"track_id\":" << event.track_id << ","
       << "\"plate_text\":\"" << json_escape(event.plate_text) << "\","
       << "\"plate_score\":" << event.plate_score << ","
       << "\"detect_score\":" << event.detect_score << ","
       << "\"bbox\":{"
       << "\"x1\":" << event.x1 << ","
       << "\"y1\":" << event.y1 << ","
       << "\"x2\":" << event.x2 << ","
       << "\"y2\":" << event.y2 << "},"
       << "\"image_url\":\"" << json_escape(event.image_url) << "\""
       << "}";
  return json.str();
}

bool mqtt_connect(int fd, const UploadEventThreadConfig& config) {
  std::string variable;
  append_mqtt_string(&variable, "MQTT");
  variable.push_back(4);
  uint8_t flags = 0x02;
  if (!config.mqtt_username.empty()) {
    flags |= 0x80;
  }
  if (!config.mqtt_password.empty()) {
    flags |= 0x40;
  }
  variable.push_back(static_cast<char>(flags));
  variable.push_back(0);
  variable.push_back(30);

  std::string payload;
  append_mqtt_string(&payload, config.mqtt_client_id);
  if (!config.mqtt_username.empty()) {
    append_mqtt_string(&payload, config.mqtt_username);
  }
  if (!config.mqtt_password.empty()) {
    append_mqtt_string(&payload, config.mqtt_password);
  }

  const std::string packet = mqtt_packet(0x10, variable + payload);
  if (!send_all(fd, packet)) {
    return false;
  }

  char ack[4] = {0};
  const ssize_t n = recv(fd, ack, sizeof(ack), 0);
  return n == 4 && static_cast<uint8_t>(ack[0]) == 0x20 && static_cast<uint8_t>(ack[1]) == 0x02 &&
         static_cast<uint8_t>(ack[3]) == 0x00;
}

bool publish_mqtt_payload(const UploadEventThreadConfig& config, const std::string& topic, const std::string& message) {
  if (!config.mqtt_enabled || config.mqtt_host.empty() || topic.empty()) {
    return false;
  }

  const int fd = connect_tcp(config.mqtt_host, std::to_string(config.mqtt_port), config.mqtt_timeout_ms);
  if (fd < 0) {
    return false;
  }

  bool ok = mqtt_connect(fd, config);
  if (ok) {
    std::string payload;
    append_mqtt_string(&payload, topic);
    payload += message;
    ok = send_all(fd, mqtt_packet(0x30, payload));
  }

  if (ok) {
    std::string packet;
    packet.push_back(static_cast<char>(0xe0));
    packet.push_back(0);
    send_all(fd, packet);
  }
  close(fd);
  return ok;
}

bool publish_mqtt_event(const UploadEventThreadConfig& config, const UploadEvent& event) {
  return publish_mqtt_payload(config, config.mqtt_topic, make_mqtt_payload(event));
}

std::string make_status_payload(const UploadEventThreadConfig& config,
                                const char* type,
                                int64_t consumed,
                                int64_t snapshot_saved,
                                int64_t snapshot_failed,
                                int64_t http_uploaded,
                                int64_t http_failed,
                                int64_t mqtt_published,
                                int64_t mqtt_failed,
                                const UploadEventQueueStats& queue_stats,
                                const std::string& detail) {
  std::ostringstream json;
  json << "{"
       << "\"device_id\":\"" << json_escape(config.mqtt_client_id) << "\","
       << "\"type\":\"" << type << "\","
       << "\"timestamp_ms\":" << now_ms() << ","
       << "\"consumed\":" << consumed << ","
       << "\"snapshot_saved\":" << snapshot_saved << ","
       << "\"snapshot_failed\":" << snapshot_failed << ","
       << "\"http_uploaded\":" << http_uploaded << ","
       << "\"http_failed\":" << http_failed << ","
       << "\"mqtt_published\":" << mqtt_published << ","
       << "\"mqtt_failed\":" << mqtt_failed << ","
       << "\"queue_pushed\":" << queue_stats.pushed << ","
       << "\"queue_dropped\":" << queue_stats.dropped << ","
       << "\"detail\":\"" << json_escape(detail) << "\""
       << "}";
  return json.str();
}

bool encode_nv12_crop_to_jpeg(const DecodedFrame& frame,
                              const UploadEvent& event,
                              const std::string& output_path,
                              int jpeg_quality) {
  if (!frame.frame || frame.width <= 0 || frame.height <= 0 || frame.hor_stride <= 0 || frame.ver_stride <= 0) {
    return false;
  }

  MppBuffer buffer = mpp_frame_get_buffer(frame.frame);
  if (!buffer) {
    return false;
  }
  auto* base = static_cast<uint8_t*>(mpp_buffer_get_ptr(buffer));
  if (!base) {
    return false;
  }

  const int x1 = clamp_int(event.x1, 0, frame.width - 1);
  const int y1 = clamp_int(event.y1, 0, frame.height - 1);
  const int x2 = clamp_int(event.x2, x1 + 1, frame.width);
  const int y2 = clamp_int(event.y2, y1 + 1, frame.height);
  const int crop_width = x2 - x1;
  const int crop_height = y2 - y1;
  if (crop_width <= 1 || crop_height <= 1) {
    return false;
  }

  cv::Mat bgr;
  cv::Mat nv12(frame.ver_stride + frame.ver_stride / 2, frame.hor_stride, CV_8UC1, base);
  cv::cvtColor(nv12, bgr, cv::COLOR_YUV2BGR_NV12);

  const cv::Rect crop_rect(x1, y1, crop_width, crop_height);
  const cv::Mat crop = bgr(crop_rect).clone();
  std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, std::max(1, std::min(100, jpeg_quality))};
  return cv::imwrite(output_path, crop, params);
}

bool attach_local_snapshot(const UploadEventThreadConfig& config, UploadEvent* event) {
  if (event == nullptr || !event->frame) {
    return false;
  }
  if (!ensure_directory(config.snapshot_dir)) {
    return false;
  }

  const std::string filename = make_snapshot_name(*event);
  const std::string dir = trim_trailing_slash(config.snapshot_dir);
  event->image_path = dir + "/" + filename;
  if (!config.public_base_url.empty()) {
    event->image_url = trim_trailing_slash(config.public_base_url) + "/" + filename;
  }

  return encode_nv12_crop_to_jpeg(*event->frame, *event, event->image_path, config.jpeg_quality);
}

bool upload_snapshot_http(const UploadEventThreadConfig& config, UploadEvent* event) {
  if (event == nullptr || config.upload_url.empty() || event->image_path.empty()) {
    return false;
  }

  HttpUrl url;
  if (!parse_http_url(config.upload_url, &url)) {
    return false;
  }

  std::string image;
  if (!read_file(event->image_path, &image)) {
    return false;
  }

  const std::string boundary = "----rk3588-upload-boundary";
  const std::string filename = make_snapshot_name(*event);
  std::ostringstream body;
  body << "--" << boundary << "\r\n";
  body << "Content-Disposition: form-data; name=\"file\"; filename=\"" << filename << "\"\r\n";
  body << "Content-Type: image/jpeg\r\n\r\n";
  body << image;
  body << "\r\n--" << boundary << "--\r\n";
  const std::string body_data = body.str();

  std::ostringstream request;
  request << "POST " << url.path << " HTTP/1.1\r\n";
  request << "Host: " << url.host << ":" << url.port << "\r\n";
  request << "Connection: close\r\n";
  request << "Content-Type: multipart/form-data; boundary=" << boundary << "\r\n";
  request << "Content-Length: " << body_data.size() << "\r\n\r\n";
  request << body_data;

  int fd = connect_tcp(url.host, url.port, config.http_timeout_ms);
  if (fd < 0) {
    return false;
  }

  std::string response;
  const bool ok = send_all(fd, request.str()) && recv_all(fd, &response);
  close(fd);
  if (!ok || response.find("HTTP/1.1 2") != 0) {
    return false;
  }

  const std::string returned_url = extract_json_string(response, "url");
  if (returned_url.empty()) {
    return true;
  }
  if (returned_url.find("://") != std::string::npos) {
    event->image_url = returned_url;
  } else if (!config.public_base_url.empty()) {
    event->image_url = trim_trailing_slash(config.public_base_url) + "/" + returned_url.substr(returned_url[0] == '/' ? 1 : 0);
  } else {
    event->image_url = trim_trailing_slash(origin_from_url(config.upload_url)) + returned_url;
  }
  return true;
}

}  // namespace

UploadEventGate::UploadEventGate(UploadEventConfig config) : config_(std::move(config)) {}

std::vector<UploadEvent> UploadEventGate::select_events(const DecodedFramePtr& frame,
                                                        const std::vector<Detection>& detections) {
  std::vector<UploadEvent> events;
  if (!config_.enabled || !frame) {
    return events;
  }

  for (const auto& detection : detections) {
    if (!should_emit(*frame, detection) || is_in_cooldown(*frame, detection)) {
      continue;
    }

    UploadEvent event = make_event(frame, detection);
    remember(event);
    events.push_back(std::move(event));
  }
  return events;
}

void UploadEventGate::log_event(const UploadEvent& event) const {
  if (!config_.verbose) {
    return;
  }

  std::fprintf(stderr,
               "[UPLOAD_EVENT] {\"device_id\":\"%s\",\"event_id\":\"%s\",\"frame_id\":%ld,\"pts\":%ld,\"track_id\":%d,\"class_id\":%d,\"plate_text\":\"%s\",\"plate_score\":%.3f,\"detect_score\":%.3f,\"bbox\":{\"x1\":%.1f,\"y1\":%.1f,\"x2\":%.1f,\"y2\":%.1f},\"image_width\":%d,\"image_height\":%d,\"image_path\":\"%s\",\"image_url\":\"%s\"}\n",
               event.device_id.c_str(),
               event.event_id.c_str(),
               event.frame_id,
               event.pts,
               event.track_id,
               event.class_id,
               event.plate_text.c_str(),
               event.plate_score,
               event.detect_score,
               event.x1,
               event.y1,
               event.x2,
               event.y2,
               event.image_width,
               event.image_height,
               event.image_path.c_str(),
               event.image_url.c_str());
}

bool UploadEventGate::should_emit(const DecodedFrame&, const Detection& detection) const {
  if (detection.track_id <= 0) {
    return false;
  }
  if (detection.score < config_.min_detect_score) {
    return false;
  }
  if (!detection.plate_recognized || detection.plate_text.empty()) {
    return false;
  }
  if (detection.plate_score < config_.min_plate_score) {
    return false;
  }
  if (detection.x2 <= detection.x1 || detection.y2 <= detection.y1) {
    return false;
  }
  return true;
}

bool UploadEventGate::is_in_cooldown(const DecodedFrame& frame, const Detection& detection) const {
  const auto track_it = last_track_frame_.find(detection.track_id);
  if (track_it != last_track_frame_.end() && frame.frame_id - track_it->second < config_.track_cooldown_frames) {
    return true;
  }

  const auto plate_it = last_plate_frame_.find(detection.plate_text);
  if (plate_it != last_plate_frame_.end() && frame.frame_id - plate_it->second < config_.plate_cooldown_frames) {
    return true;
  }
  return false;
}

void UploadEventGate::remember(const UploadEvent& event) {
  last_track_frame_[event.track_id] = event.frame_id;
  last_plate_frame_[event.plate_text] = event.frame_id;
}

UploadEvent UploadEventGate::make_event(const DecodedFramePtr& frame, const Detection& detection) const {
  UploadEvent event;
  event.device_id = config_.device_id;
  event.frame_id = frame->frame_id;
  event.pts = frame->pts;
  event.image_width = frame->width;
  event.image_height = frame->height;
  event.track_id = detection.track_id;
  event.class_id = detection.class_id;
  event.detect_score = detection.score;
  event.plate_text = detection.plate_text;
  event.plate_score = detection.plate_score;
  event.x1 = clamp_non_negative(detection.x1);
  event.y1 = clamp_non_negative(detection.y1);
  event.x2 = clamp_non_negative(detection.x2);
  event.y2 = clamp_non_negative(detection.y2);
  event.frame = frame;

  std::ostringstream event_id;
  event_id << event.device_id << '-' << event.frame_id << '-' << event.track_id;
  event.event_id = event_id.str();
  return event;
}

UploadEventQueue::UploadEventQueue(std::size_t capacity) : capacity_(capacity) {}

bool UploadEventQueue::try_push(UploadEvent event) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (closed_ || capacity_ == 0 || queue_.size() >= capacity_) {
    ++stats_.dropped;
    return false;
  }

  queue_.push_back(std::move(event));
  ++stats_.pushed;
  not_empty_.notify_one();
  return true;
}

bool UploadEventQueue::pop(UploadEvent& event) {
  std::unique_lock<std::mutex> lock(mutex_);
  not_empty_.wait(lock, [&] { return closed_ || !queue_.empty(); });
  if (queue_.empty()) {
    return false;
  }

  event = std::move(queue_.front());
  queue_.pop_front();
  return true;
}

void UploadEventQueue::close() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
  }
  not_empty_.notify_all();
}

UploadEventQueueStats UploadEventQueue::stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return stats_;
}

void upload_event_thread(const UploadEventThreadConfig& config, StopFlag& stop, UploadEventQueue& input) {
  const auto stage_start = std::chrono::steady_clock::now();
  int64_t consumed = 0;
  int64_t snapshot_saved = 0;
  int64_t snapshot_failed = 0;
  int64_t http_uploaded = 0;
  int64_t http_failed = 0;
  int64_t mqtt_published = 0;
  int64_t mqtt_failed = 0;
  auto last_heartbeat = stage_start - std::chrono::seconds(config.mqtt_heartbeat_interval_sec);
  UploadEvent event;

  while (!stop.stop_requested() && input.pop(event)) {
    ++consumed;
    bool event_has_error = false;
    std::string error_detail;
    if (attach_local_snapshot(config, &event)) {
      ++snapshot_saved;
      if (!config.upload_url.empty()) {
        if (upload_snapshot_http(config, &event)) {
          ++http_uploaded;
        } else {
          ++http_failed;
          event_has_error = true;
          error_detail = "http_upload_failed";
        }
      }
      if (config.mqtt_enabled) {
        if (publish_mqtt_event(config, event)) {
          ++mqtt_published;
        } else {
          ++mqtt_failed;
        }
      }
    } else {
      ++snapshot_failed;
      event_has_error = true;
      error_detail = "snapshot_failed";
    }

    if (config.mqtt_enabled) {
      const UploadEventQueueStats queue_stats = input.stats();
      if (event_has_error && !config.mqtt_error_topic.empty()) {
        const std::string payload = make_status_payload(config,
                                                        "error",
                                                        consumed,
                                                        snapshot_saved,
                                                        snapshot_failed,
                                                        http_uploaded,
                                                        http_failed,
                                                        mqtt_published,
                                                        mqtt_failed,
                                                        queue_stats,
                                                        error_detail);
        if (!publish_mqtt_payload(config, config.mqtt_error_topic, payload)) {
          ++mqtt_failed;
        }
      }

      const auto now = std::chrono::steady_clock::now();
      if (config.mqtt_heartbeat_interval_sec > 0 &&
          now - last_heartbeat >= std::chrono::seconds(config.mqtt_heartbeat_interval_sec) &&
          !config.mqtt_heartbeat_topic.empty()) {
        const std::string payload = make_status_payload(config,
                                                        "heartbeat",
                                                        consumed,
                                                        snapshot_saved,
                                                        snapshot_failed,
                                                        http_uploaded,
                                                        http_failed,
                                                        mqtt_published,
                                                        mqtt_failed,
                                                        queue_stats,
                                                        "ok");
        if (publish_mqtt_payload(config, config.mqtt_heartbeat_topic, payload)) {
          last_heartbeat = now;
        } else {
          ++mqtt_failed;
        }
      }
    }

    if (config.enabled && config.verbose) {
      std::fprintf(stderr,
                   "[UPLOAD_THREAD] consumed event_id=%s frame=%ld track=%d plate=%s image_path=%s image_url=%s\n",
                   event.event_id.c_str(),
                   event.frame_id,
                   event.track_id,
                   event.plate_text.c_str(),
                   event.image_path.c_str(),
                   event.image_url.c_str());
    }
  }

  const auto stage_end = std::chrono::steady_clock::now();
  const double elapsed_ms = std::chrono::duration<double, std::milli>(stage_end - stage_start).count();
  const UploadEventQueueStats queue_stats = input.stats();
  std::fprintf(stderr,
               "[PERF] upload_event_thread consumed=%ld snapshot_saved=%ld snapshot_failed=%ld http_uploaded=%ld http_failed=%ld mqtt_published=%ld mqtt_failed=%ld pushed=%ld dropped=%ld elapsed_ms=%.3f\n",
               consumed,
               snapshot_saved,
               snapshot_failed,
               http_uploaded,
               http_failed,
               mqtt_published,
               mqtt_failed,
               queue_stats.pushed,
               queue_stats.dropped,
               elapsed_ms);
}

}  // namespace rkai