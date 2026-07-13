#include "plate_relay.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <poll.h>
#include <termios.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace rkai {
namespace {

std::string trim(std::string value) {
  const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char c) { return std::isspace(c) != 0; });
  const auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char c) { return std::isspace(c) != 0; }).base();
  return first >= last ? std::string() : std::string(first, last);
}

uint16_t modbus_crc16(const uint8_t* data, std::size_t size) {
  uint16_t crc = 0xffff;
  for (std::size_t i = 0; i < size; ++i) {
    crc ^= data[i];
    for (int bit = 0; bit < 8; ++bit) {
      crc = (crc & 1U) != 0 ? static_cast<uint16_t>((crc >> 1U) ^ 0xa001U) : static_cast<uint16_t>(crc >> 1U);
    }
  }
  return crc;
}

std::vector<uint8_t> coil_command(int slave_address, int coil_address, bool enabled) {
  std::vector<uint8_t> frame{
      static_cast<uint8_t>(slave_address),
      0x05,
      static_cast<uint8_t>((coil_address >> 8) & 0xff),
      static_cast<uint8_t>(coil_address & 0xff),
      static_cast<uint8_t>(enabled ? 0xff : 0x00),
      0x00,
  };
  const uint16_t crc = modbus_crc16(frame.data(), frame.size());
  frame.push_back(static_cast<uint8_t>(crc & 0xff));
  frame.push_back(static_cast<uint8_t>((crc >> 8) & 0xff));
  return frame;
}

speed_t termios_speed(int baud_rate) {
  switch (baud_rate) {
    case 1200: return B1200;
    case 2400: return B2400;
    case 4800: return B4800;
    case 9600: return B9600;
    case 19200: return B19200;
    case 115200: return B115200;
    default: return 0;
  }
}

bool write_all(int fd, const std::vector<uint8_t>& frame) {
  std::size_t offset = 0;
  while (offset < frame.size()) {
    const ssize_t written = write(fd, frame.data() + offset, frame.size() - offset);
    if (written > 0) {
      offset += static_cast<std::size_t>(written);
      continue;
    }
    if (written < 0 && errno == EINTR) {
      continue;
    }
    return false;
  }
  return true;
}

bool read_exact(int fd, std::vector<uint8_t>* response, std::size_t expected_size, int timeout_ms) {
  if (response == nullptr) {
    return false;
  }
  response->clear();
  while (response->size() < expected_size) {
    pollfd descriptor{};
    descriptor.fd = fd;
    descriptor.events = POLLIN;
    const int polled = poll(&descriptor, 1, timeout_ms);
    if (polled <= 0 || (descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
      return false;
    }

    std::array<uint8_t, 32> buffer{};
    const ssize_t read_size = read(fd, buffer.data(), std::min(buffer.size(), expected_size - response->size()));
    if (read_size > 0) {
      response->insert(response->end(), buffer.begin(), buffer.begin() + read_size);
      continue;
    }
    if (read_size < 0 && errno == EINTR) {
      continue;
    }
    return false;
  }
  return true;
}

class ModbusRelayPort {
 public:
  ~ModbusRelayPort() { close_port(); }

  bool open_port(const RelayControllerConfig& config) {
    close_port();
    const speed_t speed = termios_speed(config.baud_rate);
    if (speed == 0) {
      std::fprintf(stderr, "[RELAY] unsupported baud rate: %d\n", config.baud_rate);
      return false;
    }

    fd_ = open(config.device.c_str(), O_RDWR | O_NOCTTY | O_CLOEXEC);
    if (fd_ < 0) {
      std::fprintf(stderr, "[RELAY] cannot open %s: %s\n", config.device.c_str(), std::strerror(errno));
      return false;
    }

    termios options{};
    if (tcgetattr(fd_, &options) != 0) {
      std::fprintf(stderr, "[RELAY] cannot read serial settings: %s\n", std::strerror(errno));
      close_port();
      return false;
    }
    cfmakeraw(&options);
    options.c_cflag &= static_cast<tcflag_t>(~(PARENB | CSTOPB | CSIZE | CRTSCTS));
    options.c_cflag |= static_cast<tcflag_t>(CS8 | CLOCAL | CREAD);
    options.c_iflag &= static_cast<tcflag_t>(~(IXON | IXOFF | IXANY));
    options.c_cc[VMIN] = 0;
    options.c_cc[VTIME] = 0;
    if (cfsetispeed(&options, speed) != 0 || cfsetospeed(&options, speed) != 0 || tcsetattr(fd_, TCSANOW, &options) != 0) {
      std::fprintf(stderr, "[RELAY] cannot configure serial port: %s\n", std::strerror(errno));
      close_port();
      return false;
    }
    return true;
  }

  bool write_coil(const RelayControllerConfig& config, bool enabled) {
    if (fd_ < 0) {
      return false;
    }
    const std::vector<uint8_t> request = coil_command(config.slave_address, config.coil_address, enabled);
    tcflush(fd_, TCIOFLUSH);
    if (!write_all(fd_, request)) {
      std::fprintf(stderr, "[RELAY] write failed: %s\n", std::strerror(errno));
      return false;
    }

    std::vector<uint8_t> response;
    if (!read_exact(fd_, &response, request.size(), config.io_timeout_ms) || response != request) {
      std::fprintf(stderr, "[RELAY] invalid response while turning %s\n", enabled ? "on" : "off");
      return false;
    }
    return true;
  }

  void close_port() {
    if (fd_ >= 0) {
      close(fd_);
      fd_ = -1;
    }
  }

 private:
  int fd_ = -1;
};

}  // namespace

std::string normalize_plate_text(std::string value) {
  value = trim(std::move(value));
  for (char& c : value) {
    const unsigned char byte = static_cast<unsigned char>(c);
    if (byte < 0x80) {
      c = static_cast<char>(std::toupper(byte));
    }
  }
  return value;
}

RelayEventQueue::RelayEventQueue(std::size_t capacity) : capacity_(capacity) {}

bool RelayEventQueue::try_push(RelayEvent event) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (closed_ || !accepting_ || capacity_ == 0 || !queue_.empty()) {
    ++stats_.dropped;
    return false;
  }
  queue_.push_back(std::move(event));
  ++stats_.pushed;
  not_empty_.notify_one();
  return true;
}

bool RelayEventQueue::pop_for(RelayEvent* event, int timeout_ms) {
  if (event == nullptr) {
    return false;
  }
  std::unique_lock<std::mutex> lock(mutex_);
  not_empty_.wait_for(lock, std::chrono::milliseconds(std::max(0, timeout_ms)), [&] { return closed_ || !queue_.empty(); });
  if (queue_.empty()) {
    return false;
  }
  *event = std::move(queue_.front());
  queue_.pop_front();
  return true;
}

void RelayEventQueue::set_accepting(bool accepting) {
  std::lock_guard<std::mutex> lock(mutex_);
  accepting_ = accepting;
  if (!accepting_) {
    queue_.clear();
  }
}

bool RelayEventQueue::is_closed() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return closed_;
}

void RelayEventQueue::close() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
    accepting_ = false;
    queue_.clear();
  }
  not_empty_.notify_all();
}

RelayEventQueueStats RelayEventQueue::stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return stats_;
}

PlateRelayGate::PlateRelayGate(PlateRelayGateConfig config) : config_(std::move(config)) {}

bool PlateRelayGate::init() {
  whitelist_.clear();
  last_triggered_.clear();
  last_error_.clear();
  ready_ = false;
  if (!config_.enabled) {
    return true;
  }
  if (config_.whitelist_path.empty()) {
    last_error_ = "missing whitelist path";
    return false;
  }

  std::ifstream file(config_.whitelist_path);
  if (!file) {
    last_error_ = "cannot open whitelist: " + config_.whitelist_path;
    return false;
  }

  std::string line;
  while (std::getline(file, line)) {
    const std::string plate = normalize_plate_text(line);
    if (plate.empty() || plate[0] == '#') {
      continue;
    }
    whitelist_.insert(plate);
  }
  if (whitelist_.empty()) {
    last_error_ = "whitelist is empty";
    return false;
  }

  ready_ = true;
  std::fprintf(stderr, "[RELAY] whitelist loaded: %zu plates\n", whitelist_.size());
  return true;
}

bool PlateRelayGate::select_event(const Detection& detection, int64_t frame_id, RelayEvent* event) {
  if (!ready_ || event == nullptr || detection.score < config_.min_detect_score || !detection.plate_recognized ||
      detection.plate_score < config_.min_plate_score) {
    return false;
  }

  const std::string plate = normalize_plate_text(detection.plate_text);
  if (plate.empty() || whitelist_.find(plate) == whitelist_.end()) {
    return false;
  }

  const auto now = std::chrono::steady_clock::now();
  if (is_in_cooldown(plate, now)) {
    return false;
  }

  event->plate_text = plate;
  event->plate_score = detection.plate_score;
  event->track_id = detection.track_id;
  event->frame_id = frame_id;
  if (config_.verbose) {
    std::fprintf(stderr, "[RELAY] accepted plate=%s score=%.3f track=%d\n", plate.c_str(), detection.plate_score, detection.track_id);
  }
  return true;
}

void PlateRelayGate::commit_event(const RelayEvent& event) {
  last_triggered_[event.plate_text] = std::chrono::steady_clock::now();
}

const std::string& PlateRelayGate::last_error() const {
  return last_error_;
}

bool PlateRelayGate::is_in_cooldown(const std::string& plate, std::chrono::steady_clock::time_point now) const {
  const auto it = last_triggered_.find(plate);
  if (it == last_triggered_.end()) {
    return false;
  }
  return now - it->second < std::chrono::seconds(std::max(0, config_.plate_cooldown_sec));
}

void relay_controller_thread(const RelayControllerConfig& config, StopFlag& stop, RelayEventQueue& input) {
  if (!config.enabled) {
    return;
  }
  if (config.slave_address < 1 || config.slave_address > 247 || config.coil_address < 0 || config.coil_address > 0xffff ||
      config.pulse_ms <= 0) {
    std::fprintf(stderr, "[RELAY] invalid relay configuration\n");
    input.set_accepting(false);
    return;
  }

  ModbusRelayPort port;
  if (!port.open_port(config)) {
    input.set_accepting(false);
    return;
  }

  input.set_accepting(false);
  if (!port.write_coil(config, false)) {
    std::fprintf(stderr, "[RELAY] startup close failed; relay control disabled\n");
    return;
  }
  input.set_accepting(true);

  while (!stop.stop_requested()) {
    RelayEvent event;
    if (!input.pop_for(&event, 100)) {
      if (input.is_closed()) {
        break;
      }
      continue;
    }

    input.set_accepting(false);
    if (config.verbose) {
      std::fprintf(stderr, "[RELAY] pulse start plate=%s track=%d\n", event.plate_text.c_str(), event.track_id);
    }
    const bool enabled = port.write_coil(config, true);
    if (enabled) {
      std::this_thread::sleep_for(std::chrono::milliseconds(config.pulse_ms));
    }
    const bool disabled = port.write_coil(config, false);
    if (config.verbose) {
      std::fprintf(stderr, "[RELAY] pulse finished result=%s\n", enabled && disabled ? "ok" : "failed");
    }
    if (enabled && disabled) {
      input.set_accepting(true);
    } else {
      std::fprintf(stderr, "[RELAY] pulse failed; relay control disabled\n");
      break;
    }
  }

  input.set_accepting(false);
  port.write_coil(config, false);
}

}  // namespace rkai