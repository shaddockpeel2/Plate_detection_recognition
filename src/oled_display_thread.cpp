#include "oled_display_thread.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <sstream>
#include <string>
#include <sys/ioctl.h>
#include <thread>
#include <unistd.h>
#include <vector>

extern "C" {
#include <ft2build.h>
#include FT_FREETYPE_H
}

namespace rkai {
namespace {

constexpr int kOledWidth = 128;
constexpr int kOledHeight = 64;
constexpr int kOledPages = kOledHeight / 8;

struct Glyph8x8 {
  char ch;
  uint8_t rows[8];
};

const Glyph8x8 kFont8x8Basic[] = {
    {' ', {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}},
    {'!', {0x18,0x3c,0x3c,0x18,0x18,0x00,0x18,0x00}},
    {'.', {0x00,0x00,0x00,0x00,0x00,0x18,0x18,0x00}},
    {'-', {0x00,0x00,0x00,0x7e,0x00,0x00,0x00,0x00}},
    {'_', {0x00,0x00,0x00,0x00,0x00,0x00,0x7e,0x00}},
    {'0', {0x3c,0x66,0x6e,0x76,0x66,0x66,0x3c,0x00}},
    {'1', {0x18,0x38,0x18,0x18,0x18,0x18,0x7e,0x00}},
    {'2', {0x3c,0x66,0x06,0x0c,0x18,0x30,0x7e,0x00}},
    {'3', {0x3c,0x66,0x06,0x1c,0x06,0x66,0x3c,0x00}},
    {'4', {0x0c,0x1c,0x3c,0x6c,0x7e,0x0c,0x0c,0x00}},
    {'5', {0x7e,0x60,0x7c,0x06,0x06,0x66,0x3c,0x00}},
    {'6', {0x1c,0x30,0x60,0x7c,0x66,0x66,0x3c,0x00}},
    {'7', {0x7e,0x66,0x06,0x0c,0x18,0x18,0x18,0x00}},
    {'8', {0x3c,0x66,0x66,0x3c,0x66,0x66,0x3c,0x00}},
    {'9', {0x3c,0x66,0x66,0x3e,0x06,0x0c,0x38,0x00}},
    {':', {0x00,0x18,0x18,0x00,0x00,0x18,0x18,0x00}},
    {'A', {0x18,0x3c,0x66,0x66,0x7e,0x66,0x66,0x00}},
    {'B', {0x7c,0x66,0x66,0x7c,0x66,0x66,0x7c,0x00}},
    {'C', {0x3c,0x66,0x60,0x60,0x60,0x66,0x3c,0x00}},
    {'D', {0x78,0x6c,0x66,0x66,0x66,0x6c,0x78,0x00}},
    {'E', {0x7e,0x60,0x60,0x7c,0x60,0x60,0x7e,0x00}},
    {'F', {0x7e,0x60,0x60,0x7c,0x60,0x60,0x60,0x00}},
    {'G', {0x3c,0x66,0x60,0x6e,0x66,0x66,0x3c,0x00}},
    {'H', {0x66,0x66,0x66,0x7e,0x66,0x66,0x66,0x00}},
    {'I', {0x3c,0x18,0x18,0x18,0x18,0x18,0x3c,0x00}},
    {'J', {0x1e,0x0c,0x0c,0x0c,0x0c,0x6c,0x38,0x00}},
    {'K', {0x66,0x6c,0x78,0x70,0x78,0x6c,0x66,0x00}},
    {'L', {0x60,0x60,0x60,0x60,0x60,0x60,0x7e,0x00}},
    {'M', {0x63,0x77,0x7f,0x6b,0x63,0x63,0x63,0x00}},
    {'N', {0x66,0x76,0x7e,0x7e,0x6e,0x66,0x66,0x00}},
    {'O', {0x3c,0x66,0x66,0x66,0x66,0x66,0x3c,0x00}},
    {'P', {0x7c,0x66,0x66,0x7c,0x60,0x60,0x60,0x00}},
    {'Q', {0x3c,0x66,0x66,0x66,0x6e,0x3c,0x0e,0x00}},
    {'R', {0x7c,0x66,0x66,0x7c,0x78,0x6c,0x66,0x00}},
    {'S', {0x3c,0x66,0x60,0x3c,0x06,0x66,0x3c,0x00}},
    {'T', {0x7e,0x5a,0x18,0x18,0x18,0x18,0x3c,0x00}},
    {'U', {0x66,0x66,0x66,0x66,0x66,0x66,0x3c,0x00}},
    {'V', {0x66,0x66,0x66,0x66,0x66,0x3c,0x18,0x00}},
    {'W', {0x63,0x63,0x63,0x6b,0x7f,0x77,0x63,0x00}},
    {'X', {0x66,0x66,0x3c,0x18,0x3c,0x66,0x66,0x00}},
    {'Y', {0x66,0x66,0x66,0x3c,0x18,0x18,0x3c,0x00}},
    {'Z', {0x7e,0x06,0x0c,0x18,0x30,0x60,0x7e,0x00}},
};

const uint8_t* glyph_rows(char ch) {
  for (const auto& glyph : kFont8x8Basic) {
    if (glyph.ch == ch) {
      return glyph.rows;
    }
  }
  return kFont8x8Basic[0].rows;
}

bool write_all(int fd, const uint8_t* data, std::size_t size) {
  while (size > 0) {
    const ssize_t written = write(fd, data, size);
    if (written < 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    data += written;
    size -= static_cast<std::size_t>(written);
  }
  return true;
}

bool oled_commands(int fd, const uint8_t* commands, std::size_t count) {
  uint8_t buffer[32];
  if (count + 1 > sizeof(buffer)) {
    return false;
  }
  buffer[0] = 0x00;
  std::memcpy(buffer + 1, commands, count);
  return write_all(fd, buffer, count + 1);
}

bool oled_data(int fd, const uint8_t* data, std::size_t count) {
  uint8_t buffer[17];
  while (count > 0) {
    const std::size_t chunk = std::min<std::size_t>(16, count);
    buffer[0] = 0x40;
    std::memcpy(buffer + 1, data, chunk);
    if (!write_all(fd, buffer, chunk + 1)) {
      return false;
    }
    data += chunk;
    count -= chunk;
  }
  return true;
}

bool oled_set_cursor(int fd, uint8_t page, uint8_t column) {
  const uint8_t commands[] = {
      static_cast<uint8_t>(0xb0 | (page & 0x07)),
      static_cast<uint8_t>(0x00 | (column & 0x0f)),
      static_cast<uint8_t>(0x10 | ((column >> 4) & 0x0f)),
  };
  return oled_commands(fd, commands, sizeof(commands));
}

bool oled_clear(int fd) {
  uint8_t zeros[kOledWidth] = {0};
  for (uint8_t page = 0; page < kOledPages; ++page) {
    if (!oled_set_cursor(fd, page, 0) || !oled_data(fd, zeros, sizeof(zeros))) {
      return false;
    }
  }
  return true;
}

struct OledFramebuffer {
  uint8_t pixels[kOledWidth * kOledHeight];

  void clear() {
    std::memset(pixels, 0, sizeof(pixels));
  }

  void set_pixel(int x, int y, bool on = true) {
    if (x < 0 || x >= kOledWidth || y < 0 || y >= kOledHeight) {
      return;
    }
    pixels[y * kOledWidth + x] = on ? 1 : 0;
  }
};

bool oled_flush_framebuffer(int fd, const OledFramebuffer& fb) {
  uint8_t page_data[kOledWidth];
  for (uint8_t page = 0; page < kOledPages; ++page) {
    std::memset(page_data, 0, sizeof(page_data));
    for (int x = 0; x < kOledWidth; ++x) {
      uint8_t value = 0;
      for (int bit = 0; bit < 8; ++bit) {
        const int y = page * 8 + bit;
        if (fb.pixels[y * kOledWidth + x]) {
          value |= static_cast<uint8_t>(1 << bit);
        }
      }
      page_data[x] = value;
    }
    if (!oled_set_cursor(fd, page, 0) || !oled_data(fd, page_data, sizeof(page_data))) {
      return false;
    }
  }
  return true;
}

bool oled_init(int fd) {
  const uint8_t init[] = {
      0xae, 0xd5, 0x80, 0xa8, 0x3f, 0xd3, 0x00, 0x40,
      0x8d, 0x14, 0x20, 0x02, 0xa1, 0xc8, 0xda, 0x12,
      0x81, 0xcf, 0xd9, 0xf1, 0xdb, 0x40, 0xa4, 0xa6, 0xaf,
  };
  return oled_commands(fd, init, sizeof(init)) && oled_clear(fd);
}

bool oled_draw_char(int fd, char ch) {
  if (ch < 32 || ch > 127) {
    ch = ' ';
  }

  const uint8_t* rows = glyph_rows(ch);
  uint8_t columns[8];
  for (int x = 0; x < 8; ++x) {
    uint8_t column = 0;
    for (int y = 0; y < 8; ++y) {
      if (rows[y] & static_cast<uint8_t>(1 << (7 - x))) {
        column |= static_cast<uint8_t>(1 << y);
      }
    }
    columns[x] = column;
  }
  return oled_data(fd, columns, sizeof(columns));
}

bool oled_print(int fd, uint8_t page, uint8_t column, const std::string& text) {
  if (!oled_set_cursor(fd, page, column)) {
    return false;
  }
  for (char ch : text) {
    if (column > kOledWidth - 8) {
      break;
    }
    if (!oled_draw_char(fd, ch)) {
      return false;
    }
    column += 8;
  }
  return true;
}

std::string ascii_visible_text(const std::string& text) {
  std::string out;
  out.reserve(text.size());
  for (unsigned char ch : text) {
    if (ch >= 32 && ch <= 126) {
      out.push_back(static_cast<char>(ch));
    }
  }
  return out.empty() ? std::string("UNKNOWN") : out;
}

std::string fixed_width(std::string text, std::size_t width) {
  if (text.size() > width) {
    text.resize(width);
  }
  if (text.size() < width) {
    text.append(width - text.size(), ' ');
  }
  return text;
}

class OledTextRenderer {
 public:
  OledTextRenderer() = default;
  OledTextRenderer(const OledTextRenderer&) = delete;
  OledTextRenderer& operator=(const OledTextRenderer&) = delete;

  ~OledTextRenderer() {
    release();
  }

  bool init(const char* font_path, int pixel_size) {
    release();
    if (FT_Init_FreeType(&library_) != 0) {
      return false;
    }
    if (FT_New_Face(library_, font_path, 0, &face_) != 0) {
      release();
      return false;
    }
    if (FT_Set_Pixel_Sizes(face_, 0, static_cast<FT_UInt>(pixel_size)) != 0) {
      release();
      return false;
    }
    ready_ = true;
    return true;
  }

  bool ready() const {
    return ready_;
  }

  void draw_text(OledFramebuffer* fb, int x, int baseline_y, const std::string& text) const {
    if (!ready_ || fb == nullptr) {
      return;
    }
    int pen_x = x;
    for (std::size_t i = 0; i < text.size();) {
      const uint32_t codepoint = next_codepoint(text, &i);
      if (codepoint == 0) {
        continue;
      }
      if (FT_Load_Char(face_, codepoint, FT_LOAD_RENDER) != 0) {
        pen_x += 8;
        continue;
      }
      const FT_GlyphSlot glyph = face_->glyph;
      const int glyph_x = pen_x + glyph->bitmap_left;
      const int glyph_y = baseline_y - glyph->bitmap_top;
      blend_bitmap(fb, glyph_x, glyph_y, glyph->bitmap);
      pen_x += static_cast<int>(glyph->advance.x >> 6);
      if (pen_x >= kOledWidth) {
        break;
      }
    }
  }

 private:
  static uint32_t next_codepoint(const std::string& text, std::size_t* index) {
    if (index == nullptr || *index >= text.size()) {
      return 0;
    }
    const unsigned char c0 = static_cast<unsigned char>(text[*index]);
    if (c0 < 0x80) {
      *index += 1;
      return c0;
    }
    if ((c0 & 0xe0) == 0xc0 && *index + 1 < text.size()) {
      const unsigned char c1 = static_cast<unsigned char>(text[*index + 1]);
      *index += 2;
      return ((c0 & 0x1f) << 6) | (c1 & 0x3f);
    }
    if ((c0 & 0xf0) == 0xe0 && *index + 2 < text.size()) {
      const unsigned char c1 = static_cast<unsigned char>(text[*index + 1]);
      const unsigned char c2 = static_cast<unsigned char>(text[*index + 2]);
      *index += 3;
      return ((c0 & 0x0f) << 12) | ((c1 & 0x3f) << 6) | (c2 & 0x3f);
    }
    if ((c0 & 0xf8) == 0xf0 && *index + 3 < text.size()) {
      const unsigned char c1 = static_cast<unsigned char>(text[*index + 1]);
      const unsigned char c2 = static_cast<unsigned char>(text[*index + 2]);
      const unsigned char c3 = static_cast<unsigned char>(text[*index + 3]);
      *index += 4;
      return ((c0 & 0x07) << 18) | ((c1 & 0x3f) << 12) | ((c2 & 0x3f) << 6) | (c3 & 0x3f);
    }
    *index += 1;
    return 0;
  }

  static void blend_bitmap(OledFramebuffer* fb, int x, int y, const FT_Bitmap& bitmap) {
    for (int row = 0; row < static_cast<int>(bitmap.rows); ++row) {
      for (int col = 0; col < static_cast<int>(bitmap.width); ++col) {
        const uint8_t alpha = bitmap.buffer[row * bitmap.pitch + col];
        if (alpha > 64) {
          fb->set_pixel(x + col, y + row);
        }
      }
    }
  }

  void release() {
    if (face_ != nullptr) {
      FT_Done_Face(face_);
      face_ = nullptr;
    }
    if (library_ != nullptr) {
      FT_Done_FreeType(library_);
      library_ = nullptr;
    }
    ready_ = false;
  }

  FT_Library library_ = nullptr;
  FT_Face face_ = nullptr;
  bool ready_ = false;
};

void draw_ascii_char(OledFramebuffer* fb, int x, int y, char ch) {
  if (fb == nullptr) {
    return;
  }
  const uint8_t* rows = glyph_rows(ch);
  for (int row = 0; row < 8; ++row) {
    for (int col = 0; col < 8; ++col) {
      if (rows[row] & static_cast<uint8_t>(1 << (7 - col))) {
        fb->set_pixel(x + col, y + row);
      }
    }
  }
}

void draw_ascii_text(OledFramebuffer* fb, int x, int y, const std::string& text) {
  int pen_x = x;
  for (char ch : text) {
    if (pen_x > kOledWidth - 8) {
      break;
    }
    draw_ascii_char(fb, pen_x, y, ch);
    pen_x += 8;
  }
}

bool draw_plate_screen_ascii(int fd, const OledPlateEvent& event) {
  OledFramebuffer fb;
  fb.clear();
  std::ostringstream score;
  score.setf(std::ios::fixed);
  score.precision(2);
  score << "SCORE " << event.plate_score;
  std::ostringstream meta;
  meta << "T" << event.track_id << " F" << event.frame_id;
  draw_ascii_text(&fb, 0, 0, "PLATE");
  draw_ascii_text(&fb, 0, 16, fixed_width(ascii_visible_text(event.plate_text), 16));
  draw_ascii_text(&fb, 0, 32, fixed_width(score.str(), 16));
  draw_ascii_text(&fb, 0, 48, fixed_width(meta.str(), 16));
  return oled_flush_framebuffer(fd, fb);
}

bool draw_plate_screen_utf8(int fd, const OledTextRenderer& renderer, const OledPlateEvent& event) {
  OledFramebuffer fb;
  fb.clear();
  std::ostringstream score;
  score.setf(std::ios::fixed);
  score.precision(2);
  score << "SCORE " << event.plate_score;
  std::ostringstream meta;
  meta << "T" << event.track_id << " F" << event.frame_id;
  renderer.draw_text(&fb, 0, 12, "PLATE");
  renderer.draw_text(&fb, 0, 34, event.plate_text);
  renderer.draw_text(&fb, 0, 50, score.str());
  renderer.draw_text(&fb, 0, 63, meta.str());
  return oled_flush_framebuffer(fd, fb);
}

bool draw_plate_screen(int fd, const OledTextRenderer& renderer, const OledPlateEvent& event) {
  if (renderer.ready()) {
    return draw_plate_screen_utf8(fd, renderer, event);
  }
  return draw_plate_screen_ascii(fd, event);
}

int open_oled(const OledDisplayConfig& config) {
  const int fd = open(config.i2c_device.c_str(), O_RDWR);
  if (fd < 0) {
    std::fprintf(stderr, "[OLED] open %s failed: %s\n", config.i2c_device.c_str(), std::strerror(errno));
    return -1;
  }
  if (ioctl(fd, I2C_SLAVE, config.i2c_address) < 0) {
    std::fprintf(stderr, "[OLED] select i2c address 0x%02x failed: %s\n", config.i2c_address, std::strerror(errno));
    close(fd);
    return -1;
  }
  if (!oled_init(fd)) {
    std::fprintf(stderr, "[OLED] init failed: %s\n", std::strerror(errno));
    close(fd);
    return -1;
  }
  oled_print(fd, 0, 0, "PLATE OLED READY");
  return fd;
}

}  // namespace

OledPlateQueue::OledPlateQueue(std::size_t capacity) : capacity_(std::max<std::size_t>(1, capacity)) {}

bool OledPlateQueue::try_push(OledPlateEvent event) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (closed_) {
    return false;
  }
  while (queue_.size() >= capacity_) {
    queue_.pop_front();
    ++stats_.dropped;
  }
  queue_.push_back(std::move(event));
  ++stats_.pushed;
  not_empty_.notify_one();
  return true;
}

bool OledPlateQueue::pop(OledPlateEvent* event) {
  if (event == nullptr) {
    return false;
  }
  std::unique_lock<std::mutex> lock(mutex_);
  not_empty_.wait(lock, [&] { return closed_ || !queue_.empty(); });
  if (queue_.empty()) {
    return false;
  }
  *event = std::move(queue_.back());
  queue_.clear();
  return true;
}

void OledPlateQueue::close() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
  }
  not_empty_.notify_all();
}

OledPlateQueueStats OledPlateQueue::stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return stats_;
}

void oled_display_thread(const OledDisplayConfig& config, StopFlag& stop, OledPlateQueue& input) {
  if (!config.enabled) {
    return;
  }

  const int fd = open_oled(config);
  if (fd < 0) {
    return;
  }

  OledTextRenderer text_renderer;
  if (!text_renderer.init("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 14)) {
    std::fprintf(stderr, "[OLED] UTF-8 font unavailable, fallback to ASCII plate text\n");
  }

  std::string last_text;
  int last_track_id = -1;
  auto last_refresh = std::chrono::steady_clock::time_point::min();
  OledPlateEvent event;
  int64_t displayed = 0;

  while (!stop.stop_requested() && input.pop(&event)) {
    if (event.plate_text.empty()) {
      continue;
    }
    const auto now = std::chrono::steady_clock::now();
    if (event.plate_text == last_text && event.track_id == last_track_id &&
        now - last_refresh < std::chrono::milliseconds(config.min_refresh_interval_ms)) {
      continue;
    }

    if (!draw_plate_screen(fd, text_renderer, event)) {
      std::fprintf(stderr, "[OLED] draw failed: %s\n", std::strerror(errno));
      break;
    }
    last_text = event.plate_text;
    last_track_id = event.track_id;
    last_refresh = now;
    ++displayed;

    if (config.verbose) {
      std::fprintf(stderr,
                   "[OLED] display frame=%ld track=%d plate=%s score=%.4f\n",
                   event.frame_id,
                   event.track_id,
                   event.plate_text.c_str(),
                   event.plate_score);
    }
  }

  oled_clear(fd);
  close(fd);
  const auto stats = input.stats();
  std::fprintf(stderr,
               "[PERF] oled_display displayed=%ld pushed=%ld dropped=%ld\n",
               displayed,
               stats.pushed,
               stats.dropped);
}

}  // namespace rkai