#pragma once

#include "decoder_thread.hpp"

#include <cstddef>
#include <string>

namespace rkai {

enum class CameraPixelFormat {
  Auto,
  YUYV,
};

struct CameraSourceConfig {
  std::string device = "/dev/video0";
  int width = 640;
  int height = 480;
  int fps_num = 25;
  int fps_den = 1;
  int frame_limit = 300;
  CameraPixelFormat format = CameraPixelFormat::Auto;
  std::size_t output_queue_capacity = 8;
  bool verbose = false;
};

void camera_source_thread(const CameraSourceConfig& config, StopFlag& stop, DecodedFrameQueue& output);

}  // namespace rkai