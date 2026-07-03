#pragma once

#include "decoder_thread.hpp"
#include "pipeline_types.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

extern "C" {
#include <rockchip/rk_type.h>
}

namespace rkai {

enum class EncodedVideoCodec {
  H264,
  H265,
};

struct EncoderWriterConfig {
  std::string output_path;
  EncodedVideoCodec codec = EncodedVideoCodec::H264;
  int fps_num = 30;
  int fps_den = 1;
  int bitrate = 4 * 1000 * 1000;
  int gop = 60;
  std::size_t output_queue_capacity = 8;
  bool verbose = false;
};

struct EncodedPacketView {
  const uint8_t* data = nullptr;
  int size = 0;
  int64_t frame_index = 0;
  bool keyframe = false;
};

class IEncodedSink {
 public:
  virtual ~IEncodedSink() = default;
  virtual bool open(const EncoderWriterConfig& config,
                    int width,
                    int height,
                    const uint8_t* header,
                    int header_size) = 0;
  virtual bool write(const EncodedPacketView& packet) = 0;
  virtual void close() = 0;
};

void encoder_writer_thread(const EncoderWriterConfig& config, StopFlag& stop, OsdFrameQueue& input);

}  // namespace rkai