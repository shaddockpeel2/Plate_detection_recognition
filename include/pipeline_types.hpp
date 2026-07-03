#pragma once

#include "decoder_thread.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

extern "C" {
#include <rknn_api.h>
}

namespace rkai {

struct LetterBoxInfo {
  float scale = 1.0f;
  int resized_width = 0;
  int resized_height = 0;
  int x_pad = 0;
  int y_pad = 0;
};

struct RknnInputTensor {
  rknn_tensor_mem* mem = nullptr;
  int dma_fd = -1;
  void* virt_addr = nullptr;
  uint32_t size = 0;
  int width = 0;
  int height = 0;
  int channels = 0;
};

struct PreprocessedFrame {
  DecodedFramePtr source;
  RknnInputTensor input;
  LetterBoxInfo letterbox;
};

using PreprocessedFramePtr = std::shared_ptr<PreprocessedFrame>;
using PreprocessedFrameQueue = BlockingQueue<PreprocessedFramePtr>;

struct InferenceOutputTensor {
  uint32_t index = 0;
  rknn_tensor_attr attr{};
  std::vector<uint8_t> data;
  bool want_float = false;
};

struct InferenceResult {
  PreprocessedFramePtr frame;
  std::vector<InferenceOutputTensor> outputs;
  double run_ms = 0.0;
  double output_get_ms = 0.0;
  double total_ms = 0.0;
};

using InferenceResultPtr = std::shared_ptr<InferenceResult>;
using InferenceResultQueue = BlockingQueue<InferenceResultPtr>;

struct Detection {
  int class_id = 0;
  int track_id = -1;
  float score = 0.0f;
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  std::string plate_text;
  float plate_score = 0.0f;
  bool plate_recognized = false;
};

struct OsdFrame {
  InferenceResultPtr inference;
  std::vector<Detection> detections;
};

using OsdFramePtr = std::shared_ptr<OsdFrame>;
using OsdFrameQueue = BlockingQueue<OsdFramePtr>;

}  // namespace rkai