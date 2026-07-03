#pragma once

#include "decoder_thread.hpp"
#include "pipeline_types.hpp"
#include "rknn_input_runtime.hpp"

#include <cstddef>

namespace rkai {

struct PreprocessConfig {
  bool letterbox = true;
  bool model_rgb = true;
  bool verbose = false;
  std::size_t output_queue_capacity = 4;
};

void preprocess_thread(const PreprocessConfig& config,
                       StopFlag& stop,
                       DecodedFrameQueue& input,
                       PreprocessedFrameQueue& output,
                       RknnInputRuntime& rknn_inputs);

}  // namespace rkai