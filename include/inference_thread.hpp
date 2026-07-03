#pragma once

#include "decoder_thread.hpp"
#include "pipeline_types.hpp"
#include "rknn_input_runtime.hpp"

#include <cstddef>

namespace rkai {

struct InferenceConfig {
  bool want_float_output = false;
  bool verbose = false;
  std::size_t output_queue_capacity = 4;
};

void inference_thread(const InferenceConfig& config,
                      StopFlag& stop,
                      PreprocessedFrameQueue& input,
                      InferenceResultQueue& output,
                      RknnInputRuntime& runtime);

}  // namespace rkai