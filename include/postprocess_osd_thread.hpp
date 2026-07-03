#pragma once

#include "decoder_thread.hpp"
#include "pipeline_types.hpp"

#include <cstddef>

namespace rkai {

enum class PostprocessModel {
  Auto,
  Yolo26,
  YoloV8Dfl,
};

struct PostprocessOsdConfig {
  PostprocessModel model = PostprocessModel::Auto;
  float box_threshold = 0.25f;
  float nms_threshold = 0.45f;
  int max_detections = 128;
  int class_count = 1;
  int dfl_len = 16;
  int model_width = 640;
  int model_height = 640;
  int line_thickness = 3;
  bool draw_osd = true;
  bool verbose = false;
  std::size_t output_queue_capacity = 4;
};

void postprocess_osd_thread(const PostprocessOsdConfig& config,
                            StopFlag& stop,
                            InferenceResultQueue& input,
                            OsdFrameQueue& output);

}  // namespace rkai