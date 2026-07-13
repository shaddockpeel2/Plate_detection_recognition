#pragma once

#include "bytetrack.hpp"
#include "decoder_thread.hpp"
#include "oled_display_thread.hpp"
#include "pipeline_types.hpp"
#include "plate_ocr_stage.hpp"
#include "plate_relay.hpp"
#include "upload_event.hpp"

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
  PlateOcrStageConfig plate_ocr;
  PlateRelayGateConfig plate_relay;
  ByteTrackConfig tracker;
  UploadEventConfig upload_event;
  bool oled_display_enabled = false;
  bool verbose = false;
  std::size_t output_queue_capacity = 4;
};

void postprocess_osd_thread(const PostprocessOsdConfig& config,
                            StopFlag& stop,
                            InferenceResultQueue& input,
                            OsdFrameQueue& output,
                            UploadEventQueue* upload_events = nullptr,
                            OledPlateQueue* oled_events = nullptr,
                            RelayEventQueue* relay_events = nullptr);

}  // namespace rkai