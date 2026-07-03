#pragma once

#include "decoder_thread.hpp"
#include "license_plate_ocr.hpp"
#include "pipeline_types.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace rkai {

struct PlateOcrStageConfig {
  bool enabled = false;
  LicensePlateOcrConfig recognizer;
  int crop_width = 320;
  int crop_height = 48;
  int min_box_width = 16;
  int min_box_height = 8;
  int max_targets_per_frame = 4;
  int ocr_interval_frames = 15;
  int max_cache_age_frames = 90;
  float min_ocr_score = 0.80f;
  bool verbose = false;
};

struct PlateOcrStageStats {
  int attempted = 0;
  int recognized = 0;
  int cache_hits = 0;
  int skipped = 0;
  double rga_ms = 0.0;
  double ocr_ms = 0.0;
};

class PlateOcrStage {
 public:
  PlateOcrStage() = default;
  PlateOcrStage(const PlateOcrStage&) = delete;
  PlateOcrStage& operator=(const PlateOcrStage&) = delete;

  bool init(const PlateOcrStageConfig& config);
  bool ready() const;
  void release();

  PlateOcrStageStats process(const DecodedFrame& frame, std::vector<Detection>* detections);

 private:
  struct TrackOcrCache {
    std::string text;
    float score = 0.0f;
    int64_t last_frame_id = -1;
    int64_t next_retry_frame_id = 0;
    int stable_hits = 0;
  };

  bool crop_with_rga(const DecodedFrame& frame,
                     const Detection& detection,
                     std::vector<uint8_t>* bgr_crop,
                     int* crop_width,
                     int* crop_height,
                     int* crop_stride,
                     double* rga_ms) const;
  bool crop_to_direct_input(const DecodedFrame& frame, const Detection& detection, double* rga_ms) const;
  bool apply_cache(const DecodedFrame& frame, Detection* detection, PlateOcrStageStats* stats);
  void update_cache(const DecodedFrame& frame, const Detection& detection, const LicensePlateOcrResult& result);
  void update_retry(const DecodedFrame& frame, const Detection& detection);

  PlateOcrStageConfig config_;
  LicensePlateOcrRecognizer recognizer_;
  LicensePlateOcrInputInfo input_info_;
  LicensePlateOcrInputMem direct_input_;
  bool direct_input_ready_ = false;
  std::unordered_map<int, TrackOcrCache> track_cache_;
  bool ready_ = false;
};

}  // namespace rkai