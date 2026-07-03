#include "plate_ocr_stage.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <opencv2/core.hpp>

#include "RgaUtils.h"
#include "im2d.h"
#include "rga.h"

namespace rkai {
namespace {

int clamp_int(float value, int lo, int hi) {
  return std::max(lo, std::min(hi, static_cast<int>(value + 0.5f)));
}

int align_even_floor(int value) {
  return value & ~1;
}

int align_even_ceil(int value) {
  return (value + 1) & ~1;
}

void expand_box_for_ocr(const DecodedFrame& frame, int* x1, int* y1, int* x2, int* y2) {
  const int w = std::max(1, *x2 - *x1);
  const int h = std::max(1, *y2 - *y1);
  const int pad_x = std::max(2, w / 12);
  const int pad_y = std::max(2, h / 5);
  *x1 = std::max(0, *x1 - pad_x);
  *y1 = std::max(0, *y1 - pad_y);
  *x2 = std::min(frame.width, *x2 + pad_x);
  *y2 = std::min(frame.height, *y2 + pad_y);
}

im_rect compute_plate_dst_rect(int src_w, int src_h, int dst_w, int dst_h) {
  const float ratio = static_cast<float>(src_w) / static_cast<float>(std::max(1, src_h));
  const int scaled_w = std::max(2, std::min(dst_w, align_even_floor(static_cast<int>(std::ceil(dst_h * ratio)))));
  const int y = (dst_h - dst_h) / 2;
  return im_rect{0, y, scaled_w, dst_h};
}

bool valid_frame_for_rga(const DecodedFrame& frame) {
  return frame.dma_fd >= 0 && frame.width > 0 && frame.height > 0 && frame.hor_stride > 0 && frame.ver_stride > 0;
}

}  // namespace

bool PlateOcrStage::init(const PlateOcrStageConfig& config) {
  release();
  config_ = config;
  if (!config_.enabled) {
    return true;
  }
  if (config_.crop_width <= 0 || config_.crop_height <= 0) {
    std::fprintf(stderr, "[OCR] invalid crop size: %dx%d\n", config_.crop_width, config_.crop_height);
    return false;
  }
  ready_ = recognizer_.init(config_.recognizer);
  if (!ready_) {
    std::fprintf(stderr, "[OCR] recognizer init failed, OCR stage disabled\n");
    return false;
  }
  input_info_ = recognizer_.input_info();
  if (input_info_.supports_direct_uint8_mem) {
    direct_input_ = recognizer_.create_input_mem();
    direct_input_ready_ = direct_input_.mem != nullptr && direct_input_.mem->fd >= 0;
    if (config_.verbose) {
      std::fprintf(stderr,
                   "[OCR] direct input mem %s type=%d fd=%d input=%dx%dx%d\n",
                   direct_input_ready_ ? "enabled" : "unavailable",
                   input_info_.type,
                   direct_input_.mem ? direct_input_.mem->fd : -1,
                   input_info_.width,
                   input_info_.height,
                   input_info_.channels);
    }
  } else if (config_.verbose) {
    std::fprintf(stderr,
                 "[OCR] direct input mem disabled for float model type=%d; keep RGA crop + Mat normalize path\n",
                 input_info_.type);
  }
  return true;
}

bool PlateOcrStage::ready() const {
  return config_.enabled && ready_ && recognizer_.ready();
}

void PlateOcrStage::release() {
  recognizer_.destroy_input_mem(&direct_input_);
  direct_input_ready_ = false;
  input_info_ = LicensePlateOcrInputInfo();
  recognizer_.release();
  track_cache_.clear();
  ready_ = false;
}

PlateOcrStageStats PlateOcrStage::process(const DecodedFrame& frame, std::vector<Detection>* detections) {
  PlateOcrStageStats stats;
  if (!ready() || detections == nullptr || detections->empty()) {
    return stats;
  }
  if (!valid_frame_for_rga(frame)) {
    if (config_.verbose) {
      std::fprintf(stderr, "[OCR] skip frame=%ld invalid rga source\n", frame.frame_id);
    }
    return stats;
  }

  std::vector<uint8_t> crop(static_cast<std::size_t>(config_.crop_width) * config_.crop_height * 3);
  int processed_targets = 0;
  for (auto& detection : *detections) {
    if (detection.track_id < 0) {
      stats.skipped += 1;
      continue;
    }
    if (apply_cache(frame, &detection, &stats)) {
      continue;
    }
    if (detection.track_id >= 0) {
      const auto cache_it = track_cache_.find(detection.track_id);
      if (cache_it != track_cache_.end() && frame.frame_id < cache_it->second.next_retry_frame_id) {
        stats.skipped += 1;
        continue;
      }
    }

    if (processed_targets >= config_.max_targets_per_frame) {
      stats.skipped += 1;
      continue;
    }

    const int box_w = clamp_int(detection.x2 - detection.x1, 0, frame.width);
    const int box_h = clamp_int(detection.y2 - detection.y1, 0, frame.height);
    if (box_w < config_.min_box_width || box_h < config_.min_box_height) {
      stats.skipped += 1;
      continue;
    }

    double rga_ms = 0.0;
    LicensePlateOcrResult result;
    if (direct_input_ready_) {
      if (!crop_to_direct_input(frame, detection, &rga_ms)) {
        stats.skipped += 1;
        continue;
      }
      stats.rga_ms += rga_ms;
      stats.attempted += 1;
      processed_targets += 1;

      const auto ocr_start = std::chrono::steady_clock::now();
      result = recognizer_.recognize_preprocessed_mem(&direct_input_);
      const auto ocr_end = std::chrono::steady_clock::now();
      stats.ocr_ms += std::chrono::duration<double, std::milli>(ocr_end - ocr_start).count();
    } else {
      int actual_crop_width = 0;
      int actual_crop_height = 0;
      int actual_crop_stride = 0;
      if (!crop_with_rga(frame, detection, &crop, &actual_crop_width, &actual_crop_height, &actual_crop_stride, &rga_ms)) {
        stats.skipped += 1;
        continue;
      }
      stats.rga_ms += rga_ms;
      stats.attempted += 1;
      processed_targets += 1;

      cv::Mat bgr_crop(actual_crop_height, actual_crop_width, CV_8UC3, crop.data(), static_cast<std::size_t>(actual_crop_stride) * 3);
      const auto ocr_start = std::chrono::steady_clock::now();
      result = recognizer_.recognize(bgr_crop);
      const auto ocr_end = std::chrono::steady_clock::now();
      stats.ocr_ms += std::chrono::duration<double, std::milli>(ocr_end - ocr_start).count();
    }

    if (!result.ok || result.text.empty()) {
      update_retry(frame, detection);
      if (config_.verbose) {
        std::fprintf(stderr,
                     "[OCR] frame=%ld track=%d failed: %s\n",
                     frame.frame_id,
                     detection.track_id,
                     result.error.c_str());
      }
      continue;
    }

    std::fprintf(stderr,
                 "[OCR_RAW] frame=%ld track=%d text=%s score=%.4f\n",
                 frame.frame_id,
                 detection.track_id,
                 result.text.c_str(),
                 result.score);

    if (result.score < config_.min_ocr_score) {
      update_retry(frame, detection);
      continue;
    }

    detection.plate_text = result.text;
    detection.plate_score = result.score;
    detection.plate_recognized = true;
    update_cache(frame, detection, result);
    stats.recognized += 1;

    if (config_.verbose) {
      std::fprintf(stderr,
                   "[OCR] frame=%ld track=%d text=%s score=%.4f rga_ms=%.3f\n",
                   frame.frame_id,
                   detection.track_id,
                   detection.plate_text.c_str(),
                   detection.plate_score,
                   rga_ms);
    }
  }
  return stats;
}

bool PlateOcrStage::apply_cache(const DecodedFrame& frame, Detection* detection, PlateOcrStageStats* stats) {
  if (detection == nullptr || stats == nullptr || detection->track_id < 0) {
    return false;
  }
  auto it = track_cache_.find(detection->track_id);
  if (it == track_cache_.end()) {
    return false;
  }

  TrackOcrCache& cache = it->second;
  if (cache.text.empty() || cache.score < config_.min_ocr_score) {
    return false;
  }
  if (frame.frame_id - cache.last_frame_id > config_.max_cache_age_frames) {
    track_cache_.erase(it);
    return false;
  }
  if (frame.frame_id >= cache.next_retry_frame_id) {
    return false;
  }

  detection->plate_text = cache.text;
  detection->plate_score = cache.score;
  detection->plate_recognized = true;
  cache.last_frame_id = frame.frame_id;
  cache.stable_hits += 1;
  stats->cache_hits += 1;
  return true;
}

void PlateOcrStage::update_cache(const DecodedFrame& frame,
                                 const Detection& detection,
                                 const LicensePlateOcrResult& result) {
  if (detection.track_id < 0 || !result.ok || result.text.empty()) {
    return;
  }

  TrackOcrCache& cache = track_cache_[detection.track_id];
  if (result.score >= cache.score || cache.text.empty()) {
    cache.text = result.text;
    cache.score = result.score;
  }
  cache.last_frame_id = frame.frame_id;
  cache.next_retry_frame_id = frame.frame_id + std::max(1, config_.ocr_interval_frames);
  cache.stable_hits += 1;
}

void PlateOcrStage::update_retry(const DecodedFrame& frame, const Detection& detection) {
  if (detection.track_id < 0) {
    return;
  }
  TrackOcrCache& cache = track_cache_[detection.track_id];
  cache.last_frame_id = frame.frame_id;
  cache.next_retry_frame_id = frame.frame_id + std::max(1, config_.ocr_interval_frames / 2);
}

bool PlateOcrStage::crop_to_direct_input(const DecodedFrame& frame,
                                         const Detection& detection,
                                         double* rga_ms) const {
  if (!direct_input_ready_ || direct_input_.mem == nullptr || direct_input_.mem->fd < 0) {
    return false;
  }

  int x1 = clamp_int(detection.x1, 0, frame.width - 1);
  int y1 = clamp_int(detection.y1, 0, frame.height - 1);
  int x2 = clamp_int(detection.x2, 0, frame.width - 1);
  int y2 = clamp_int(detection.y2, 0, frame.height - 1);
  x1 = align_even_floor(x1);
  y1 = align_even_floor(y1);
  x2 = std::min(frame.width, align_even_ceil(x2));
  y2 = std::min(frame.height, align_even_ceil(y2));
  expand_box_for_ocr(frame, &x1, &y1, &x2, &y2);
  x1 = align_even_floor(x1);
  y1 = align_even_floor(y1);
  x2 = std::min(frame.width, align_even_ceil(x2));
  y2 = std::min(frame.height, align_even_ceil(y2));

  const int crop_w = x2 - x1;
  const int crop_h = y2 - y1;
  if (crop_w < config_.min_box_width || crop_h < config_.min_box_height) {
    return false;
  }

  const int dst_format = input_info_.channels == 3 ? RK_FORMAT_BGR_888 : RK_FORMAT_YCbCr_420_SP;
  if (direct_input_.mem->virt_addr != nullptr && direct_input_.mem->size > 0) {
    std::memset(direct_input_.mem->virt_addr, 0, direct_input_.mem->size);
  }
  rga_buffer_t src = wrapbuffer_fd(frame.dma_fd,
                                   frame.hor_stride,
                                   frame.ver_stride,
                                   RK_FORMAT_YCbCr_420_SP,
                                   frame.hor_stride,
                                   frame.ver_stride);
  rga_buffer_t dst = wrapbuffer_fd(direct_input_.mem->fd,
                                   input_info_.width,
                                   input_info_.height,
                                   dst_format,
                                   input_info_.width,
                                   input_info_.height);

  im_rect src_rect{x1, y1, crop_w, crop_h};
  im_rect dst_rect = compute_plate_dst_rect(crop_w, crop_h, input_info_.width, input_info_.height);

  const auto start = std::chrono::steady_clock::now();
  const IM_STATUS status = improcess(src, dst, {}, src_rect, dst_rect, {}, IM_SYNC);
  const auto end = std::chrono::steady_clock::now();
  if (rga_ms != nullptr) {
    *rga_ms = std::chrono::duration<double, std::milli>(end - start).count();
  }

  if (status != IM_STATUS_SUCCESS) {
    if (config_.verbose) {
      std::fprintf(stderr,
                   "[OCR] RGA direct crop failed frame=%ld status=%d message=%s src=%d,%d,%d,%d dst=%d,%d,%d,%d fd=%d\n",
                   frame.frame_id,
                   status,
                   imStrError(status),
                   x1,
                   y1,
                   crop_w,
                   crop_h,
                   dst_rect.x,
                   dst_rect.y,
                   dst_rect.width,
                   dst_rect.height,
                   direct_input_.mem->fd);
    }
    return false;
  }
  return true;
}

bool PlateOcrStage::crop_with_rga(const DecodedFrame& frame,
                                  const Detection& detection,
                                  std::vector<uint8_t>* bgr_crop,
                                  int* crop_width_out,
                                  int* crop_height_out,
                                  int* crop_stride_out,
                                  double* rga_ms) const {
  if (bgr_crop == nullptr || crop_width_out == nullptr || crop_height_out == nullptr || crop_stride_out == nullptr) {
    return false;
  }

  int x1 = clamp_int(detection.x1, 0, frame.width - 1);
  int y1 = clamp_int(detection.y1, 0, frame.height - 1);
  int x2 = clamp_int(detection.x2, 0, frame.width - 1);
  int y2 = clamp_int(detection.y2, 0, frame.height - 1);
  x1 = align_even_floor(x1);
  y1 = align_even_floor(y1);
  x2 = std::min(frame.width, align_even_ceil(x2));
  y2 = std::min(frame.height, align_even_ceil(y2));
  expand_box_for_ocr(frame, &x1, &y1, &x2, &y2);
  x1 = align_even_floor(x1);
  y1 = align_even_floor(y1);
  x2 = std::min(frame.width, align_even_ceil(x2));
  y2 = std::min(frame.height, align_even_ceil(y2));

  const int crop_w = x2 - x1;
  const int crop_h = y2 - y1;
  if (crop_w < config_.min_box_width || crop_h < config_.min_box_height) {
    return false;
  }

  const int dst_w = config_.crop_width;
  const int dst_h = config_.crop_height;
  const std::size_t dst_size = static_cast<std::size_t>(dst_w) * dst_h * 3;
  if (bgr_crop->size() < dst_size) {
    bgr_crop->resize(dst_size);
  }
  std::fill(bgr_crop->begin(), bgr_crop->begin() + static_cast<std::ptrdiff_t>(dst_size), 0);
  const im_rect dst_rect = compute_plate_dst_rect(crop_w, crop_h, dst_w, dst_h);
  *crop_width_out = dst_rect.width;
  *crop_height_out = dst_rect.height;
  *crop_stride_out = dst_w;

  rga_buffer_t src = wrapbuffer_fd(frame.dma_fd,
                                   frame.hor_stride,
                                   frame.ver_stride,
                                   RK_FORMAT_YCbCr_420_SP,
                                   frame.hor_stride,
                                   frame.ver_stride);
  rga_buffer_t dst = wrapbuffer_virtualaddr(bgr_crop->data(),
                                            dst_w,
                                            dst_h,
                                            RK_FORMAT_BGR_888,
                                            dst_w,
                                            dst_h);

  im_rect src_rect{x1, y1, crop_w, crop_h};

  const auto start = std::chrono::steady_clock::now();
  const IM_STATUS status = improcess(src, dst, {}, src_rect, dst_rect, {}, IM_SYNC);
  const auto end = std::chrono::steady_clock::now();
  if (rga_ms != nullptr) {
    *rga_ms = std::chrono::duration<double, std::milli>(end - start).count();
  }

  if (status != IM_STATUS_SUCCESS) {
    if (config_.verbose) {
      std::fprintf(stderr,
                   "[OCR] RGA crop failed frame=%ld status=%d message=%s src=%d,%d,%d,%d dst=%d,%d,%d,%d\n",
                   frame.frame_id,
                   status,
                   imStrError(status),
                   x1,
                   y1,
                   crop_w,
                   crop_h,
                   dst_rect.x,
                   dst_rect.y,
                   dst_rect.width,
                   dst_rect.height);
    }
    return false;
  }
  return true;
}

}  // namespace rkai