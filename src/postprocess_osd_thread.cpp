#include "postprocess_osd_thread.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

extern "C" {
#include <rockchip/mpp_buffer.h>
}

namespace rkai {
namespace {

int clamp_int(float value, int lo, int hi) {
  return std::max(lo, std::min(hi, static_cast<int>(value + 0.5f)));
}

float dequant_i8(int8_t value, int32_t zp, float scale) {
  return (static_cast<float>(value) - static_cast<float>(zp)) * scale;
}

float dequant_u8(uint8_t value, int32_t zp, float scale) {
  return (static_cast<float>(value) - static_cast<float>(zp)) * scale;
}

float iou(const Detection& a, const Detection& b) {
  const float xx1 = std::max(a.x1, b.x1);
  const float yy1 = std::max(a.y1, b.y1);
  const float xx2 = std::min(a.x2, b.x2);
  const float yy2 = std::min(a.y2, b.y2);
  const float w = std::max(0.0f, xx2 - xx1);
  const float h = std::max(0.0f, yy2 - yy1);
  const float inter = w * h;
  const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
  const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
  return inter / (area_a + area_b - inter + 1e-6f);
}

std::vector<Detection> nms(std::vector<Detection> detections, float threshold, int max_detections) {
  std::sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b) {
    return a.score > b.score;
  });

  std::vector<Detection> kept;
  std::vector<bool> removed(detections.size(), false);
  for (size_t i = 0; i < detections.size(); ++i) {
    if (removed[i]) {
      continue;
    }
    kept.push_back(detections[i]);
    if (static_cast<int>(kept.size()) >= max_detections) {
      break;
    }
    for (size_t j = i + 1; j < detections.size(); ++j) {
      if (!removed[j] && detections[i].class_id == detections[j].class_id && iou(detections[i], detections[j]) > threshold) {
        removed[j] = true;
      }
    }
  }
  return kept;
}

struct OutputLayout {
  int channels = 0;
  int grid_h = 0;
  int grid_w = 0;
  bool nchw = true;
};

OutputLayout parse_layout(const InferenceOutputTensor& tensor) {
  OutputLayout layout;
  const auto& attr = tensor.attr;
  if (attr.n_dims != 4) {
    return layout;
  }

  const int d1 = static_cast<int>(attr.dims[1]);
  const int d2 = static_cast<int>(attr.dims[2]);
  const int d3 = static_cast<int>(attr.dims[3]);
  if (attr.fmt == RKNN_TENSOR_NCHW) {
    layout.nchw = true;
    layout.channels = d1;
    layout.grid_h = d2;
    layout.grid_w = d3;
  } else if (attr.fmt == RKNN_TENSOR_NHWC) {
    layout.nchw = false;
    layout.grid_h = d1;
    layout.grid_w = d2;
    layout.channels = d3;
  } else if (d1 == d2 && d3 >= 5) {
    layout.nchw = false;
    layout.grid_h = d1;
    layout.grid_w = d2;
    layout.channels = d3;
  } else if (d1 >= 5 && d1 <= 512) {
    layout.nchw = true;
    layout.channels = d1;
    layout.grid_h = d2;
    layout.grid_w = d3;
  }
  return layout;
}

float read_tensor_value(const InferenceOutputTensor& tensor, const OutputLayout& layout, int channel, int h, int w) {
  size_t offset = 0;
  if (layout.nchw) {
    offset = static_cast<size_t>(channel) * layout.grid_h * layout.grid_w + h * layout.grid_w + w;
  } else {
    offset = static_cast<size_t>(h * layout.grid_w + w) * layout.channels + channel;
  }

  if (tensor.want_float) {
    const size_t byte_offset = offset * sizeof(float);
    if (byte_offset + sizeof(float) > tensor.data.size()) {
      return 0.0f;
    }
    float value = 0.0f;
    std::memcpy(&value, tensor.data.data() + byte_offset, sizeof(float));
    return value;
  }

  if (tensor.attr.type == RKNN_TENSOR_INT8) {
    if (offset >= tensor.data.size()) {
      return 0.0f;
    }
    const auto* data = reinterpret_cast<const int8_t*>(tensor.data.data());
    return dequant_i8(data[offset], tensor.attr.zp, tensor.attr.scale);
  }

  if (tensor.attr.type == RKNN_TENSOR_UINT8) {
    if (offset >= tensor.data.size()) {
      return 0.0f;
    }
    return dequant_u8(tensor.data[offset], tensor.attr.zp, tensor.attr.scale);
  }

  const size_t byte_offset = offset * sizeof(float);
  if (byte_offset + sizeof(float) > tensor.data.size()) {
    return 0.0f;
  }
  float value = 0.0f;
  std::memcpy(&value, tensor.data.data() + byte_offset, sizeof(float));
  return value;
}

float read_yolov8_score(const InferenceOutputTensor& tensor, const OutputLayout& layout, int channel, int h, int w) {
  return read_tensor_value(tensor, layout, channel, h, w);
}

Detection restore_box(float x1,
                      float y1,
                      float x2,
                      float y2,
                      int class_id,
                      float score,
                      const LetterBoxInfo& letterbox,
                      int image_width,
                      int image_height,
                      int model_width,
                      int model_height) {
  x1 = std::max(0.0f, std::min(static_cast<float>(model_width), x1 - letterbox.x_pad));
  y1 = std::max(0.0f, std::min(static_cast<float>(model_height), y1 - letterbox.y_pad));
  x2 = std::max(0.0f, std::min(static_cast<float>(model_width), x2 - letterbox.x_pad));
  y2 = std::max(0.0f, std::min(static_cast<float>(model_height), y2 - letterbox.y_pad));

  Detection det;
  det.class_id = class_id;
  det.score = score;
  det.x1 = std::max(0.0f, std::min(static_cast<float>(image_width - 1), x1 / letterbox.scale));
  det.y1 = std::max(0.0f, std::min(static_cast<float>(image_height - 1), y1 / letterbox.scale));
  det.x2 = std::max(0.0f, std::min(static_cast<float>(image_width - 1), x2 / letterbox.scale));
  det.y2 = std::max(0.0f, std::min(static_cast<float>(image_height - 1), y2 / letterbox.scale));
  return det;
}

void decode_branch(const PostprocessOsdConfig& config,
                   const InferenceOutputTensor& tensor,
                   const LetterBoxInfo& letterbox,
                   int image_width,
                   int image_height,
                   std::vector<Detection>& detections) {
  const OutputLayout layout = parse_layout(tensor);
  if (layout.channels <= 4 || layout.grid_h <= 0 || layout.grid_w <= 0) {
    std::fprintf(stderr,
                 "skip unsupported output index=%u dims=[%u,%u,%u,%u]\n",
                 tensor.index,
                 tensor.attr.dims[0],
                 tensor.attr.dims[1],
                 tensor.attr.dims[2],
                 tensor.attr.dims[3]);
    return;
  }

  const int inferred_classes = layout.channels - 4;
  const int classes = std::min(config.class_count > 0 ? config.class_count : inferred_classes, inferred_classes);
  const int stride = config.model_height / layout.grid_h;

  for (int h = 0; h < layout.grid_h; ++h) {
    for (int w = 0; w < layout.grid_w; ++w) {
      for (int cls = 0; cls < classes; ++cls) {
        const float score = read_tensor_value(tensor, layout, 4 + cls, h, w);
        if (score < config.box_threshold) {
          continue;
        }

        const float left = read_tensor_value(tensor, layout, 0, h, w);
        const float top = read_tensor_value(tensor, layout, 1, h, w);
        const float right = read_tensor_value(tensor, layout, 2, h, w);
        const float bottom = read_tensor_value(tensor, layout, 3, h, w);

        const float x1 = (static_cast<float>(w) + 0.5f - left) * stride;
        const float y1 = (static_cast<float>(h) + 0.5f - top) * stride;
        const float x2 = x1 + (left + right) * stride;
        const float y2 = y1 + (top + bottom) * stride;

        detections.push_back(restore_box(x1,
                                          y1,
                                          x2,
                                          y2,
                                          cls,
                                          score,
                                          letterbox,
                                          image_width,
                                          image_height,
                                          config.model_width,
                                          config.model_height));
      }
    }
  }
}

std::vector<Detection> decode_yolo26(const PostprocessOsdConfig& config, const InferenceResult& result) {
  std::vector<Detection> candidates;
  const auto& source = result.frame->source;
  for (const auto& output : result.outputs) {
    decode_branch(config, output, result.frame->letterbox, source->width, source->height, candidates);
  }
  return nms(std::move(candidates), config.nms_threshold, config.max_detections);
}

float compute_dfl_value(const InferenceOutputTensor& box_tensor,
                        const OutputLayout& layout,
                        int side,
                        int h,
                        int w,
                        int dfl_len) {
  float exp_sum = 0.0f;
  float weighted = 0.0f;
  float values[32];
  const int safe_len = std::min(dfl_len, 32);
  for (int i = 0; i < safe_len; ++i) {
    values[i] = read_tensor_value(box_tensor, layout, side * dfl_len + i, h, w);
  }
  const float max_value = *std::max_element(values, values + safe_len);
  for (int i = 0; i < safe_len; ++i) {
    const float ev = std::exp(values[i] - max_value);
    exp_sum += ev;
    weighted += ev * static_cast<float>(i);
  }
  return exp_sum > 0.0f ? weighted / exp_sum : 0.0f;
}

void decode_yolov8_branch(const PostprocessOsdConfig& config,
                          const InferenceOutputTensor& box_tensor,
                          const InferenceOutputTensor& score_tensor,
                          const InferenceOutputTensor* obj_tensor,
                          const LetterBoxInfo& letterbox,
                          int image_width,
                          int image_height,
                          std::vector<Detection>& detections) {
  const OutputLayout box_layout = parse_layout(box_tensor);
  const OutputLayout score_layout = parse_layout(score_tensor);
  OutputLayout obj_layout;
  if (obj_tensor) {
    obj_layout = parse_layout(*obj_tensor);
  }
  if (box_layout.channels < config.dfl_len * 4 || score_layout.channels <= 0 || box_layout.grid_h <= 0 || box_layout.grid_w <= 0) {
    return;
  }
  if (score_layout.grid_h != box_layout.grid_h || score_layout.grid_w != box_layout.grid_w) {
    return;
  }
  if (obj_tensor && (obj_layout.channels <= 0 || obj_layout.grid_h != box_layout.grid_h || obj_layout.grid_w != box_layout.grid_w)) {
    obj_tensor = nullptr;
  }

  const int classes = std::min(config.class_count > 0 ? config.class_count : score_layout.channels, score_layout.channels);
  const float stride_x = static_cast<float>(config.model_width) / static_cast<float>(box_layout.grid_w);
  const float stride_y = static_cast<float>(config.model_height) / static_cast<float>(box_layout.grid_h);
  float branch_max_score = 0.0f;
  int branch_candidates = 0;

  for (int h = 0; h < box_layout.grid_h; ++h) {
    for (int w = 0; w < box_layout.grid_w; ++w) {
      int best_class = -1;
      float best_score = config.box_threshold;
      for (int cls = 0; cls < classes; ++cls) {
        const float score = read_yolov8_score(score_tensor, score_layout, cls, h, w);
        branch_max_score = std::max(branch_max_score, score);
        if (score > best_score) {
          best_score = score;
          best_class = cls;
        }
      }
      if (best_class < 0) {
        continue;
      }
      ++branch_candidates;

      const float left = compute_dfl_value(box_tensor, box_layout, 0, h, w, config.dfl_len);
      const float top = compute_dfl_value(box_tensor, box_layout, 1, h, w, config.dfl_len);
      const float right = compute_dfl_value(box_tensor, box_layout, 2, h, w, config.dfl_len);
      const float bottom = compute_dfl_value(box_tensor, box_layout, 3, h, w, config.dfl_len);

      const float x1 = (static_cast<float>(w) + 0.5f - left) * stride_x;
      const float y1 = (static_cast<float>(h) + 0.5f - top) * stride_y;
      const float x2 = (static_cast<float>(w) + 0.5f + right) * stride_x;
      const float y2 = (static_cast<float>(h) + 0.5f + bottom) * stride_y;

      detections.push_back(restore_box(x1,
                                        y1,
                                        x2,
                                        y2,
                                        best_class,
                                        best_score,
                                        letterbox,
                                        image_width,
                                        image_height,
                                        config.model_width,
                                        config.model_height));
    }
  }

  if (config.verbose) {
    std::fprintf(stderr,
                 "yolov8 branch grid=%dx%d box_index=%u score_index=%u obj_index=%d max_score=%.6f candidates=%d\n",
                 box_layout.grid_w,
                 box_layout.grid_h,
                 box_tensor.index,
                 score_tensor.index,
                 obj_tensor ? static_cast<int>(obj_tensor->index) : -1,
                 branch_max_score,
                 branch_candidates);
  }
}

struct YoloV8BranchGroup {
  int grid_h = 0;
  int grid_w = 0;
  const InferenceOutputTensor* box = nullptr;
  const InferenceOutputTensor* score = nullptr;
  const InferenceOutputTensor* obj = nullptr;
  std::vector<const InferenceOutputTensor*> one_channel_tensors;
};

bool same_grid(const YoloV8BranchGroup& group, const OutputLayout& layout) {
  return group.grid_h == layout.grid_h && group.grid_w == layout.grid_w;
}

YoloV8BranchGroup& find_or_add_group(std::vector<YoloV8BranchGroup>& groups, const OutputLayout& layout) {
  for (auto& group : groups) {
    if (same_grid(group, layout)) {
      return group;
    }
  }

  YoloV8BranchGroup group;
  group.grid_h = layout.grid_h;
  group.grid_w = layout.grid_w;
  groups.push_back(group);
  return groups.back();
}

std::vector<YoloV8BranchGroup> group_yolov8_outputs(const PostprocessOsdConfig& config,
                                                    const std::vector<InferenceOutputTensor>& outputs) {
  std::vector<YoloV8BranchGroup> groups;
  for (const auto& output : outputs) {
    const OutputLayout layout = parse_layout(output);
    if (layout.grid_h <= 0 || layout.grid_w <= 0 || layout.channels <= 0) {
      continue;
    }

    auto& group = find_or_add_group(groups, layout);
    if (layout.channels == config.dfl_len * 4) {
      group.box = &output;
    } else if (layout.channels == config.class_count && config.class_count > 1) {
      group.score = &output;
    } else if (layout.channels == 1) {
      group.one_channel_tensors.push_back(&output);
    } else if (layout.channels == config.class_count) {
      group.score = &output;
    }
  }

  for (auto& group : groups) {
    if (config.class_count == 1 && !group.one_channel_tensors.empty()) {
      group.score = group.one_channel_tensors.back();
      if (group.one_channel_tensors.size() > 1) {
        group.obj = group.one_channel_tensors.front();
      }
    } else if (!group.one_channel_tensors.empty()) {
      group.obj = group.one_channel_tensors.front();
    }
  }

  std::sort(groups.begin(), groups.end(), [](const YoloV8BranchGroup& a, const YoloV8BranchGroup& b) {
    return a.grid_h * a.grid_w > b.grid_h * b.grid_w;
  });
  return groups;
}

std::vector<Detection> decode_yolov8_dfl(const PostprocessOsdConfig& config, const InferenceResult& result) {
  std::vector<Detection> candidates;
  const auto& source = result.frame->source;
  const auto groups = group_yolov8_outputs(config, result.outputs);
  for (const auto& group : groups) {
    if (!group.box || !group.score) {
      continue;
    }
    decode_yolov8_branch(config,
                         *group.box,
                         *group.score,
                         group.obj,
                         result.frame->letterbox,
                         source->width,
                         source->height,
                         candidates);
  }
  return nms(std::move(candidates), config.nms_threshold, config.max_detections);
}

PostprocessModel resolve_model(const PostprocessOsdConfig& config, const InferenceResult& result) {
  if (config.model != PostprocessModel::Auto) {
    return config.model;
  }
  return result.outputs.size() == 9 ? PostprocessModel::YoloV8Dfl : PostprocessModel::Yolo26;
}

std::vector<Detection> decode_detections(const PostprocessOsdConfig& config, const InferenceResult& result) {
  switch (resolve_model(config, result)) {
    case PostprocessModel::YoloV8Dfl:
      return decode_yolov8_dfl(config, result);
    case PostprocessModel::Yolo26:
    case PostprocessModel::Auto:
    default:
      return decode_yolo26(config, result);
  }
}

void draw_rect_y_plane(const DecodedFrame& frame, const Detection& det, int thickness) {
  if (!frame.frame || thickness <= 0) {
    return;
  }

  MppBuffer buffer = mpp_frame_get_buffer(frame.frame);
  if (!buffer) {
    return;
  }

  auto* y = static_cast<uint8_t*>(mpp_buffer_get_ptr(buffer));
  if (!y) {
    return;
  }

  const int x1 = clamp_int(det.x1, 0, frame.width - 1);
  const int y1 = clamp_int(det.y1, 0, frame.height - 1);
  const int x2 = clamp_int(det.x2, 0, frame.width - 1);
  const int y2 = clamp_int(det.y2, 0, frame.height - 1);
  if (x2 <= x1 || y2 <= y1) {
    return;
  }

  constexpr uint8_t kLuma = 235;
  for (int t = 0; t < thickness; ++t) {
    const int top = std::min(frame.height - 1, y1 + t);
    const int bottom = std::max(0, y2 - t);
    for (int x = x1; x <= x2; ++x) {
      y[top * frame.hor_stride + x] = kLuma;
      y[bottom * frame.hor_stride + x] = kLuma;
    }

    const int left = std::min(frame.width - 1, x1 + t);
    const int right = std::max(0, x2 - t);
    for (int yy = y1; yy <= y2; ++yy) {
      y[yy * frame.hor_stride + left] = kLuma;
      y[yy * frame.hor_stride + right] = kLuma;
    }
  }
}

}  // namespace

void postprocess_osd_thread(const PostprocessOsdConfig& config,
                            StopFlag& stop,
                            InferenceResultQueue& input,
                            OsdFrameQueue& output) {
  const auto stage_start = std::chrono::steady_clock::now();
  int64_t processed = 0;
  int64_t total_detections = 0;
  double decode_total_ms = 0.0;
  double osd_total_ms = 0.0;

  InferenceResultPtr inference;
  while (!stop.stop_requested() && input.pop(inference)) {
    if (!inference || !inference->frame || !inference->frame->source) {
      std::fprintf(stderr, "postprocess received invalid inference result\n");
      stop.request_stop();
      break;
    }

    const auto decode_start = std::chrono::steady_clock::now();
    auto detections = decode_detections(config, *inference);
    const auto decode_end = std::chrono::steady_clock::now();
    decode_total_ms += std::chrono::duration<double, std::milli>(decode_end - decode_start).count();

    const auto osd_start = std::chrono::steady_clock::now();
    if (config.draw_osd) {
      for (const auto& det : detections) {
        draw_rect_y_plane(*inference->frame->source, det, config.line_thickness);
      }
    }
    const auto osd_end = std::chrono::steady_clock::now();
    osd_total_ms += std::chrono::duration<double, std::milli>(osd_end - osd_start).count();

    total_detections += static_cast<int64_t>(detections.size());
    if (config.verbose || processed % 60 == 0) {
      std::fprintf(stderr,
                   "postprocessed frame=%ld detections=%zu\n",
                   inference->frame->source->frame_id,
                   detections.size());
    }

    auto osd_frame = std::make_shared<OsdFrame>();
    osd_frame->inference = std::move(inference);
    osd_frame->detections = std::move(detections);
    ++processed;

    if (!output.push(std::move(osd_frame))) {
      break;
    }
  }

  const auto stage_end = std::chrono::steady_clock::now();
  const double elapsed_ms = std::chrono::duration<double, std::milli>(stage_end - stage_start).count();
  const double fps = elapsed_ms > 0.0 ? static_cast<double>(processed) * 1000.0 / elapsed_ms : 0.0;
  std::fprintf(stderr,
               "[PERF] postprocess_osd frames=%ld elapsed_ms=%.3f avg_stage_ms=%.3f fps=%.2f decode_avg_ms=%.3f osd_avg_ms=%.3f total_detections=%ld\n",
               processed,
               elapsed_ms,
               processed > 0 ? elapsed_ms / static_cast<double>(processed) : 0.0,
               fps,
               processed > 0 ? decode_total_ms / static_cast<double>(processed) : 0.0,
               processed > 0 ? osd_total_ms / static_cast<double>(processed) : 0.0,
               total_detections);
  std::fprintf(stderr, "postprocess_osd thread finished, frames=%ld\n", processed);
  output.close();
}

}  // namespace rkai