#include "preprocess_thread.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "RgaUtils.h"
#include "im2d.h"
#include "rga.h"

namespace rkai {
namespace {

LetterBoxInfo compute_letterbox(int src_w, int src_h, int dst_w, int dst_h) {
  LetterBoxInfo box;
  const float scale = std::min(static_cast<float>(dst_w) / static_cast<float>(src_w),
                               static_cast<float>(dst_h) / static_cast<float>(src_h));
  box.scale = scale;
  box.resized_width = static_cast<int>(static_cast<float>(src_w) * scale + 0.5f);
  box.resized_height = static_cast<int>(static_cast<float>(src_h) * scale + 0.5f);
  box.x_pad = (dst_w - box.resized_width) / 2;
  box.y_pad = (dst_h - box.resized_height) / 2;
  return box;
}

LetterBoxInfo compute_resize(int, int, int dst_w, int dst_h) {
  LetterBoxInfo box;
  box.scale = 1.0f;
  box.resized_width = dst_w;
  box.resized_height = dst_h;
  box.x_pad = 0;
  box.y_pad = 0;
  return box;
}

int rga_input_format(bool rgb) {
  return rgb ? RK_FORMAT_RGB_888 : RK_FORMAT_BGR_888;
}

bool clear_letterbox_background(const RknnInputTensor& tensor) {
  if (!tensor.virt_addr) {
    return false;
  }
  std::memset(tensor.virt_addr, 114, tensor.size);
  return true;
}

bool validate_frame(const DecodedFrame& frame) {
  if (frame.dma_fd < 0) {
    std::fprintf(stderr, "preprocess received invalid source dma_fd frame=%ld\n", frame.frame_id);
    return false;
  }
  if (frame.width <= 0 || frame.height <= 0 || frame.hor_stride <= 0 || frame.ver_stride <= 0) {
    std::fprintf(stderr, "preprocess received invalid frame geometry frame=%ld\n", frame.frame_id);
    return false;
  }
  return true;
}

bool configure_rga_scheduler() {
  const uint64_t rga3_cores = IM_SCHEDULER_RGA3_CORE0 | IM_SCHEDULER_RGA3_CORE1;
  const IM_STATUS status = imconfig(IM_CONFIG_SCHEDULER_CORE, rga3_cores);
  if (status != IM_STATUS_SUCCESS) {
    std::fprintf(stderr, "RGA scheduler config failed status=%d message=%s\n", status, imStrError(status));
    return false;
  }
  std::fprintf(stderr, "RGA scheduler pinned to RGA3 cores mask=0x%lx\n", static_cast<unsigned long>(rga3_cores));
  return true;
}

bool run_rga_preprocess(const PreprocessConfig& config,
                        const DecodedFrame& src_frame,
                        const RknnInputTensor& dst_tensor,
                        const LetterBoxInfo& box,
                        double& rga_ms) {
  rga_buffer_t src = wrapbuffer_fd(src_frame.dma_fd,
                                   src_frame.hor_stride,
                                   src_frame.ver_stride,
                                   RK_FORMAT_YCbCr_420_SP,
                                   src_frame.hor_stride,
                                   src_frame.ver_stride);

  rga_buffer_t dst = wrapbuffer_fd(dst_tensor.dma_fd,
                                   dst_tensor.width,
                                   dst_tensor.height,
                                   rga_input_format(config.model_rgb));

  im_rect src_rect{0, 0, src_frame.width, src_frame.height};
  im_rect dst_rect{box.x_pad, box.y_pad, box.resized_width, box.resized_height};

  const auto rga_start = std::chrono::steady_clock::now();
  IM_STATUS status = improcess(src, dst, {}, src_rect, dst_rect, {}, IM_SYNC);
  const auto rga_end = std::chrono::steady_clock::now();
  rga_ms = std::chrono::duration<double, std::milli>(rga_end - rga_start).count();
  if (status != IM_STATUS_SUCCESS) {
    std::fprintf(stderr,
                 "RGA preprocess failed frame=%ld status=%d message=%s\n",
                 src_frame.frame_id,
                 status,
                 imStrError(status));
    return false;
  }
  return true;
}

bool quantize_uint8_to_int8_input(const RknnInputTensor& tensor, const rknn_tensor_attr& input_attr) {
  if (!tensor.virt_addr || input_attr.type != RKNN_TENSOR_INT8 || input_attr.qnt_type == RKNN_TENSOR_QNT_NONE) {
    return true;
  }

  auto* data = static_cast<int8_t*>(tensor.virt_addr);
  const size_t elems = static_cast<size_t>(tensor.width) * tensor.height * tensor.channels;
  for (size_t i = 0; i < elems; ++i) {
    const int value = static_cast<int>(reinterpret_cast<uint8_t*>(data)[i]);
    data[i] = static_cast<int8_t>(std::max(-128, std::min(127, value + input_attr.zp)));
  }
  return true;
}

}  // namespace

void preprocess_thread(const PreprocessConfig& config,
                       StopFlag& stop,
                       DecodedFrameQueue& input,
                       PreprocessedFrameQueue& output,
                       RknnInputRuntime& rknn_inputs) {
  const auto stage_start = std::chrono::steady_clock::now();
  double rga_total_ms = 0.0;
  double rga_max_ms = 0.0;
  int64_t processed = 0;
  DecodedFramePtr decoded;

  if (!configure_rga_scheduler()) {
    stop.request_stop();
    output.close();
    return;
  }

  while (!stop.stop_requested() && input.pop(decoded)) {
    if (!decoded || !validate_frame(*decoded)) {
      stop.request_stop();
      break;
    }

    RknnInputTensor tensor = rknn_inputs.acquire(stop);
    if (!tensor.mem) {
      break;
    }
    if (tensor.dma_fd < 0) {
      std::fprintf(stderr, "rknn input tensor has invalid dma_fd\n");
      rknn_inputs.release(tensor.mem);
      stop.request_stop();
      break;
    }

    const LetterBoxInfo box = config.letterbox
                                  ? compute_letterbox(decoded->width, decoded->height, tensor.width, tensor.height)
                                  : compute_resize(decoded->width, decoded->height, tensor.width, tensor.height);

    if (config.letterbox && !clear_letterbox_background(tensor)) {
      std::fprintf(stderr, "failed to clear letterbox background: no virtual address\n");
      rknn_inputs.release(tensor.mem);
      stop.request_stop();
      break;
    }

    double rga_ms = 0.0;
    if (!run_rga_preprocess(config, *decoded, tensor, box, rga_ms)) {
      rknn_inputs.release(tensor.mem);
      stop.request_stop();
      break;
    }
    if (!quantize_uint8_to_int8_input(tensor, rknn_inputs.input_attr())) {
      std::fprintf(stderr, "failed to quantize rknn input tensor\n");
      rknn_inputs.release(tensor.mem);
      stop.request_stop();
      break;
    }
    rga_total_ms += rga_ms;
    rga_max_ms = std::max(rga_max_ms, rga_ms);

    auto preprocessed = std::make_shared<PreprocessedFrame>();
    preprocessed->source = std::move(decoded);
    preprocessed->input = tensor;
    preprocessed->letterbox = box;

    if (config.verbose || processed % 60 == 0) {
      std::fprintf(stderr,
                   "preprocessed frame=%ld src_fd=%d input_fd=%d dst=%dx%d letterbox=%dx%d+%d+%d\n",
                   preprocessed->source->frame_id,
                   preprocessed->source->dma_fd,
                   preprocessed->input.dma_fd,
                   preprocessed->input.width,
                   preprocessed->input.height,
                   preprocessed->letterbox.resized_width,
                   preprocessed->letterbox.resized_height,
                   preprocessed->letterbox.x_pad,
                   preprocessed->letterbox.y_pad);
    }
    ++processed;

    if (!output.push(std::move(preprocessed))) {
      rknn_inputs.release(tensor.mem);
      break;
    }
  }

  const auto stage_end = std::chrono::steady_clock::now();
  const double elapsed_ms = std::chrono::duration<double, std::milli>(stage_end - stage_start).count();
  const double fps = elapsed_ms > 0.0 ? static_cast<double>(processed) * 1000.0 / elapsed_ms : 0.0;
  std::fprintf(stderr,
               "[PERF] preprocess frames=%ld elapsed_ms=%.3f avg_stage_ms=%.3f fps=%.2f rga_avg_ms=%.3f rga_max_ms=%.3f\n",
               processed,
               elapsed_ms,
               processed > 0 ? elapsed_ms / static_cast<double>(processed) : 0.0,
               fps,
               processed > 0 ? rga_total_ms / static_cast<double>(processed) : 0.0,
               rga_max_ms);
  std::fprintf(stderr, "preprocess thread finished, processed_frames=%ld\n", processed);
  output.close();
}

}  // namespace rkai