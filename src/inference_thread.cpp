#include "inference_thread.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

namespace rkai {
namespace {

bool output_should_be_float(const InferenceConfig& config, const rknn_tensor_attr& attr) {
  if (config.want_float_output) {
    return true;
  }
  return attr.qnt_type == RKNN_TENSOR_QNT_NONE;
}

bool copy_rknn_outputs(const InferenceConfig& config,
                       RknnInputRuntime& runtime,
                       std::vector<InferenceOutputTensor>& copied_outputs,
                       double& get_ms) {
  const auto& output_attrs = runtime.output_attrs();
  std::vector<rknn_output> outputs(output_attrs.size());

  for (size_t i = 0; i < outputs.size(); ++i) {
    outputs[i].index = static_cast<uint32_t>(i);
    outputs[i].want_float = output_should_be_float(config, output_attrs[i]) ? 1 : 0;
    outputs[i].is_prealloc = 0;
  }

  const auto get_start = std::chrono::steady_clock::now();
  int ret = rknn_outputs_get(runtime.context(), static_cast<uint32_t>(outputs.size()), outputs.data(), nullptr);
  const auto get_end = std::chrono::steady_clock::now();
  get_ms = std::chrono::duration<double, std::milli>(get_end - get_start).count();

  if (ret < 0) {
    std::fprintf(stderr, "rknn_outputs_get failed ret=%d\n", ret);
    return false;
  }

  copied_outputs.clear();
  copied_outputs.reserve(outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    InferenceOutputTensor tensor;
    tensor.index = static_cast<uint32_t>(i);
    tensor.attr = output_attrs[i];
    tensor.want_float = outputs[i].want_float != 0;

    const size_t bytes = tensor.want_float
                             ? static_cast<size_t>(tensor.attr.n_elems) * sizeof(float)
                             : static_cast<size_t>(tensor.attr.size);
    tensor.data.resize(bytes);
    if (outputs[i].buf && bytes > 0) {
      std::memcpy(tensor.data.data(), outputs[i].buf, bytes);
    }
    copied_outputs.push_back(std::move(tensor));
  }

  rknn_outputs_release(runtime.context(), static_cast<uint32_t>(outputs.size()), outputs.data());
  return true;
}

bool run_inference_once(const InferenceConfig& config,
                        RknnInputRuntime& runtime,
                        const PreprocessedFramePtr& frame,
                        InferenceResult& result) {
  if (!frame || !frame->input.mem || frame->input.dma_fd < 0) {
    std::fprintf(stderr, "inference received invalid input tensor\n");
    return false;
  }

  std::lock_guard<std::mutex> lock(runtime.context_mutex());
  if (!runtime.bind_input_mem(frame->input.mem)) {
    return false;
  }

  const auto total_start = std::chrono::steady_clock::now();
  const auto run_start = std::chrono::steady_clock::now();
  int ret = rknn_run(runtime.context(), nullptr);
  const auto run_end = std::chrono::steady_clock::now();
  result.run_ms = std::chrono::duration<double, std::milli>(run_end - run_start).count();

  if (ret < 0) {
    std::fprintf(stderr, "rknn_run failed ret=%d\n", ret);
    return false;
  }

  if (!copy_rknn_outputs(config, runtime, result.outputs, result.output_get_ms)) {
    return false;
  }

  const auto total_end = std::chrono::steady_clock::now();
  result.total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
  return true;
}

}  // namespace

void inference_thread(const InferenceConfig& config,
                      StopFlag& stop,
                      PreprocessedFrameQueue& input,
                      InferenceResultQueue& output,
                      RknnInputRuntime& runtime) {
  const auto stage_start = std::chrono::steady_clock::now();
  int64_t inferred = 0;
  double run_total_ms = 0.0;
  double get_total_ms = 0.0;
  double infer_total_ms = 0.0;
  double infer_max_ms = 0.0;

  PreprocessedFramePtr preprocessed;
  while (!stop.stop_requested() && input.pop(preprocessed)) {
    auto result = std::make_shared<InferenceResult>();
    result->frame = preprocessed;

    const bool ok = run_inference_once(config, runtime, preprocessed, *result);
    rknn_tensor_mem* input_mem = preprocessed ? preprocessed->input.mem : nullptr;
    if (preprocessed) {
      preprocessed->input.mem = nullptr;
    }
    runtime.release(input_mem);

    if (!ok) {
      stop.request_stop();
      break;
    }

    run_total_ms += result->run_ms;
    get_total_ms += result->output_get_ms;
    infer_total_ms += result->total_ms;
    infer_max_ms = std::max(infer_max_ms, result->total_ms);

    if (config.verbose || inferred % 60 == 0) {
      std::fprintf(stderr,
                   "inferred frame=%ld outputs=%zu run_ms=%.3f get_ms=%.3f total_ms=%.3f\n",
                   result->frame->source->frame_id,
                   result->outputs.size(),
                   result->run_ms,
                   result->output_get_ms,
                   result->total_ms);
    }
    ++inferred;

    if (!output.push(std::move(result))) {
      break;
    }
  }

  const auto stage_end = std::chrono::steady_clock::now();
  const double elapsed_ms = std::chrono::duration<double, std::milli>(stage_end - stage_start).count();
  const double fps = elapsed_ms > 0.0 ? static_cast<double>(inferred) * 1000.0 / elapsed_ms : 0.0;
  std::fprintf(stderr,
               "[PERF] inference frames=%ld elapsed_ms=%.3f avg_stage_ms=%.3f fps=%.2f run_avg_ms=%.3f outputs_get_avg_ms=%.3f infer_avg_ms=%.3f infer_max_ms=%.3f\n",
               inferred,
               elapsed_ms,
               inferred > 0 ? elapsed_ms / static_cast<double>(inferred) : 0.0,
               fps,
               inferred > 0 ? run_total_ms / static_cast<double>(inferred) : 0.0,
               inferred > 0 ? get_total_ms / static_cast<double>(inferred) : 0.0,
               inferred > 0 ? infer_total_ms / static_cast<double>(inferred) : 0.0,
               infer_max_ms);
  std::fprintf(stderr, "inference thread finished, inferred_frames=%ld\n", inferred);
  output.close();
}

}  // namespace rkai