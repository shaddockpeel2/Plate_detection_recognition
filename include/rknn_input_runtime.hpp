#pragma once

#include "decoder_thread.hpp"
#include "pipeline_types.hpp"

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

extern "C" {
#include <rknn_api.h>
}

namespace rkai {

struct RknnTensorRuntimeConfig {
  std::string model_path;
  int input_pool_size = 4;
};

struct ModelInputInfo {
  int width = 0;
  int height = 0;
  int channels = 0;
  bool rgb = true;
  bool quantized = false;
  rknn_tensor_format layout = RKNN_TENSOR_NHWC;
  rknn_tensor_type type = RKNN_TENSOR_UINT8;
};

class RknnInputRuntime {
 public:
  RknnInputRuntime() = default;
  RknnInputRuntime(const RknnInputRuntime&) = delete;
  RknnInputRuntime& operator=(const RknnInputRuntime&) = delete;
  ~RknnInputRuntime();

  bool init(const RknnTensorRuntimeConfig& config);
  RknnInputTensor acquire(StopFlag& stop);
  void release(rknn_tensor_mem* mem);
  bool bind_input_mem(rknn_tensor_mem* mem);
  std::mutex& context_mutex() { return context_mutex_; }

  rknn_context context() const { return ctx_; }
  const rknn_input_output_num& io_num() const { return io_num_; }
  const rknn_tensor_attr& input_attr() const { return input_attr_; }
  const std::vector<rknn_tensor_attr>& output_attrs() const { return output_attrs_; }
  const ModelInputInfo& input_info() const { return input_info_; }

 private:
  rknn_context ctx_ = 0;
  rknn_input_output_num io_num_{};
  rknn_tensor_attr input_attr_{};
  std::vector<rknn_tensor_attr> output_attrs_;
  ModelInputInfo input_info_{};
  std::vector<rknn_tensor_mem*> input_pool_;
  std::queue<rknn_tensor_mem*> free_inputs_;
  std::mutex mutex_;
  std::mutex context_mutex_;
  std::condition_variable cv_;
};

}  // namespace rkai