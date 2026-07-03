#pragma once

#include <opencv2/core.hpp>

extern "C" {
#include <rknn_api.h>
}

#include <string>
#include <vector>

namespace rkai {

struct LicensePlateOcrConfig {
  std::string model_path;
  std::string vocab_path;
  bool verbose = false;
};

struct LicensePlateOcrResult {
  std::string text;
  float score = 0.0f;
  int char_count = 0;
  int sequence_length = 0;
  int class_count = 0;
  bool ok = false;
  std::string error;
};

struct LicensePlateOcrInputInfo {
  int width = 0;
  int height = 0;
  int channels = 0;
  rknn_tensor_format format = RKNN_TENSOR_NHWC;
  rknn_tensor_type type = RKNN_TENSOR_FLOAT32;
  bool supports_direct_uint8_mem = false;
};

struct LicensePlateOcrInputMem {
  rknn_tensor_mem* mem = nullptr;
  rknn_tensor_attr attr{};
};

class LicensePlateOcrRecognizer {
 public:
  LicensePlateOcrRecognizer();
  ~LicensePlateOcrRecognizer();

  LicensePlateOcrRecognizer(const LicensePlateOcrRecognizer&) = delete;
  LicensePlateOcrRecognizer& operator=(const LicensePlateOcrRecognizer&) = delete;

  bool init(const LicensePlateOcrConfig& config);
  void release();
  bool ready() const;
  LicensePlateOcrInputInfo input_info() const;
  LicensePlateOcrInputMem create_input_mem();
  void destroy_input_mem(LicensePlateOcrInputMem* input_mem);

  LicensePlateOcrResult recognize(const cv::Mat& bgr_image);
  LicensePlateOcrResult recognize_preprocessed_mem(LicensePlateOcrInputMem* input_mem);

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace rkai