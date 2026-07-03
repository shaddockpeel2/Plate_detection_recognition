#include "license_plate_ocr.hpp"

#include <rknn_api.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <utility>

#include <opencv2/imgproc.hpp>

namespace rkai {
namespace {

constexpr int kDefaultInputHeight = 48;
constexpr int kDefaultInputWidth = 320;
constexpr int kBlankClass = 0;

using ModelBuffer = std::unique_ptr<unsigned char, void (*)(void*)>;

ModelBuffer load_model_file(const std::string& path, int* model_size) {
  FILE* fp = std::fopen(path.c_str(), "rb");
  if (fp == nullptr) {
    return ModelBuffer(nullptr, std::free);
  }

  std::fseek(fp, 0, SEEK_END);
  const long length = std::ftell(fp);
  std::fseek(fp, 0, SEEK_SET);
  if (length <= 0) {
    std::fclose(fp);
    return ModelBuffer(nullptr, std::free);
  }

  auto* raw = static_cast<unsigned char*>(std::malloc(static_cast<std::size_t>(length)));
  if (raw == nullptr) {
    std::fclose(fp);
    return ModelBuffer(nullptr, std::free);
  }

  const std::size_t read_size = std::fread(raw, 1, static_cast<std::size_t>(length), fp);
  std::fclose(fp);
  if (read_size != static_cast<std::size_t>(length)) {
    std::free(raw);
    return ModelBuffer(nullptr, std::free);
  }

  *model_size = static_cast<int>(length);
  return ModelBuffer(raw, std::free);
}

bool load_vocab_file(const std::string& path, std::vector<std::string>* vocab, std::string* error) {
  std::ifstream file(path.c_str());
  if (!file.is_open()) {
    *error = "failed to open vocab: " + path;
    return false;
  }

  std::vector<std::string> loaded;
  std::string line;
  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (!line.empty()) {
      loaded.push_back(line);
    }
  }

  if (loaded.empty()) {
    *error = "empty vocab: " + path;
    return false;
  }

  *vocab = std::move(loaded);
  return true;
}

std::string dims_to_string(const rknn_tensor_attr& attr) {
  std::string text = "[";
  for (uint32_t i = 0; i < attr.n_dims; ++i) {
    if (i > 0) {
      text += ",";
    }
    text += std::to_string(attr.dims[i]);
  }
  text += "]";
  return text;
}

cv::Mat resize_with_padding(const cv::Mat& bgr_image, int target_h, int target_w) {
  const float ratio = bgr_image.cols / std::max(1.0f, static_cast<float>(bgr_image.rows));
  const int resized_w = std::min(target_w, std::max(1, static_cast<int>(std::ceil(target_h * ratio))));

  cv::Mat resized;
  cv::resize(bgr_image, resized, cv::Size(resized_w, target_h), 0, 0, cv::INTER_LINEAR);
  resized.convertTo(resized, CV_32FC3);
  resized = (resized - 127.5f) / 127.5f;

  cv::Mat padded = cv::Mat::zeros(target_h, target_w, CV_32FC3);
  resized.copyTo(padded(cv::Rect(0, 0, resized_w, target_h)));
  return padded;
}

int output_sequence_length(const rknn_tensor_attr& attr, int model_width) {
  const int expected = std::max(1, model_width / 8);
  for (uint32_t i = 0; i < attr.n_dims; ++i) {
    const int dim = static_cast<int>(attr.dims[i]);
    if (dim == 40 || dim == expected) {
      return dim;
    }
  }
  return expected;
}

int output_class_count(const rknn_tensor_attr& attr, int sequence_length, int vocab_size) {
  for (int i = static_cast<int>(attr.n_dims) - 1; i >= 0; --i) {
    const int dim = static_cast<int>(attr.dims[i]);
    if (dim > 1 && dim != sequence_length) {
      return dim;
    }
  }
  return vocab_size + 1;
}

LicensePlateOcrResult ctc_decode(const float* out_data,
                                  int sequence_length,
                                  int class_count,
                                  const std::vector<std::string>& vocab) {
  LicensePlateOcrResult result;
  result.sequence_length = sequence_length;
  result.class_count = class_count;

  std::string text;
  float score_sum = 0.0f;
  int selected_count = 0;
  int last_index = -1;

  for (int n = 0; n < sequence_length; ++n) {
    const float* row = out_data + n * class_count;
    int argmax_idx = 0;
    float max_value = row[0];
    for (int j = 1; j < class_count; ++j) {
      if (row[j] > max_value) {
        max_value = row[j];
        argmax_idx = j;
      }
    }

    if (argmax_idx != kBlankClass && argmax_idx != last_index) {
      const int char_index = argmax_idx - 1;
      if (char_index >= 0 && char_index < static_cast<int>(vocab.size())) {
        text += vocab[char_index];
        score_sum += max_value;
        selected_count += 1;
      }
    }
    last_index = argmax_idx;
  }

  result.text = text;
  result.char_count = selected_count;
  result.score = selected_count > 0 ? score_sum / selected_count : 0.0f;
  result.ok = true;
  return result;
}

}  // namespace

struct LicensePlateOcrRecognizer::Impl {
  LicensePlateOcrConfig config;
  std::vector<std::string> vocab;
  rknn_context ctx = 0;
  rknn_input_output_num io_num{};
  std::vector<rknn_tensor_attr> input_attrs;
  std::vector<rknn_tensor_attr> output_attrs;
  int model_channel = 3;
  int model_width = kDefaultInputWidth;
  int model_height = kDefaultInputHeight;
  bool initialized = false;
};

LicensePlateOcrRecognizer::LicensePlateOcrRecognizer() : impl_(new Impl()) {}

LicensePlateOcrRecognizer::~LicensePlateOcrRecognizer() {
  release();
  delete impl_;
}

bool LicensePlateOcrRecognizer::init(const LicensePlateOcrConfig& config) {
  release();
  impl_->config = config;

  std::string error;
  if (!load_vocab_file(config.vocab_path, &impl_->vocab, &error)) {
    if (config.verbose) {
      std::fprintf(stderr, "[OCR] %s\n", error.c_str());
    }
    return false;
  }

  int model_size = 0;
  ModelBuffer model = load_model_file(config.model_path, &model_size);
  if (!model) {
    if (config.verbose) {
      std::fprintf(stderr, "[OCR] failed to load model: %s\n", config.model_path.c_str());
    }
    return false;
  }

  int ret = rknn_init(&impl_->ctx, model.get(), model_size, 0, nullptr);
  if (ret < 0) {
    if (config.verbose) {
      std::fprintf(stderr, "[OCR] rknn_init failed ret=%d model=%s\n", ret, config.model_path.c_str());
    }
    impl_->ctx = 0;
    return false;
  }

  ret = rknn_query(impl_->ctx, RKNN_QUERY_IN_OUT_NUM, &impl_->io_num, sizeof(impl_->io_num));
  if (ret != RKNN_SUCC) {
    if (config.verbose) {
      std::fprintf(stderr, "[OCR] RKNN_QUERY_IN_OUT_NUM failed ret=%d\n", ret);
    }
    release();
    return false;
  }

  impl_->input_attrs.assign(impl_->io_num.n_input, rknn_tensor_attr());
  for (uint32_t i = 0; i < impl_->io_num.n_input; ++i) {
    impl_->input_attrs[i].index = i;
    ret = rknn_query(impl_->ctx, RKNN_QUERY_INPUT_ATTR, &impl_->input_attrs[i], sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      if (config.verbose) {
        std::fprintf(stderr, "[OCR] RKNN_QUERY_INPUT_ATTR failed index=%u ret=%d\n", i, ret);
      }
      release();
      return false;
    }
  }

  impl_->output_attrs.assign(impl_->io_num.n_output, rknn_tensor_attr());
  for (uint32_t i = 0; i < impl_->io_num.n_output; ++i) {
    impl_->output_attrs[i].index = i;
    ret = rknn_query(impl_->ctx, RKNN_QUERY_OUTPUT_ATTR, &impl_->output_attrs[i], sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      if (config.verbose) {
        std::fprintf(stderr, "[OCR] RKNN_QUERY_OUTPUT_ATTR failed index=%u ret=%d\n", i, ret);
      }
      release();
      return false;
    }
  }

  const rknn_tensor_attr& input_attr = impl_->input_attrs[0];
  if (input_attr.fmt == RKNN_TENSOR_NCHW) {
    impl_->model_channel = input_attr.dims[1];
    impl_->model_height = input_attr.dims[2];
    impl_->model_width = input_attr.dims[3];
  } else {
    impl_->model_height = input_attr.dims[1];
    impl_->model_width = input_attr.dims[2];
    impl_->model_channel = input_attr.dims[3];
  }

  if (impl_->model_height <= 0) {
    impl_->model_height = kDefaultInputHeight;
  }
  if (impl_->model_width <= 0) {
    impl_->model_width = kDefaultInputWidth;
  }
  if (impl_->model_channel <= 0) {
    impl_->model_channel = 3;
  }

  impl_->initialized = true;
  if (config.verbose) {
    std::fprintf(stderr,
                 "[OCR] ready model=%s vocab=%zu input=%dx%dx%d output=%s\n",
                 config.model_path.c_str(),
                 impl_->vocab.size(),
                 impl_->model_width,
                 impl_->model_height,
                 impl_->model_channel,
                 dims_to_string(impl_->output_attrs[0]).c_str());
  }
  return true;
}

void LicensePlateOcrRecognizer::release() {
  if (impl_->ctx != 0) {
    rknn_destroy(impl_->ctx);
    impl_->ctx = 0;
  }
  impl_->input_attrs.clear();
  impl_->output_attrs.clear();
  impl_->vocab.clear();
  impl_->io_num = rknn_input_output_num{};
  impl_->model_channel = 3;
  impl_->model_width = kDefaultInputWidth;
  impl_->model_height = kDefaultInputHeight;
  impl_->initialized = false;
}

bool LicensePlateOcrRecognizer::ready() const {
  return impl_->initialized && impl_->ctx != 0;
}

LicensePlateOcrInputInfo LicensePlateOcrRecognizer::input_info() const {
  LicensePlateOcrInputInfo info;
  if (!ready() || impl_->input_attrs.empty()) {
    return info;
  }
  const rknn_tensor_attr& attr = impl_->input_attrs[0];
  info.width = impl_->model_width;
  info.height = impl_->model_height;
  info.channels = impl_->model_channel;
  info.format = attr.fmt;
  info.type = attr.type;
  info.supports_direct_uint8_mem = attr.type == RKNN_TENSOR_UINT8 || attr.type == RKNN_TENSOR_INT8;
  return info;
}

LicensePlateOcrInputMem LicensePlateOcrRecognizer::create_input_mem() {
  LicensePlateOcrInputMem input_mem;
  if (!ready() || impl_->input_attrs.empty()) {
    return input_mem;
  }

  input_mem.attr = impl_->input_attrs[0];
  const uint32_t size = input_mem.attr.size_with_stride > 0 ? input_mem.attr.size_with_stride : input_mem.attr.size;
  input_mem.mem = rknn_create_mem(impl_->ctx, size);
  return input_mem;
}

void LicensePlateOcrRecognizer::destroy_input_mem(LicensePlateOcrInputMem* input_mem) {
  if (input_mem == nullptr || input_mem->mem == nullptr || impl_->ctx == 0) {
    return;
  }
  rknn_destroy_mem(impl_->ctx, input_mem->mem);
  input_mem->mem = nullptr;
}

LicensePlateOcrResult LicensePlateOcrRecognizer::recognize(const cv::Mat& bgr_image) {
  LicensePlateOcrResult result;
  if (!ready()) {
    result.error = "OCR recognizer is not initialized";
    return result;
  }
  if (bgr_image.empty()) {
    result.error = "empty OCR input image";
    return result;
  }
  if (bgr_image.channels() != 3) {
    result.error = "OCR input image must be BGR with 3 channels";
    return result;
  }

  cv::Mat input_image = resize_with_padding(bgr_image, impl_->model_height, impl_->model_width);

  rknn_input input;
  std::memset(&input, 0, sizeof(input));
  input.index = 0;
  input.type = RKNN_TENSOR_FLOAT32;
  input.fmt = RKNN_TENSOR_NHWC;
  input.size = impl_->model_width * impl_->model_height * impl_->model_channel * sizeof(float);
  input.buf = input_image.data;

  int ret = rknn_inputs_set(impl_->ctx, 1, &input);
  if (ret < 0) {
    result.error = "rknn_inputs_set failed ret=" + std::to_string(ret);
    return result;
  }

  ret = rknn_run(impl_->ctx, nullptr);
  if (ret < 0) {
    result.error = "rknn_run failed ret=" + std::to_string(ret);
    return result;
  }

  rknn_output output;
  std::memset(&output, 0, sizeof(output));
  output.want_float = 1;
  ret = rknn_outputs_get(impl_->ctx, 1, &output, nullptr);
  if (ret < 0) {
    result.error = "rknn_outputs_get failed ret=" + std::to_string(ret);
    return result;
  }

  const rknn_tensor_attr& output_attr = impl_->output_attrs[0];
  const int sequence_length = output_sequence_length(output_attr, impl_->model_width);
  const int class_count = output_class_count(output_attr, sequence_length, static_cast<int>(impl_->vocab.size()));
  result = ctc_decode(static_cast<float*>(output.buf), sequence_length, class_count, impl_->vocab);

  rknn_outputs_release(impl_->ctx, 1, &output);
  return result;
}

LicensePlateOcrResult LicensePlateOcrRecognizer::recognize_preprocessed_mem(LicensePlateOcrInputMem* input_mem) {
  LicensePlateOcrResult result;
  if (!ready()) {
    result.error = "OCR recognizer is not initialized";
    return result;
  }
  if (input_mem == nullptr || input_mem->mem == nullptr) {
    result.error = "empty OCR input memory";
    return result;
  }

  rknn_tensor_attr input_attr = input_mem->attr;
  input_attr.index = 0;
  input_attr.pass_through = 0;
  int ret = rknn_set_io_mem(impl_->ctx, input_mem->mem, &input_attr);
  if (ret < 0) {
    result.error = "rknn_set_io_mem input failed ret=" + std::to_string(ret);
    return result;
  }

  ret = rknn_run(impl_->ctx, nullptr);
  if (ret < 0) {
    result.error = "rknn_run failed ret=" + std::to_string(ret);
    return result;
  }

  rknn_output output;
  std::memset(&output, 0, sizeof(output));
  output.want_float = 1;
  ret = rknn_outputs_get(impl_->ctx, 1, &output, nullptr);
  if (ret < 0) {
    result.error = "rknn_outputs_get failed ret=" + std::to_string(ret);
    return result;
  }

  const rknn_tensor_attr& output_attr = impl_->output_attrs[0];
  const int sequence_length = output_sequence_length(output_attr, impl_->model_width);
  const int class_count = output_class_count(output_attr, sequence_length, static_cast<int>(impl_->vocab.size()));
  result = ctc_decode(static_cast<float*>(output.buf), sequence_length, class_count, impl_->vocab);

  rknn_outputs_release(impl_->ctx, 1, &output);
  return result;
}

}  // namespace rkai