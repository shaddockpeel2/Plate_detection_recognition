#include "rknn_input_runtime.hpp"

#include <cstdio>
#include <fstream>

namespace rkai {
namespace {

std::vector<uint8_t> read_binary_file(const std::string& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    return {};
  }
  const auto size = input.tellg();
  std::vector<uint8_t> data(static_cast<size_t>(size));
  input.seekg(0, std::ios::beg);
  input.read(reinterpret_cast<char*>(data.data()), size);
  return data;
}

void dump_tensor_attr(const char* title, const rknn_tensor_attr& attr) {
  std::fprintf(stderr,
               "%s index=%u name=%s dims=[%u,%u,%u,%u] n_dims=%u size=%u size_with_stride=%u fmt=%d type=%d qnt=%d zp=%d scale=%f\n",
               title,
               attr.index,
               attr.name,
               attr.dims[0],
               attr.dims[1],
               attr.dims[2],
               attr.dims[3],
               attr.n_dims,
               attr.size,
               attr.size_with_stride,
               attr.fmt,
               attr.type,
               attr.qnt_type,
               attr.zp,
               attr.scale);
}

ModelInputInfo parse_model_input(const rknn_tensor_attr& attr) {
  ModelInputInfo info;
  info.layout = static_cast<rknn_tensor_format>(attr.fmt);
  info.type = static_cast<rknn_tensor_type>(attr.type);
  info.quantized = attr.qnt_type != RKNN_TENSOR_QNT_NONE;

  if (attr.fmt == RKNN_TENSOR_NCHW) {
    info.channels = static_cast<int>(attr.dims[1]);
    info.height = static_cast<int>(attr.dims[2]);
    info.width = static_cast<int>(attr.dims[3]);
  } else {
    info.height = static_cast<int>(attr.dims[1]);
    info.width = static_cast<int>(attr.dims[2]);
    info.channels = static_cast<int>(attr.dims[3]);
  }

  info.rgb = true;
  return info;
}

}  // namespace

RknnInputRuntime::~RknnInputRuntime() {
  for (auto* mem : input_pool_) {
    if (ctx_ && mem) {
      rknn_destroy_mem(ctx_, mem);
    }
  }
  input_pool_.clear();
  while (!free_inputs_.empty()) {
    free_inputs_.pop();
  }
  if (ctx_) {
    rknn_destroy(ctx_);
    ctx_ = 0;
  }
}

bool RknnInputRuntime::init(const RknnTensorRuntimeConfig& config) {
  auto model = read_binary_file(config.model_path);
  if (model.empty()) {
    std::fprintf(stderr, "failed to read rknn model: %s\n", config.model_path.c_str());
    return false;
  }

  int ret = rknn_init(&ctx_, model.data(), static_cast<uint32_t>(model.size()), 0, nullptr);
  if (ret < 0) {
    std::fprintf(stderr, "rknn_init failed ret=%d\n", ret);
    return false;
  }

  ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
  if (ret < 0) {
    std::fprintf(stderr, "rknn_query RKNN_QUERY_IN_OUT_NUM failed ret=%d\n", ret);
    return false;
  }
  if (io_num_.n_input < 1) {
    std::fprintf(stderr, "rknn model has no input tensor\n");
    return false;
  }

  input_attr_.index = 0;
  ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &input_attr_, sizeof(input_attr_));
  if (ret < 0) {
    std::fprintf(stderr, "rknn_query RKNN_QUERY_INPUT_ATTR failed ret=%d\n", ret);
    return false;
  }
  dump_tensor_attr("rknn input", input_attr_);

  input_info_ = parse_model_input(input_attr_);
  if (input_info_.channels != 3 || input_info_.width <= 0 || input_info_.height <= 0) {
    std::fprintf(stderr,
                 "unsupported input shape: width=%d height=%d channels=%d\n",
                 input_info_.width,
                 input_info_.height,
                 input_info_.channels);
    return false;
  }

  output_attrs_.resize(io_num_.n_output);
  for (uint32_t i = 0; i < io_num_.n_output; ++i) {
    output_attrs_[i] = {};
    output_attrs_[i].index = i;
    ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_attrs_[i], sizeof(output_attrs_[i]));
    if (ret < 0) {
      std::fprintf(stderr, "rknn_query RKNN_QUERY_OUTPUT_ATTR failed index=%u ret=%d\n", i, ret);
      return false;
    }
    dump_tensor_attr("rknn output", output_attrs_[i]);
  }

  const uint32_t input_size = input_attr_.size_with_stride > 0 ? input_attr_.size_with_stride : input_attr_.size;
  const int pool_size = config.input_pool_size > 0 ? config.input_pool_size : 4;
  for (int i = 0; i < pool_size; ++i) {
    rknn_tensor_mem* mem = rknn_create_mem(ctx_, input_size);
    if (!mem) {
      std::fprintf(stderr, "rknn_create_mem failed at slot=%d size=%u\n", i, input_size);
      return false;
    }
    input_pool_.push_back(mem);
    free_inputs_.push(mem);
  }

  std::fprintf(stderr,
               "rknn input pool ready: model=%dx%dx%d pool=%d tensor_size=%u\n",
               input_info_.width,
               input_info_.height,
               input_info_.channels,
               pool_size,
               input_size);
  return true;
}

RknnInputTensor RknnInputRuntime::acquire(StopFlag& stop) {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [&] { return stop.stop_requested() || !free_inputs_.empty(); });
  if (free_inputs_.empty()) {
    return {};
  }

  rknn_tensor_mem* mem = free_inputs_.front();
  free_inputs_.pop();

  RknnInputTensor tensor;
  tensor.mem = mem;
  tensor.dma_fd = mem->fd;
  tensor.virt_addr = mem->virt_addr;
  tensor.size = input_attr_.size_with_stride > 0 ? input_attr_.size_with_stride : input_attr_.size;
  tensor.width = input_info_.width;
  tensor.height = input_info_.height;
  tensor.channels = input_info_.channels;
  return tensor;
}

bool RknnInputRuntime::bind_input_mem(rknn_tensor_mem* mem) {
  if (!ctx_ || !mem) {
    return false;
  }

  rknn_tensor_attr attr = input_attr_;
  attr.index = 0;
  attr.fmt = RKNN_TENSOR_NHWC;
  attr.type = input_attr_.type;
  attr.pass_through = 0;
  attr.size = input_attr_.size;
  attr.size_with_stride = input_attr_.size_with_stride > 0 ? input_attr_.size_with_stride : input_attr_.size;
  attr.w_stride = static_cast<uint32_t>(input_info_.width);
  attr.h_stride = static_cast<uint32_t>(input_info_.height);

  const int ret = rknn_set_io_mem(ctx_, mem, &attr);
  if (ret < 0) {
    std::fprintf(stderr, "rknn_set_io_mem input failed ret=%d fd=%d\n", ret, mem->fd);
    return false;
  }
  return true;
}

void RknnInputRuntime::release(rknn_tensor_mem* mem) {
  if (!mem) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    free_inputs_.push(mem);
  }
  cv_.notify_one();
}

}  // namespace rkai