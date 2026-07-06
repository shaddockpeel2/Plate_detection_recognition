#include "camera_source_thread.hpp"
#include "decoder_thread.hpp"
#include "encoder_writer_thread.hpp"
#include "inference_thread.hpp"
#include "pipeline_types.hpp"
#include "postprocess_osd_thread.hpp"
#include "preprocess_thread.hpp"
#include "rknn_input_runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace {

bool arg_enabled(const char* value) {
  return value != nullptr && std::strcmp(value, "off") != 0 && std::strcmp(value, "0") != 0 &&
         std::strcmp(value, "false") != 0;
}

bool is_flag(const char* value, const char* flag) {
  return value != nullptr && std::strcmp(value, flag) == 0;
}

const char* flag_value(int argc, char** argv, const char* flag, const char* fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (is_flag(argv[i], flag)) {
      return argv[i + 1];
    }
  }
  return fallback;
}

bool has_flag(int argc, char** argv, const char* flag) {
  for (int i = 1; i < argc; ++i) {
    if (is_flag(argv[i], flag)) {
      return true;
    }
  }
  return false;
}

int parse_named_int_arg(int argc, char** argv, const char* flag, int fallback) {
  const char* value = flag_value(argc, argv, flag, nullptr);
  return value ? std::atoi(value) : fallback;
}

int parse_int_arg(int argc, char** argv, int index, int fallback) {
  return argc > index ? std::atoi(argv[index]) : fallback;
}

float parse_float_arg(int argc, char** argv, int index, float fallback) {
  return argc > index ? static_cast<float>(std::atof(argv[index])) : fallback;
}

}  // namespace

int main(int argc, char** argv) {
  const char* default_video = "/home/cat/mpp-main/yolo26videeotest/main2/video/output_5s.mp4";
  const char* default_model = "/home/cat/mpp-main/yolo26videeotest/main2/car-v8/v8-car-relu-3588.rknn";
  const char* default_output = "/home/cat/mpp-main/yolo26videeotest/main2/output_stage5.mp4";
  const char* default_ocr_model = "/home/cat/mpp-main/yolo26videeotest/main2/ppocrv5/PP-OCRv5_mobile_rec_license_plate.rknn";
  const char* default_ocr_vocab = "/home/cat/mpp-main/yolo26videeotest/main2/ppocrv5/model/license_plate_dict.txt";

  const bool named_args = has_flag(argc, argv, "--input");
  const char* input_mode = flag_value(argc, argv, "--input", "mp4");
  const bool camera_input = std::strcmp(input_mode, "camera") == 0;

  rkai::DecoderConfig decoder_config;
  decoder_config.input_path = named_args ? flag_value(argc, argv, "--video", default_video) : (argc > 1 ? argv[1] : default_video);
  decoder_config.output_queue_capacity = 8;
  decoder_config.verbose = false;

  rkai::CameraSourceConfig camera_config;
  camera_config.device = flag_value(argc, argv, "--device", "/dev/video0");
  camera_config.width = parse_named_int_arg(argc, argv, "--width", 640);
  camera_config.height = parse_named_int_arg(argc, argv, "--height", 480);
  camera_config.fps_num = parse_named_int_arg(argc, argv, "--fps", 25);
  camera_config.fps_den = 1;
  camera_config.frame_limit = parse_named_int_arg(argc, argv, "--frame-limit", 300);
  camera_config.output_queue_capacity = 8;
  camera_config.verbose = has_flag(argc, argv, "--camera-verbose");

  rkai::RknnTensorRuntimeConfig rknn_config;
  rknn_config.model_path = named_args ? flag_value(argc, argv, "--model", default_model) : (argc > 2 ? argv[2] : default_model);
  rknn_config.input_pool_size = 4;

  rkai::PreprocessConfig preprocess_config;
  preprocess_config.letterbox = true;
  preprocess_config.model_rgb = true;
  preprocess_config.verbose = false;
  preprocess_config.output_queue_capacity = 4;

  rkai::InferenceConfig inference_config;
  inference_config.want_float_output = true;
  inference_config.verbose = false;
  inference_config.output_queue_capacity = 4;

  rkai::PostprocessOsdConfig postprocess_config;
  postprocess_config.model = rkai::PostprocessModel::Auto;
  postprocess_config.box_threshold = 0.25f;
  postprocess_config.nms_threshold = 0.45f;
  postprocess_config.class_count = 1;
  postprocess_config.model_width = 640;
  postprocess_config.model_height = 640;
  postprocess_config.draw_osd = true;
  postprocess_config.plate_ocr.enabled = named_args ? arg_enabled(flag_value(argc, argv, "--ocr-model", "off")) : (argc > 4 && arg_enabled(argv[4]));
  postprocess_config.plate_ocr.recognizer.model_path = named_args ? flag_value(argc, argv, "--ocr-model", default_ocr_model)
                                                            : (argc > 4 && arg_enabled(argv[4]) ? argv[4] : default_ocr_model);
  postprocess_config.plate_ocr.recognizer.vocab_path = named_args ? flag_value(argc, argv, "--ocr-vocab", default_ocr_vocab)
                                                            : (argc > 5 ? argv[5] : default_ocr_vocab);
  postprocess_config.plate_ocr.recognizer.verbose = false;
  postprocess_config.plate_ocr.ocr_interval_frames = std::max(1, named_args ? parse_named_int_arg(argc, argv, "--ocr-interval", 15) : parse_int_arg(argc, argv, 6, 15));
  postprocess_config.plate_ocr.max_cache_age_frames = std::max(1, named_args ? parse_named_int_arg(argc, argv, "--ocr-cache", 90) : parse_int_arg(argc, argv, 7, 90));
  postprocess_config.plate_ocr.min_ocr_score = named_args ? static_cast<float>(std::atof(flag_value(argc, argv, "--ocr-min-score", "0.80")))
                                                          : parse_float_arg(argc, argv, 8, 0.80f);
  postprocess_config.plate_ocr.verbose = named_args ? has_flag(argc, argv, "--ocr-verbose") : (argc > 9 && arg_enabled(argv[9]));
  postprocess_config.verbose = false;
  postprocess_config.output_queue_capacity = 8;

  rkai::EncoderWriterConfig writer_config;
  writer_config.output_path = named_args ? flag_value(argc, argv, "--output", default_output) : (argc > 3 ? argv[3] : default_output);
  writer_config.codec = rkai::EncodedVideoCodec::H264;
  writer_config.fps_num = camera_input ? camera_config.fps_num : 30;
  writer_config.fps_den = camera_input ? camera_config.fps_den : 1;
  writer_config.bitrate = 4 * 1000 * 1000;
  writer_config.gop = 60;
  writer_config.verbose = false;

  rkai::StopFlag stop;
  rkai::RknnInputRuntime rknn_inputs;
  if (!rknn_inputs.init(rknn_config)) {
    return EXIT_FAILURE;
  }
  postprocess_config.model_width = rknn_inputs.input_info().width;
  postprocess_config.model_height = rknn_inputs.input_info().height;

  const std::size_t source_queue_capacity = camera_input ? camera_config.output_queue_capacity : decoder_config.output_queue_capacity;
  rkai::DecodedFrameQueue decoded_frames(source_queue_capacity);
  rkai::PreprocessedFrameQueue preprocessed_frames(preprocess_config.output_queue_capacity);
  rkai::InferenceResultQueue inference_results(inference_config.output_queue_capacity);
  rkai::OsdFrameQueue osd_frames(postprocess_config.output_queue_capacity);

  const auto pipeline_start = std::chrono::steady_clock::now();
  std::thread source([&] {
    if (camera_input) {
      rkai::camera_source_thread(camera_config, stop, decoded_frames);
    } else {
      rkai::decoder_thread(decoder_config, stop, decoded_frames);
    }
  });
  std::thread preprocessor([&] {
    rkai::preprocess_thread(preprocess_config, stop, decoded_frames, preprocessed_frames, rknn_inputs);
  });
  std::thread inference([&] {
    rkai::inference_thread(inference_config, stop, preprocessed_frames, inference_results, rknn_inputs);
  });
  std::thread postprocess([&] {
    rkai::postprocess_osd_thread(postprocess_config, stop, inference_results, osd_frames);
  });
  std::thread writer([&] { rkai::encoder_writer_thread(writer_config, stop, osd_frames); });

  if (source.joinable()) {
    source.join();
  }
  if (preprocessor.joinable()) {
    preprocessor.join();
  }
  if (inference.joinable()) {
    inference.join();
  }
  if (postprocess.joinable()) {
    postprocess.join();
  }
  if (writer.joinable()) {
    writer.join();
  }

  const auto pipeline_end = std::chrono::steady_clock::now();
  const double pipeline_ms = std::chrono::duration<double, std::milli>(pipeline_end - pipeline_start).count();
  std::fprintf(stderr,
               "[PERF] pipeline elapsed_ms=%.3f output=%s\n",
               pipeline_ms,
               writer_config.output_path.c_str());
  std::fprintf(stderr, "stage5 test finished output=%s\n", writer_config.output_path.c_str());
  return stop.stop_requested() ? EXIT_FAILURE : EXIT_SUCCESS;
}