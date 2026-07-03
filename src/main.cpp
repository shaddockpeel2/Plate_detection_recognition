#include "decoder_thread.hpp"
#include "encoder_writer_thread.hpp"
#include "inference_thread.hpp"
#include "pipeline_types.hpp"
#include "postprocess_osd_thread.hpp"
#include "preprocess_thread.hpp"
#include "rknn_input_runtime.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

int main(int argc, char** argv) {
  const char* default_video = "/home/cat/mpp-main/yolo26videeotest/main2/video/output_5s.mp4";
  const char* default_model = "/home/cat/mpp-main/yolo26videeotest/main2/car-v8/v8-car-relu-3588.rknn";
  const char* default_output = "/home/cat/mpp-main/yolo26videeotest/main2/output_stage5.mp4";

  rkai::DecoderConfig decoder_config;
  decoder_config.input_path = argc > 1 ? argv[1] : default_video;
  decoder_config.output_queue_capacity = 8;
  decoder_config.verbose = false;

  rkai::RknnTensorRuntimeConfig rknn_config;
  rknn_config.model_path = argc > 2 ? argv[2] : default_model;
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
  postprocess_config.verbose = false;
  postprocess_config.output_queue_capacity = 8;

  rkai::EncoderWriterConfig writer_config;
  writer_config.output_path = argc > 3 ? argv[3] : default_output;
  writer_config.codec = rkai::EncodedVideoCodec::H264;
  writer_config.fps_num = 30;
  writer_config.fps_den = 1;
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

  rkai::DecodedFrameQueue decoded_frames(decoder_config.output_queue_capacity);
  rkai::PreprocessedFrameQueue preprocessed_frames(preprocess_config.output_queue_capacity);
  rkai::InferenceResultQueue inference_results(inference_config.output_queue_capacity);
  rkai::OsdFrameQueue osd_frames(postprocess_config.output_queue_capacity);

  const auto pipeline_start = std::chrono::steady_clock::now();
  std::thread decoder([&] { rkai::decoder_thread(decoder_config, stop, decoded_frames); });
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

  if (decoder.joinable()) {
    decoder.join();
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