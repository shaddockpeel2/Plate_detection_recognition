#include "camera_source_thread.hpp"
#include "decoder_thread.hpp"
#include "encoder_writer_thread.hpp"
#include "inference_thread.hpp"
#include "oled_display_thread.hpp"
#include "pipeline_types.hpp"
#include "plate_relay.hpp"
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

int parse_named_int_auto_arg(int argc, char** argv, const char* flag, int fallback) {
  const char* value = flag_value(argc, argv, flag, nullptr);
  return value ? static_cast<int>(std::strtol(value, nullptr, 0)) : fallback;
}

float parse_named_float_arg(int argc, char** argv, const char* flag, float fallback) {
  const char* value = flag_value(argc, argv, flag, nullptr);
  return value ? static_cast<float>(std::atof(value)) : fallback;
}

int parse_int_arg(int argc, char** argv, int index, int fallback) {
  return argc > index ? std::atoi(argv[index]) : fallback;
}

float parse_float_arg(int argc, char** argv, int index, float fallback) {
  return argc > index ? static_cast<float>(std::atof(argv[index])) : fallback;
}

bool looks_like_number(const char* value) {
  if (value == nullptr || *value == '\0') {
    return false;
  }
  char* end = nullptr;
  std::strtof(value, &end);
  return end != value && *end == '\0';
}

int parse_int_auto_arg(int argc, char** argv, int index, int fallback) {
  return argc > index ? static_cast<int>(std::strtol(argv[index], nullptr, 0)) : fallback;
}

}  // namespace

int main(int argc, char** argv) {
  /*
   * 启动参数有两种模式：
   *
   * 1. 旧位置参数模式（保持兼容，适合已有脚本）：
   *    rk_mp4_yolo_stage5 <视频> <YOLO模型> <输出MP4> <OCR模型> <OCR字典> [OCR间隔帧] [OCR缓存帧] [OCR最低分] [OCR日志]
   *    例如：
   *    ./build/rk_mp4_yolo_stage5 ./video/test-video/5s.mp4 ./models/car-v8/v8-car-relu-3588.rknn \
   *      ./output.mp4 ./ppocrv5/PP-OCRv5_mobile_rec_license_plate.rknn ./ppocrv5/model/license_plate_dict.txt 15 90
   *
   * 2. 推荐命名参数模式。必须提供 --input mp4 或 --input camera：
   *    输入/推理：
   *      --input mp4|camera     选择视频文件或 USB 摄像头输入。
   *      --video <路径>         MP4 输入文件；摄像头模式下不使用。
   *      --device /dev/video0   摄像头设备；--width、--height、--fps 设置采集格式。
   *      --model <路径>         YOLO RKNN 模型；--output <路径> 设置输出 MP4。
   *    OCR：
   *      --ocr-model <路径>     启用并指定车牌 OCR RKNN 模型。
   *      --ocr-vocab <路径>     OCR 字典文件。
   *      --ocr-interval <帧数>  每隔多少帧重新 OCR 一次，默认 15；越小越及时但消耗更多算力。
   *      --ocr-cache <帧数>     同一目标识别结果的缓存上限，默认 90。
   *      --ocr-min-score <0~1>  OCR 最低置信度，默认 0.80。
   *    白名单继电器（默认关闭）：
   *      --relay-enabled on     显式启用；未提供时绝不访问 /dev/ttyS0。
   *      --relay-whitelist <路径> 每行一个车牌的白名单文件。
   *      --relay-pulse-ms 2000  白名单命中后吸合时间；到期强制关闭。
   *      --relay-plate-cooldown-sec 30 同一车牌允许再次触发前的等待时间。
   *      --relay-min-detect-score 0.50、--relay-min-plate-score 0.90 设置检测/OCR 门槛。
   *      --relay-device /dev/ttyS0、--relay-baud 9600、--relay-address 1、--relay-channel 0 为已验证的硬件参数。
   *    其他可选功能：
   *      --oled on              启用 OLED；--oled-i2c、--oled-addr 设置其 I2C 参数。
   *      --upload-event-log on  启用抓图/上传事件；--mqtt on 及 --mqtt-* 配置 MQTT。
   *
   * 推荐的继电器运行方式见 README 中“车牌白名单继电器”章节。
   */
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
  const bool positional_arg8_is_number = !named_args && argc > 8 && looks_like_number(argv[8]);
  postprocess_config.plate_ocr.min_ocr_score = named_args ? static_cast<float>(std::atof(flag_value(argc, argv, "--ocr-min-score", "0.80")))
                                                          : (positional_arg8_is_number ? parse_float_arg(argc, argv, 8, 0.80f) : 0.80f);
  postprocess_config.plate_ocr.verbose = named_args ? has_flag(argc, argv, "--ocr-verbose")
                                                    : (positional_arg8_is_number && argc > 9 && arg_enabled(argv[9]));
  postprocess_config.upload_event.enabled = arg_enabled(flag_value(argc, argv, "--upload-event-log", "off"));
  postprocess_config.upload_event.verbose = postprocess_config.upload_event.enabled;
  postprocess_config.upload_event.device_id = flag_value(argc, argv, "--device-id", "rk3588-001");
  postprocess_config.upload_event.min_detect_score = parse_named_float_arg(argc, argv, "--upload-min-detect-score", 0.50f);
  postprocess_config.upload_event.min_plate_score = parse_named_float_arg(argc, argv, "--upload-min-plate-score", 0.80f);
  postprocess_config.upload_event.track_cooldown_frames =
      std::max(0, parse_named_int_arg(argc, argv, "--upload-track-cooldown-frames", 90));
  postprocess_config.upload_event.plate_cooldown_frames =
      std::max(0, parse_named_int_arg(argc, argv, "--upload-plate-cooldown-frames", 1800));
  const std::size_t upload_queue_capacity =
      static_cast<std::size_t>(std::max(1, parse_named_int_arg(argc, argv, "--upload-queue-capacity", 32)));
  postprocess_config.plate_relay.enabled = named_args && arg_enabled(flag_value(argc, argv, "--relay-enabled", "off"));
  postprocess_config.plate_relay.verbose = has_flag(argc, argv, "--relay-verbose");
  postprocess_config.plate_relay.whitelist_path =
      flag_value(argc, argv, "--relay-whitelist", "./config/plate_whitelist.txt");
  postprocess_config.plate_relay.min_detect_score = parse_named_float_arg(argc, argv, "--relay-min-detect-score", 0.50f);
  postprocess_config.plate_relay.min_plate_score = parse_named_float_arg(argc, argv, "--relay-min-plate-score", 0.90f);
  postprocess_config.plate_relay.plate_cooldown_sec =
      std::max(0, parse_named_int_arg(argc, argv, "--relay-plate-cooldown-sec", 30));
  rkai::RelayControllerConfig relay_config;
  relay_config.enabled = postprocess_config.plate_relay.enabled;
  relay_config.verbose = postprocess_config.plate_relay.verbose;
  relay_config.device = flag_value(argc, argv, "--relay-device", "/dev/ttyS0");
  relay_config.baud_rate = parse_named_int_arg(argc, argv, "--relay-baud", 9600);
  relay_config.slave_address = parse_named_int_arg(argc, argv, "--relay-address", 1);
  relay_config.coil_address = parse_named_int_arg(argc, argv, "--relay-channel", 0);
  relay_config.pulse_ms = std::max(1, parse_named_int_arg(argc, argv, "--relay-pulse-ms", 2000));
  relay_config.io_timeout_ms = std::max(50, parse_named_int_arg(argc, argv, "--relay-timeout-ms", 500));
  rkai::OledDisplayConfig oled_config;
  const int positional_oled_index = positional_arg8_is_number ? 10 : 8;
  oled_config.enabled = named_args ? arg_enabled(flag_value(argc, argv, "--oled", "off"))
                                   : (argc > positional_oled_index && arg_enabled(argv[positional_oled_index]));
  oled_config.verbose = named_args ? has_flag(argc, argv, "--oled-verbose") : false;
  oled_config.i2c_device = named_args ? flag_value(argc, argv, "--oled-i2c", "/dev/i2c-5")
                                      : (argc > positional_oled_index + 1 ? argv[positional_oled_index + 1] : "/dev/i2c-5");
  oled_config.i2c_address = named_args ? parse_named_int_auto_arg(argc, argv, "--oled-addr", 0x3c)
                                       : parse_int_auto_arg(argc, argv, positional_oled_index + 2, 0x3c);
  oled_config.queue_capacity = static_cast<std::size_t>(std::max(1, parse_named_int_arg(argc, argv, "--oled-queue-capacity", 2)));
  oled_config.min_refresh_interval_ms = std::max(0, parse_named_int_arg(argc, argv, "--oled-refresh-ms", 120));
  postprocess_config.oled_display_enabled = oled_config.enabled;
  rkai::UploadEventThreadConfig upload_thread_config;
  upload_thread_config.enabled = postprocess_config.upload_event.enabled;
  upload_thread_config.verbose = postprocess_config.upload_event.verbose;
  upload_thread_config.snapshot_dir = flag_value(argc, argv, "--upload-snapshot-dir", "./upload_snapshots");
  upload_thread_config.upload_url = flag_value(argc, argv, "--upload-url", "");
  upload_thread_config.public_base_url = flag_value(argc, argv, "--upload-public-base-url", "");
  upload_thread_config.jpeg_quality =
      std::max(1, std::min(100, parse_named_int_arg(argc, argv, "--upload-jpeg-quality", 85)));
  upload_thread_config.http_timeout_ms =
      std::max(1000, parse_named_int_arg(argc, argv, "--upload-http-timeout-ms", 3000));
  upload_thread_config.mqtt_enabled = arg_enabled(flag_value(argc, argv, "--mqtt", "off"));
  upload_thread_config.mqtt_host = flag_value(argc, argv, "--mqtt-host", "");
  upload_thread_config.mqtt_port = parse_named_int_arg(argc, argv, "--mqtt-port", 1883);
  upload_thread_config.mqtt_username = flag_value(argc, argv, "--mqtt-username", "");
  upload_thread_config.mqtt_password = flag_value(argc, argv, "--mqtt-password", "");
  upload_thread_config.mqtt_client_id = flag_value(argc, argv, "--mqtt-client-id", postprocess_config.upload_event.device_id.c_str());
  upload_thread_config.mqtt_topic = flag_value(argc, argv, "--mqtt-topic", "devices/rk3588-001/plate/events");
  upload_thread_config.mqtt_heartbeat_topic =
      flag_value(argc, argv, "--mqtt-heartbeat-topic", "devices/rk3588-001/status/heartbeat");
  upload_thread_config.mqtt_error_topic =
      flag_value(argc, argv, "--mqtt-error-topic", "devices/rk3588-001/status/error");
  upload_thread_config.mqtt_heartbeat_interval_sec =
      std::max(0, parse_named_int_arg(argc, argv, "--mqtt-heartbeat-interval-sec", 30));
  upload_thread_config.mqtt_timeout_ms =
      std::max(1000, parse_named_int_arg(argc, argv, "--mqtt-timeout-ms", 3000));
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
  rkai::UploadEventQueue upload_events(upload_queue_capacity);
  rkai::OledPlateQueue oled_events(oled_config.queue_capacity);
  rkai::RelayEventQueue relay_events(1);

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
  std::thread relay_controller;
  if (relay_config.enabled) {
    relay_controller = std::thread([&] { rkai::relay_controller_thread(relay_config, stop, relay_events); });
  }
  std::thread postprocess([&] {
    rkai::postprocess_osd_thread(postprocess_config,
                                 stop,
                                 inference_results,
                                 osd_frames,
                                 upload_thread_config.enabled ? &upload_events : nullptr,
                                 oled_config.enabled ? &oled_events : nullptr,
                                 relay_config.enabled ? &relay_events : nullptr);
  });
  std::thread writer([&] { rkai::encoder_writer_thread(writer_config, stop, osd_frames); });
  std::thread uploader;
  if (upload_thread_config.enabled) {
    uploader = std::thread([&] { rkai::upload_event_thread(upload_thread_config, stop, upload_events); });
  }
  std::thread oled_display;
  if (oled_config.enabled) {
    oled_display = std::thread([&] { rkai::oled_display_thread(oled_config, stop, oled_events); });
  }

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
  upload_events.close();
  oled_events.close();
  relay_events.close();
  if (uploader.joinable()) {
    uploader.join();
  }
  if (oled_display.joinable()) {
    oled_display.join();
  }
  if (relay_controller.joinable()) {
    relay_controller.join();
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