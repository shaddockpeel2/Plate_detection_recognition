#include "encoder_writer_thread.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <rockchip/mpp_buffer.h>
#include <rockchip/mpp_frame.h>
#include <rockchip/mpp_packet.h>
#include <rockchip/rk_mpi.h>
}

namespace rkai {
namespace {

MppCodingType to_mpp_codec(EncodedVideoCodec codec) {
  return codec == EncodedVideoCodec::H265 ? MPP_VIDEO_CodingHEVC : MPP_VIDEO_CodingAVC;
}

AVCodecID to_av_codec(EncodedVideoCodec codec) {
  return codec == EncodedVideoCodec::H265 ? AV_CODEC_ID_HEVC : AV_CODEC_ID_H264;
}

const char* raw_extension(EncodedVideoCodec codec) {
  return codec == EncodedVideoCodec::H265 ? ".h265" : ".h264";
}

std::string fallback_raw_path(const std::string& output_path, EncodedVideoCodec codec) {
  const std::size_t slash = output_path.find_last_of('/');
  const std::size_t dot = output_path.find_last_of('.');
  if (dot != std::string::npos && (slash == std::string::npos || dot > slash)) {
    return output_path.substr(0, dot) + raw_extension(codec);
  }
  return output_path + raw_extension(codec);
}

int safe_fps_num(const EncoderWriterConfig& config) {
  return config.fps_num > 0 ? config.fps_num : 30;
}

int safe_fps_den(const EncoderWriterConfig& config) {
  return config.fps_den > 0 ? config.fps_den : 1;
}

int safe_gop(const EncoderWriterConfig& config) {
  if (config.gop > 0) {
    return config.gop;
  }
  return std::max(1, safe_fps_num(config) * 2 / safe_fps_den(config));
}

bool looks_like_mp4_path(const std::string& path) {
  const std::size_t dot = path.find_last_of('.');
  if (dot == std::string::npos) {
    return false;
  }
  const std::string ext = path.substr(dot);
  return ext == ".mp4" || ext == ".mov" || ext == ".m4v";
}

class RawAnnexBSink final : public IEncodedSink {
 public:
  bool open(const EncoderWriterConfig& config,
            int,
            int,
            const uint8_t* header,
            int header_size) override {
    close();
    path_ = looks_like_mp4_path(config.output_path) ? fallback_raw_path(config.output_path, config.codec) : config.output_path;
    file_.open(path_, std::ios::binary | std::ios::trunc);
    if (!file_) {
      std::fprintf(stderr, "open raw stream failed: %s\n", path_.c_str());
      return false;
    }
    if (header && header_size > 0) {
      file_.write(reinterpret_cast<const char*>(header), header_size);
    }
    std::fprintf(stderr, "raw Annex-B sink opened: %s\n", path_.c_str());
    return true;
  }

  bool write(const EncodedPacketView& packet) override {
    if (!file_ || !packet.data || packet.size <= 0) {
      return false;
    }
    file_.write(reinterpret_cast<const char*>(packet.data), packet.size);
    return static_cast<bool>(file_);
  }

  void close() override {
    if (file_) {
      file_.flush();
      file_.close();
    }
  }

  ~RawAnnexBSink() override { close(); }

 private:
  std::string path_;
  std::ofstream file_;
};

class Mp4MuxerSink final : public IEncodedSink {
 public:
  bool open(const EncoderWriterConfig& config,
            int width,
            int height,
            const uint8_t* header,
            int header_size) override {
    close();
    config_ = config;
    input_time_base_ = AVRational{safe_fps_den(config), safe_fps_num(config)};

    int ret = avformat_alloc_output_context2(&fmt_, nullptr, "mp4", config.output_path.c_str());
    if (ret < 0 || !fmt_) {
      log_av_error("avformat_alloc_output_context2", ret);
      return false;
    }

    stream_ = avformat_new_stream(fmt_, nullptr);
    if (!stream_) {
      std::fprintf(stderr, "avformat_new_stream failed\n");
      close();
      return false;
    }

    stream_->id = static_cast<int>(fmt_->nb_streams - 1);
    stream_->time_base = input_time_base_;
    stream_->avg_frame_rate = AVRational{safe_fps_num(config), safe_fps_den(config)};

    AVCodecParameters* par = stream_->codecpar;
    par->codec_type = AVMEDIA_TYPE_VIDEO;
    par->codec_id = to_av_codec(config.codec);
    par->width = width;
    par->height = height;
    par->format = AV_PIX_FMT_NV12;
    par->bit_rate = config.bitrate;

    if (header && header_size > 0) {
      par->extradata = static_cast<uint8_t*>(av_mallocz(header_size + AV_INPUT_BUFFER_PADDING_SIZE));
      if (!par->extradata) {
        std::fprintf(stderr, "alloc codec extradata failed\n");
        close();
        return false;
      }
      std::memcpy(par->extradata, header, header_size);
      par->extradata_size = header_size;
    }

    if (!(fmt_->oformat->flags & AVFMT_NOFILE)) {
      ret = avio_open(&fmt_->pb, config.output_path.c_str(), AVIO_FLAG_WRITE);
      if (ret < 0) {
        log_av_error("avio_open", ret);
        close();
        return false;
      }
      owns_io_ = true;
    }

    ret = avformat_write_header(fmt_, nullptr);
    if (ret < 0) {
      log_av_error("avformat_write_header", ret);
      close();
      return false;
    }

    opened_ = true;
    std::fprintf(stderr, "mp4 muxer opened: %s\n", config.output_path.c_str());
    return true;
  }

  bool write(const EncodedPacketView& packet) override {
    if (!opened_ || !packet.data || packet.size <= 0) {
      return false;
    }

    AVPacket pkt;
    av_init_packet(&pkt);
    pkt.data = const_cast<uint8_t*>(packet.data);
    pkt.size = packet.size;
    pkt.stream_index = stream_->index;
    pkt.pts = av_rescale_q(packet.frame_index, input_time_base_, stream_->time_base);
    pkt.dts = pkt.pts;
    pkt.duration = std::max<int64_t>(1, av_rescale_q(1, input_time_base_, stream_->time_base));
    pkt.pos = -1;
    if (packet.keyframe) {
      pkt.flags |= AV_PKT_FLAG_KEY;
    }

    const int ret = av_interleaved_write_frame(fmt_, &pkt);
    if (ret < 0) {
      log_av_error("av_interleaved_write_frame", ret);
      return false;
    }
    return true;
  }

  void close() override {
    if (opened_ && fmt_) {
      av_write_trailer(fmt_);
    }
    if (fmt_ && owns_io_) {
      avio_closep(&fmt_->pb);
    }
    if (fmt_) {
      avformat_free_context(fmt_);
    }
    fmt_ = nullptr;
    stream_ = nullptr;
    owns_io_ = false;
    opened_ = false;
  }

  ~Mp4MuxerSink() override { close(); }

 private:
  static void log_av_error(const char* where, int ret) {
    char err[AV_ERROR_MAX_STRING_SIZE] = {0};
    av_strerror(ret, err, sizeof(err));
    std::fprintf(stderr, "%s failed: %s\n", where, err);
  }

  EncoderWriterConfig config_;
  AVFormatContext* fmt_ = nullptr;
  AVStream* stream_ = nullptr;
  AVRational input_time_base_{1, 30};
  bool owns_io_ = false;
  bool opened_ = false;
};

class ZlmPushSink final : public IEncodedSink {
 public:
  bool open(const EncoderWriterConfig&, int, int, const uint8_t*, int) override {
    std::fprintf(stderr, "ZLM sink is reserved but not enabled in this build\n");
    return false;
  }
  bool write(const EncodedPacketView&) override { return false; }
  void close() override {}
};

class MppEncoderRuntime {
 public:
  bool init(const EncoderWriterConfig& config, const DecodedFrame& first) {
    config_ = config;
    width_ = first.width;
    height_ = first.height;
    hor_stride_ = first.hor_stride;
    ver_stride_ = first.ver_stride;
    codec_ = to_mpp_codec(config.codec);

    MPP_RET ret = mpp_create(&ctx_, &mpi_);
    if (ret != MPP_OK) {
      std::fprintf(stderr, "mpp_create encoder failed ret=%d\n", ret);
      return false;
    }

    MppPollType timeout = MPP_POLL_BLOCK;
    mpi_->control(ctx_, MPP_SET_INPUT_TIMEOUT, &timeout);
    mpi_->control(ctx_, MPP_SET_OUTPUT_TIMEOUT, &timeout);

    ret = mpp_init(ctx_, MPP_CTX_ENC, codec_);
    if (ret != MPP_OK) {
      std::fprintf(stderr, "mpp_init encoder failed ret=%d\n", ret);
      return false;
    }

    ret = mpp_enc_cfg_init(&cfg_);
    if (ret != MPP_OK || !cfg_) {
      std::fprintf(stderr, "mpp_enc_cfg_init failed ret=%d\n", ret);
      return false;
    }

    const int fps_num = safe_fps_num(config);
    const int fps_den = safe_fps_den(config);
    const int bitrate = config.bitrate > 0 ? config.bitrate : width_ * height_ * 4;
    const int gop = safe_gop(config);

    mpp_enc_cfg_set_s32(cfg_, "prep:width", width_);
    mpp_enc_cfg_set_s32(cfg_, "prep:height", height_);
    mpp_enc_cfg_set_s32(cfg_, "prep:hor_stride", hor_stride_);
    mpp_enc_cfg_set_s32(cfg_, "prep:ver_stride", ver_stride_);
    mpp_enc_cfg_set_s32(cfg_, "prep:format", MPP_FMT_YUV420SP);
    mpp_enc_cfg_set_s32(cfg_, "rc:mode", MPP_ENC_RC_MODE_CBR);
    mpp_enc_cfg_set_s32(cfg_, "rc:bps_target", bitrate);
    mpp_enc_cfg_set_s32(cfg_, "rc:bps_max", bitrate * 17 / 16);
    mpp_enc_cfg_set_s32(cfg_, "rc:bps_min", bitrate * 15 / 16);
    mpp_enc_cfg_set_s32(cfg_, "rc:fps_in_flex", 0);
    mpp_enc_cfg_set_s32(cfg_, "rc:fps_in_num", fps_num);
    mpp_enc_cfg_set_s32(cfg_, "rc:fps_in_denorm", fps_den);
    mpp_enc_cfg_set_s32(cfg_, "rc:fps_in_denom", fps_den);
    mpp_enc_cfg_set_s32(cfg_, "rc:fps_out_flex", 0);
    mpp_enc_cfg_set_s32(cfg_, "rc:fps_out_num", fps_num);
    mpp_enc_cfg_set_s32(cfg_, "rc:fps_out_denorm", fps_den);
    mpp_enc_cfg_set_s32(cfg_, "rc:fps_out_denom", fps_den);
    mpp_enc_cfg_set_s32(cfg_, "rc:gop", gop);
    mpp_enc_cfg_set_s32(cfg_, "codec:type", codec_);

    if (codec_ == MPP_VIDEO_CodingAVC) {
      mpp_enc_cfg_set_s32(cfg_, "h264:profile", 100);
      mpp_enc_cfg_set_s32(cfg_, "h264:level", 40);
      mpp_enc_cfg_set_s32(cfg_, "h264:cabac_en", 1);
      mpp_enc_cfg_set_s32(cfg_, "h264:cabac_idc", 0);
      mpp_enc_cfg_set_s32(cfg_, "h264:trans8x8", 1);
    }

    ret = mpi_->control(ctx_, MPP_ENC_SET_CFG, cfg_);
    if (ret != MPP_OK) {
      std::fprintf(stderr, "MPP_ENC_SET_CFG failed ret=%d\n", ret);
      return false;
    }

    MppEncHeaderMode header_mode = MPP_ENC_HEADER_MODE_EACH_IDR;
    ret = mpi_->control(ctx_, MPP_ENC_SET_HEADER_MODE, &header_mode);
    if (ret != MPP_OK) {
      std::fprintf(stderr, "MPP_ENC_SET_HEADER_MODE failed ret=%d\n", ret);
      return false;
    }

    std::fprintf(stderr,
                 "encoder ready: %dx%d stride=%dx%d codec=%s fps=%d/%d bps=%d gop=%d\n",
                 width_,
                 height_,
                 hor_stride_,
                 ver_stride_,
                 codec_ == MPP_VIDEO_CodingHEVC ? "h265" : "h264",
                 fps_num,
                 fps_den,
                 bitrate,
                 gop);
    return true;
  }

  std::vector<uint8_t> header() {
    std::vector<uint8_t> data(1024 * 1024);
    MppPacket packet = nullptr;
    if (mpp_packet_init(&packet, data.data(), data.size()) != MPP_OK || !packet) {
      return {};
    }

    const MPP_RET ret = mpi_->control(ctx_, MPP_ENC_GET_HDR_SYNC, packet);
    if (ret == MPP_OK) {
      void* ptr = mpp_packet_get_pos(packet);
      const size_t len = mpp_packet_get_length(packet);
      if (ptr && len > 0) {
        const auto* bytes = static_cast<const uint8_t*>(ptr);
        data.assign(bytes, bytes + len);
      } else {
        data.clear();
      }
    } else {
      std::fprintf(stderr, "MPP_ENC_GET_HDR_SYNC failed ret=%d\n", ret);
      data.clear();
    }
    mpp_packet_deinit(&packet);
    return data;
  }

  bool encode(const OsdFramePtr& osd_frame, int64_t frame_index, std::vector<uint8_t>& out, bool& keyframe) {
    out.clear();
    keyframe = false;
    const DecodedFramePtr source = osd_frame->inference->frame->source;
    MppBuffer buffer = mpp_frame_get_buffer(source->frame);
    if (!buffer) {
      std::fprintf(stderr, "encoder input frame has no buffer frame_id=%ld\n", source->frame_id);
      return false;
    }

    MppFrame frame = nullptr;
    MPP_RET ret = mpp_frame_init(&frame);
    if (ret != MPP_OK || !frame) {
      std::fprintf(stderr, "mpp_frame_init encoder input failed ret=%d\n", ret);
      return false;
    }

    mpp_frame_set_width(frame, source->width);
    mpp_frame_set_height(frame, source->height);
    mpp_frame_set_hor_stride(frame, source->hor_stride);
    mpp_frame_set_ver_stride(frame, source->ver_stride);
    mpp_frame_set_fmt(frame, MPP_FMT_YUV420SP);
    mpp_frame_set_buffer(frame, buffer);
    mpp_frame_set_pts(frame, frame_index * 1000000LL * safe_fps_den(config_) / safe_fps_num(config_));

    if (frame_index == 0) {
      RK_U32 force_idr = 1;
      mpi_->control(ctx_, MPP_ENC_SET_IDR_FRAME, &force_idr);
    }

    ret = mpi_->encode_put_frame(ctx_, frame);
    mpp_frame_deinit(&frame);
    if (ret != MPP_OK) {
      std::fprintf(stderr, "encode_put_frame failed ret=%d frame_id=%ld\n", ret, source->frame_id);
      return false;
    }

    while (true) {
      MppPacket packet = nullptr;
      ret = mpi_->encode_get_packet(ctx_, &packet);
      if (ret != MPP_OK) {
        std::fprintf(stderr, "encode_get_packet failed ret=%d frame_id=%ld\n", ret, source->frame_id);
        return false;
      }
      if (!packet) {
        continue;
      }

      void* ptr = mpp_packet_get_pos(packet);
      const size_t len = mpp_packet_get_length(packet);
      if (ptr && len > 0) {
        const auto* bytes = static_cast<const uint8_t*>(ptr);
        out.insert(out.end(), bytes, bytes + len);
      }
      keyframe = keyframe || frame_index == 0 || (safe_gop(config_) > 0 && frame_index % safe_gop(config_) == 0);
      const bool eoi = mpp_packet_is_eoi(packet) != 0;
      mpp_packet_deinit(&packet);
      if (eoi || !out.empty()) {
        break;
      }
    }

    return !out.empty();
  }

  ~MppEncoderRuntime() {
    if (cfg_) {
      mpp_enc_cfg_deinit(cfg_);
    }
    if (ctx_) {
      mpp_destroy(ctx_);
    }
  }

 private:
  EncoderWriterConfig config_;
  MppCtx ctx_ = nullptr;
  MppApi* mpi_ = nullptr;
  MppEncCfg cfg_ = nullptr;
  MppCodingType codec_ = MPP_VIDEO_CodingAVC;
  int width_ = 0;
  int height_ = 0;
  int hor_stride_ = 0;
  int ver_stride_ = 0;
};

std::unique_ptr<IEncodedSink> make_sink(const EncoderWriterConfig& config) {
  if (looks_like_mp4_path(config.output_path)) {
    return std::unique_ptr<IEncodedSink>(new Mp4MuxerSink());
  }
  return std::unique_ptr<IEncodedSink>(new RawAnnexBSink());
}

bool open_sink_with_fallback(std::unique_ptr<IEncodedSink>& sink,
                             const EncoderWriterConfig& config,
                             int width,
                             int height,
                             const std::vector<uint8_t>& header) {
  sink = make_sink(config);
  if (sink->open(config, width, height, header.empty() ? nullptr : header.data(), static_cast<int>(header.size()))) {
    return true;
  }

  if (!looks_like_mp4_path(config.output_path)) {
    return false;
  }

  std::fprintf(stderr, "mp4 sink open failed, fallback to Annex-B raw stream\n");
  sink.reset(new RawAnnexBSink());
  return sink->open(config, width, height, header.empty() ? nullptr : header.data(), static_cast<int>(header.size()));
}

}  // namespace

void encoder_writer_thread(const EncoderWriterConfig& config, StopFlag& stop, OsdFrameQueue& input) {
  const auto stage_start = std::chrono::steady_clock::now();
  MppEncoderRuntime encoder;
  std::unique_ptr<IEncodedSink> sink;
  bool ready = false;
  int64_t encoded = 0;
  int64_t total_detections = 0;
  int64_t total_bytes = 0;
  double encode_total_ms = 0.0;
  double sink_total_ms = 0.0;

  OsdFramePtr osd_frame;
  while (!stop.stop_requested() && input.pop(osd_frame)) {
    if (!osd_frame || !osd_frame->inference || !osd_frame->inference->frame || !osd_frame->inference->frame->source) {
      std::fprintf(stderr, "encoder writer received invalid osd frame\n");
      stop.request_stop();
      break;
    }

    const DecodedFramePtr source = osd_frame->inference->frame->source;
    if (!ready) {
      if (!encoder.init(config, *source)) {
        stop.request_stop();
        break;
      }
      const std::vector<uint8_t> header = encoder.header();
      if (!open_sink_with_fallback(sink, config, source->width, source->height, header)) {
        stop.request_stop();
        break;
      }
      ready = true;
    }

    std::vector<uint8_t> packet;
    bool keyframe = false;
    const auto encode_start = std::chrono::steady_clock::now();
    if (!encoder.encode(osd_frame, encoded, packet, keyframe)) {
      stop.request_stop();
      break;
    }
    const auto encode_end = std::chrono::steady_clock::now();
    encode_total_ms += std::chrono::duration<double, std::milli>(encode_end - encode_start).count();

    EncodedPacketView view;
    view.data = packet.data();
    view.size = static_cast<int>(packet.size());
    view.frame_index = encoded;
    view.keyframe = keyframe;

    const auto sink_start = std::chrono::steady_clock::now();
    if (!sink->write(view)) {
      std::fprintf(stderr, "encoded sink write failed frame_index=%ld\n", encoded);
      stop.request_stop();
      break;
    }
    const auto sink_end = std::chrono::steady_clock::now();
    sink_total_ms += std::chrono::duration<double, std::milli>(sink_end - sink_start).count();

    total_detections += static_cast<int64_t>(osd_frame->detections.size());
    total_bytes += static_cast<int64_t>(packet.size());
    ++encoded;
    if (config.verbose || encoded % 60 == 0) {
      std::fprintf(stderr, "encoded frame=%ld bytes=%zu detections=%zu\n", encoded, packet.size(), osd_frame->detections.size());
    }
  }

  if (sink) {
    sink->close();
  }

  const auto stage_end = std::chrono::steady_clock::now();
  const double elapsed_ms = std::chrono::duration<double, std::milli>(stage_end - stage_start).count();
  const double fps = elapsed_ms > 0.0 ? static_cast<double>(encoded) * 1000.0 / elapsed_ms : 0.0;
  std::fprintf(stderr,
               "[PERF] encoder_writer frames=%ld elapsed_ms=%.3f avg_stage_ms=%.3f fps=%.2f encode_avg_ms=%.3f sink_avg_ms=%.3f bytes=%ld total_detections=%ld\n",
               encoded,
               elapsed_ms,
               encoded > 0 ? elapsed_ms / static_cast<double>(encoded) : 0.0,
               fps,
               encoded > 0 ? encode_total_ms / static_cast<double>(encoded) : 0.0,
               encoded > 0 ? sink_total_ms / static_cast<double>(encoded) : 0.0,
               total_bytes,
               total_detections);
  std::fprintf(stderr, "encoder_writer thread finished, frames=%ld\n", encoded);
}

}  // namespace rkai