#include "decoder_thread.hpp"

#include <chrono>
#include <cstdio>
#include <thread>
#include <unistd.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavcodec/bsf.h>
#include <libavformat/avformat.h>
#include <rockchip/mpp_buffer.h>
#include <rockchip/rk_mpi.h>
}

namespace rkai {
namespace {

class AvPacketOwner {
 public:
  AvPacketOwner() : pkt_(av_packet_alloc()) {}
  ~AvPacketOwner() { av_packet_free(&pkt_); }
  AVPacket* get() const { return pkt_; }

 private:
  AVPacket* pkt_ = nullptr;
};

class DemuxContext {
 public:
  explicit DemuxContext(const std::string& input_path) {
    if (avformat_open_input(&fmt_, input_path.c_str(), nullptr, nullptr) < 0) {
      std::fprintf(stderr, "avformat_open_input failed: %s\n", input_path.c_str());
      return;
    }
    if (avformat_find_stream_info(fmt_, nullptr) < 0) {
      std::fprintf(stderr, "avformat_find_stream_info failed\n");
      return;
    }
    video_stream_ = av_find_best_stream(fmt_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (video_stream_ < 0) {
      std::fprintf(stderr, "no video stream found\n");
      return;
    }
    valid_ = true;
  }

  ~DemuxContext() {
    if (fmt_) {
      avformat_close_input(&fmt_);
    }
  }

  bool valid() const { return valid_; }
  AVFormatContext* fmt() const { return fmt_; }
  AVStream* video_stream() const { return fmt_->streams[video_stream_]; }
  int video_stream_index() const { return video_stream_; }

 private:
  AVFormatContext* fmt_ = nullptr;
  int video_stream_ = -1;
  bool valid_ = false;
};

MppCodingType to_mpp_codec(AVCodecID codec_id) {
  if (codec_id == AV_CODEC_ID_H264) {
    return MPP_VIDEO_CodingAVC;
  }
  if (codec_id == AV_CODEC_ID_HEVC) {
    return MPP_VIDEO_CodingHEVC;
  }
  return MPP_VIDEO_CodingUnused;
}

const char* bsf_name(AVCodecID codec_id) {
  if (codec_id == AV_CODEC_ID_H264) {
    return "h264_mp4toannexb";
  }
  if (codec_id == AV_CODEC_ID_HEVC) {
    return "hevc_mp4toannexb";
  }
  return nullptr;
}

class BitstreamFilter {
 public:
  explicit BitstreamFilter(AVStream* stream) {
    const char* name = bsf_name(stream->codecpar->codec_id);
    if (!name) {
      std::fprintf(stderr, "unsupported codec_id=%d\n", stream->codecpar->codec_id);
      return;
    }

    const AVBitStreamFilter* filter = av_bsf_get_by_name(name);
    if (!filter || av_bsf_alloc(filter, &ctx_) < 0) {
      std::fprintf(stderr, "av_bsf_alloc failed: %s\n", name);
      return;
    }
    if (avcodec_parameters_copy(ctx_->par_in, stream->codecpar) < 0) {
      std::fprintf(stderr, "avcodec_parameters_copy to bsf failed\n");
      return;
    }
    ctx_->time_base_in = stream->time_base;
    if (av_bsf_init(ctx_) < 0) {
      std::fprintf(stderr, "av_bsf_init failed: %s\n", name);
      return;
    }
    std::fprintf(stderr, "demux bsf=%s\n", name);
    valid_ = true;
  }

  ~BitstreamFilter() { av_bsf_free(&ctx_); }

  bool valid() const { return valid_; }
  AVBSFContext* get() const { return ctx_; }

 private:
  AVBSFContext* ctx_ = nullptr;
  bool valid_ = false;
};

class MppDecoder {
 public:
  explicit MppDecoder(MppCodingType codec) {
    if (mpp_create(&ctx_, &mpi_) != MPP_OK) {
      std::fprintf(stderr, "mpp_create failed\n");
      return;
    }
    if (mpp_init(ctx_, MPP_CTX_DEC, codec) != MPP_OK) {
      std::fprintf(stderr, "mpp_init decoder failed\n");
      return;
    }

    MppDecCfg cfg = nullptr;
    if (mpp_dec_cfg_init(&cfg) == MPP_OK) {
      mpi_->control(ctx_, MPP_DEC_GET_CFG, cfg);
      mpp_dec_cfg_set_u32(cfg, "base:split_parse", 1);
      mpi_->control(ctx_, MPP_DEC_SET_CFG, cfg);
      mpp_dec_cfg_deinit(cfg);
    }
    valid_ = true;
  }

  ~MppDecoder() {
    if (frame_group_) {
      mpp_buffer_group_put(frame_group_);
    }
    if (ctx_) {
      mpp_destroy(ctx_);
    }
  }

  bool valid() const { return valid_; }
  MppCtx ctx() const { return ctx_; }
  MppApi* mpi() const { return mpi_; }
  MppBufferGroup& frame_group() { return frame_group_; }

 private:
  MppCtx ctx_ = nullptr;
  MppApi* mpi_ = nullptr;
  MppBufferGroup frame_group_ = nullptr;
  bool valid_ = false;
};

const char* format_name(int fmt) {
  switch (fmt) {
    case MPP_FMT_YUV420SP:
      return "NV12";
    case MPP_FMT_YUV420SP_VU:
      return "NV21";
    default:
      return "OTHER";
  }
}

bool configure_external_frame_group(MppDecoder& decoder, MppFrame frame) {
  MppBufferGroup& group = decoder.frame_group();
  const size_t buffer_size = mpp_frame_get_buf_size(frame);

  if (!group) {
    if (mpp_buffer_group_get_internal(&group, MPP_BUFFER_TYPE_DRM) != MPP_OK) {
      std::fprintf(stderr, "mpp_buffer_group_get_internal DRM failed\n");
      return false;
    }
    if (decoder.mpi()->control(decoder.ctx(), MPP_DEC_SET_EXT_BUF_GROUP, group) != MPP_OK) {
      std::fprintf(stderr, "MPP_DEC_SET_EXT_BUF_GROUP failed\n");
      return false;
    }
  } else {
    mpp_buffer_group_clear(group);
  }

  mpp_buffer_group_limit_config(group, buffer_size, 16);
  decoder.mpi()->control(decoder.ctx(), MPP_DEC_SET_INFO_CHANGE_READY, nullptr);

  std::fprintf(stderr,
               "decoder info-change: coded=%dx%d stride=%dx%d fmt=%s buffer_size=%zu\n",
               static_cast<int>(mpp_frame_get_width(frame)),
               static_cast<int>(mpp_frame_get_height(frame)),
               static_cast<int>(mpp_frame_get_hor_stride(frame)),
               static_cast<int>(mpp_frame_get_ver_stride(frame)),
               format_name(static_cast<int>(mpp_frame_get_fmt(frame))),
               buffer_size);
  return true;
}

bool drain_frames(MppDecoder& decoder,
                  int64_t& next_frame_id,
                  StopFlag& stop,
                  DecodedFrameQueue& output,
                  bool verbose) {
  while (!stop.stop_requested()) {
    MppFrame frame = nullptr;
    MPP_RET ret = decoder.mpi()->decode_get_frame(decoder.ctx(), &frame);
    if (ret == MPP_ERR_TIMEOUT || !frame) {
      return true;
    }
    if (ret != MPP_OK) {
      std::fprintf(stderr, "decode_get_frame failed ret=%d\n", ret);
      return false;
    }

    if (mpp_frame_get_info_change(frame)) {
      const bool ok = configure_external_frame_group(decoder, frame);
      mpp_frame_deinit(&frame);
      if (!ok) {
        return false;
      }
      continue;
    }

    const RK_U32 bad_frame = mpp_frame_get_errinfo(frame) | mpp_frame_get_discard(frame);
    if (bad_frame) {
      std::fprintf(stderr, "discard bad frame err=%u\n", bad_frame);
      mpp_frame_deinit(&frame);
      continue;
    }

    MppBuffer buffer = mpp_frame_get_buffer(frame);
    if (!buffer) {
      std::fprintf(stderr, "decoded frame has no mpp buffer\n");
      mpp_frame_deinit(&frame);
      continue;
    }

    auto decoded = std::make_shared<DecodedFrame>();
    decoded->frame_id = next_frame_id++;
    decoded->pts = mpp_frame_get_pts(frame);
    decoded->width = static_cast<int>(mpp_frame_get_width(frame));
    decoded->height = static_cast<int>(mpp_frame_get_height(frame));
    decoded->hor_stride = static_cast<int>(mpp_frame_get_hor_stride(frame));
    decoded->ver_stride = static_cast<int>(mpp_frame_get_ver_stride(frame));
    decoded->format = static_cast<int>(mpp_frame_get_fmt(frame));
    decoded->dma_fd = mpp_buffer_get_fd(buffer);
    decoded->frame = frame;

    if (verbose) {
      std::fprintf(stderr,
                   "decoded frame=%ld pts=%ld size=%dx%d stride=%dx%d fmt=%s dma_fd=%d\n",
                   decoded->frame_id,
                   decoded->pts,
                   decoded->width,
                   decoded->height,
                   decoded->hor_stride,
                   decoded->ver_stride,
                   format_name(decoded->format),
                   decoded->dma_fd);
    }

    if (!output.push(std::move(decoded))) {
      return false;
    }
  }
  return true;
}

bool put_mpp_packet_once(MppDecoder& decoder, AVPacket* packet, bool eos, MPP_RET& ret) {
  MppPacket mpp_packet = nullptr;

  if (packet) {
    ret = mpp_packet_init(&mpp_packet, packet->data, packet->size);
    if (ret == MPP_OK) {
      mpp_packet_set_pts(mpp_packet, packet->pts);
      mpp_packet_set_dts(mpp_packet, packet->dts);
    }
  } else {
    ret = mpp_packet_init(&mpp_packet, nullptr, 0);
  }

  if (ret != MPP_OK || !mpp_packet) {
    std::fprintf(stderr, "mpp_packet_init failed ret=%d\n", ret);
    return false;
  }
  if (eos) {
    mpp_packet_set_eos(mpp_packet);
  }

  ret = decoder.mpi()->decode_put_packet(decoder.ctx(), mpp_packet);
  mpp_packet_deinit(&mpp_packet);
  return ret == MPP_OK;
}

bool put_mpp_packet(MppDecoder& decoder,
                    AVPacket* packet,
                    bool eos,
                    int64_t& next_frame_id,
                    StopFlag& stop,
                    DecodedFrameQueue& output,
                    bool verbose) {
  MPP_RET ret = MPP_OK;
  for (int retry = 0; retry < 200 && !stop.stop_requested(); ++retry) {
    if (put_mpp_packet_once(decoder, packet, eos, ret)) {
      return true;
    }
    if (!drain_frames(decoder, next_frame_id, stop, output, verbose)) {
      return false;
    }
    usleep(1000);
  }

  std::fprintf(stderr, "decode_put_packet failed ret=%d size=%d eos=%d\n", ret, packet ? packet->size : 0, eos ? 1 : 0);
  return false;
}

bool send_filtered_packets(BitstreamFilter& bsf,
                           AVPacket* input,
                           MppDecoder& decoder,
                           int64_t& next_frame_id,
                           StopFlag& stop,
                           DecodedFrameQueue& output,
                           bool verbose) {
  int ret = av_bsf_send_packet(bsf.get(), input);
  if (ret < 0) {
    std::fprintf(stderr, "av_bsf_send_packet failed ret=%d\n", ret);
    return false;
  }

  AvPacketOwner filtered;
  while (!stop.stop_requested()) {
    ret = av_bsf_receive_packet(bsf.get(), filtered.get());
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      return true;
    }
    if (ret < 0) {
      std::fprintf(stderr, "av_bsf_receive_packet failed ret=%d\n", ret);
      return false;
    }

    if (!put_mpp_packet(decoder, filtered.get(), false, next_frame_id, stop, output, verbose)) {
      av_packet_unref(filtered.get());
      return false;
    }
    av_packet_unref(filtered.get());
    if (!drain_frames(decoder, next_frame_id, stop, output, verbose)) {
      return false;
    }
  }
  return true;
}

}  // namespace

DecodedFrame::DecodedFrame(DecodedFrame&& other) noexcept {
  *this = std::move(other);
}

DecodedFrame& DecodedFrame::operator=(DecodedFrame&& other) noexcept {
  if (this != &other) {
    if (frame) {
      mpp_frame_deinit(&frame);
    }
    frame_id = other.frame_id;
    pts = other.pts;
    width = other.width;
    height = other.height;
    hor_stride = other.hor_stride;
    ver_stride = other.ver_stride;
    format = other.format;
    dma_fd = other.dma_fd;
    frame = other.frame;
    other.frame = nullptr;
    other.dma_fd = -1;
  }
  return *this;
}

DecodedFrame::~DecodedFrame() {
  if (frame) {
    mpp_frame_deinit(&frame);
  }
}

void StopFlag::request_stop() {
  stopped_.store(true, std::memory_order_release);
}

bool StopFlag::stop_requested() const {
  return stopped_.load(std::memory_order_acquire);
}

void decoder_thread(const DecoderConfig& config, StopFlag& stop, DecodedFrameQueue& output) {
  const auto stage_start = std::chrono::steady_clock::now();
  DemuxContext demux(config.input_path);
  if (!demux.valid()) {
    stop.request_stop();
    output.close();
    return;
  }

  AVStream* stream = demux.video_stream();
  const MppCodingType codec = to_mpp_codec(stream->codecpar->codec_id);
  if (codec == MPP_VIDEO_CodingUnused) {
    std::fprintf(stderr, "only H264/H265 MP4 input is supported, codec_id=%d\n", stream->codecpar->codec_id);
    stop.request_stop();
    output.close();
    return;
  }

  std::fprintf(stderr,
               "input video: %dx%d codec_id=%d stream=%d\n",
               stream->codecpar->width,
               stream->codecpar->height,
               stream->codecpar->codec_id,
               demux.video_stream_index());

  BitstreamFilter bsf(stream);
  if (!bsf.valid()) {
    stop.request_stop();
    output.close();
    return;
  }

  MppDecoder decoder(codec);
  if (!decoder.valid()) {
    stop.request_stop();
    output.close();
    return;
  }

  AvPacketOwner packet;
  int64_t next_frame_id = 0;
  while (!stop.stop_requested() && av_read_frame(demux.fmt(), packet.get()) >= 0) {
    if (packet.get()->stream_index == demux.video_stream_index()) {
      if (!send_filtered_packets(bsf, packet.get(), decoder, next_frame_id, stop, output, config.verbose)) {
        stop.request_stop();
        break;
      }
    }
    av_packet_unref(packet.get());
  }

  if (!stop.stop_requested()) {
    send_filtered_packets(bsf, nullptr, decoder, next_frame_id, stop, output, config.verbose);
    put_mpp_packet(decoder, nullptr, true, next_frame_id, stop, output, config.verbose);
    for (int i = 0; i < 200 && !stop.stop_requested(); ++i) {
      drain_frames(decoder, next_frame_id, stop, output, config.verbose);
      usleep(1000);
    }
  }

  const auto stage_end = std::chrono::steady_clock::now();
  const double elapsed_ms = std::chrono::duration<double, std::milli>(stage_end - stage_start).count();
  const double fps = elapsed_ms > 0.0 ? static_cast<double>(next_frame_id) * 1000.0 / elapsed_ms : 0.0;
  std::fprintf(stderr,
               "[PERF] decoder frames=%ld elapsed_ms=%.3f avg_ms=%.3f fps=%.2f\n",
               next_frame_id,
               elapsed_ms,
               next_frame_id > 0 ? elapsed_ms / static_cast<double>(next_frame_id) : 0.0,
               fps);
  std::fprintf(stderr, "decoder thread finished, decoded_frames=%ld\n", next_frame_id);
  output.close();
}

}  // namespace rkai