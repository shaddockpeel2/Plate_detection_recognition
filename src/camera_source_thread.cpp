#include "camera_source_thread.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <linux/videodev2.h>

extern "C" {
#include <rockchip/mpp_buffer.h>
#include <rockchip/mpp_frame.h>
}

namespace rkai {
namespace {

constexpr int kBufferCount = 4;
constexpr int kPollTimeoutMs = 2000;

int align16(int value) {
  return (value + 15) & ~15;
}

bool xioctl(int fd, unsigned long request, void* arg, const char* name) {
  for (;;) {
    if (ioctl(fd, request, arg) == 0) {
      return true;
    }
    if (errno == EINTR) {
      continue;
    }
    std::fprintf(stderr, "%s failed errno=%d message=%s\n", name, errno, std::strerror(errno));
    return false;
  }
}

struct FdOwner {
  int fd = -1;

  FdOwner() = default;
  explicit FdOwner(int value) : fd(value) {}
  FdOwner(const FdOwner&) = delete;
  FdOwner& operator=(const FdOwner&) = delete;
  ~FdOwner() {
    if (fd >= 0) {
      close(fd);
    }
  }

  bool valid() const { return fd >= 0; }
};

struct MappedBuffer {
  void* start = nullptr;
  size_t length = 0;
};

class CameraMmapRuntime {
 public:
  ~CameraMmapRuntime() { close_stream(); }

  bool open_device(const std::string& device) {
    fd_ = std::unique_ptr<FdOwner>(new FdOwner(open(device.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC)));
    if (!fd_->valid()) {
      std::fprintf(stderr, "open camera failed: %s errno=%d message=%s\n", device.c_str(), errno, std::strerror(errno));
      return false;
    }
    return true;
  }

  bool configure(const CameraSourceConfig& config) {
    v4l2_capability cap{};
    if (!xioctl(fd_->fd, VIDIOC_QUERYCAP, &cap, "VIDIOC_QUERYCAP")) {
      return false;
    }
    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) || !(cap.capabilities & V4L2_CAP_STREAMING)) {
      std::fprintf(stderr, "camera does not support video capture streaming\n");
      return false;
    }

    v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = config.width;
    fmt.fmt.pix.height = config.height;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    if (!xioctl(fd_->fd, VIDIOC_S_FMT, &fmt, "VIDIOC_S_FMT")) {
      return false;
    }
    if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_YUYV) {
      std::fprintf(stderr, "camera selected unsupported format fourcc=0x%x, first version supports YUYV only\n", fmt.fmt.pix.pixelformat);
      return false;
    }

    width_ = static_cast<int>(fmt.fmt.pix.width);
    height_ = static_cast<int>(fmt.fmt.pix.height);
    yuyv_stride_ = static_cast<int>(fmt.fmt.pix.bytesperline);
    if (yuyv_stride_ <= 0) {
      yuyv_stride_ = width_ * 2;
    }

    v4l2_streamparm parm{};
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator = config.fps_den > 0 ? config.fps_den : 1;
    parm.parm.capture.timeperframe.denominator = config.fps_num > 0 ? config.fps_num : 25;
    xioctl(fd_->fd, VIDIOC_S_PARM, &parm, "VIDIOC_S_PARM");

    std::fprintf(stderr,
                 "camera configured: %s %dx%d yuyv_stride=%d fps=%d/%d\n",
                 reinterpret_cast<const char*>(cap.card),
                 width_,
                 height_,
                 yuyv_stride_,
                 config.fps_num,
                 config.fps_den);
    return true;
  }

  bool init_mmap() {
    v4l2_requestbuffers req{};
    req.count = kBufferCount;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (!xioctl(fd_->fd, VIDIOC_REQBUFS, &req, "VIDIOC_REQBUFS")) {
      return false;
    }
    if (req.count < 2) {
      std::fprintf(stderr, "insufficient V4L2 buffers count=%u\n", req.count);
      return false;
    }

    buffers_.resize(req.count);
    for (uint32_t i = 0; i < req.count; ++i) {
      v4l2_buffer buf{};
      buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buf.memory = V4L2_MEMORY_MMAP;
      buf.index = i;
      if (!xioctl(fd_->fd, VIDIOC_QUERYBUF, &buf, "VIDIOC_QUERYBUF")) {
        return false;
      }
      void* start = mmap(nullptr, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_->fd, buf.m.offset);
      if (start == MAP_FAILED) {
        std::fprintf(stderr, "mmap camera buffer failed index=%u errno=%d message=%s\n", i, errno, std::strerror(errno));
        return false;
      }
      buffers_[i].start = start;
      buffers_[i].length = buf.length;
    }
    return true;
  }

  bool start() {
    for (uint32_t i = 0; i < buffers_.size(); ++i) {
      v4l2_buffer buf{};
      buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buf.memory = V4L2_MEMORY_MMAP;
      buf.index = i;
      if (!xioctl(fd_->fd, VIDIOC_QBUF, &buf, "VIDIOC_QBUF")) {
        return false;
      }
    }
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (!xioctl(fd_->fd, VIDIOC_STREAMON, &type, "VIDIOC_STREAMON")) {
      return false;
    }
    streaming_ = true;
    return true;
  }

  bool wait_frame() {
    pollfd pfd{};
    pfd.fd = fd_->fd;
    pfd.events = POLLIN;
    const int ret = poll(&pfd, 1, kPollTimeoutMs);
    if (ret == 0) {
      std::fprintf(stderr, "camera poll timeout\n");
      return false;
    }
    if (ret < 0) {
      if (errno == EINTR) {
        return true;
      }
      std::fprintf(stderr, "camera poll failed errno=%d message=%s\n", errno, std::strerror(errno));
      return false;
    }
    return (pfd.revents & POLLIN) != 0;
  }

  bool dequeue(v4l2_buffer& buf) {
    std::memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    if (ioctl(fd_->fd, VIDIOC_DQBUF, &buf) == 0) {
      return true;
    }
    if (errno == EAGAIN || errno == EINTR) {
      return false;
    }
    std::fprintf(stderr, "VIDIOC_DQBUF failed errno=%d message=%s\n", errno, std::strerror(errno));
    return false;
  }

  bool requeue(const v4l2_buffer& buf) {
    v4l2_buffer qbuf = buf;
    return xioctl(fd_->fd, VIDIOC_QBUF, &qbuf, "VIDIOC_QBUF");
  }

  const MappedBuffer& buffer(uint32_t index) const { return buffers_[index]; }
  int width() const { return width_; }
  int height() const { return height_; }
  int yuyv_stride() const { return yuyv_stride_; }

 private:
  void close_stream() {
    if (streaming_ && fd_ && fd_->valid()) {
      v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      ioctl(fd_->fd, VIDIOC_STREAMOFF, &type);
      streaming_ = false;
    }
    for (auto& buffer : buffers_) {
      if (buffer.start && buffer.start != MAP_FAILED) {
        munmap(buffer.start, buffer.length);
        buffer.start = nullptr;
        buffer.length = 0;
      }
    }
    buffers_.clear();
  }

  std::unique_ptr<FdOwner> fd_;
  std::vector<MappedBuffer> buffers_;
  bool streaming_ = false;
  int width_ = 0;
  int height_ = 0;
  int yuyv_stride_ = 0;
};

struct MppBufferOwner {
  MppBuffer buffer = nullptr;

  explicit MppBufferOwner(MppBuffer value) : buffer(value) {}
  MppBufferOwner(const MppBufferOwner&) = delete;
  MppBufferOwner& operator=(const MppBufferOwner&) = delete;
  ~MppBufferOwner() {
    if (buffer) {
      mpp_buffer_put(buffer);
    }
  }
};

bool make_mpp_frame(int width,
                    int height,
                    int hor_stride,
                    int ver_stride,
                    MppBuffer buffer,
                    int64_t frame_id,
                    MppFrame& out) {
  MppFrame frame = nullptr;
  if (mpp_frame_init(&frame) != MPP_OK || !frame) {
    std::fprintf(stderr, "mpp_frame_init camera frame failed\n");
    return false;
  }
  mpp_frame_set_width(frame, width);
  mpp_frame_set_height(frame, height);
  mpp_frame_set_hor_stride(frame, hor_stride);
  mpp_frame_set_ver_stride(frame, ver_stride);
  mpp_frame_set_fmt(frame, MPP_FMT_YUV420SP);
  mpp_frame_set_buffer(frame, buffer);
  mpp_frame_set_pts(frame, frame_id);
  out = frame;
  return true;
}

void yuyv_to_nv12(const uint8_t* src,
                  int width,
                  int height,
                  int src_stride,
                  uint8_t* dst,
                  int dst_stride,
                  int dst_ver_stride) {
  uint8_t* y_plane = dst;
  uint8_t* uv_plane = dst + dst_stride * dst_ver_stride;

  for (int y = 0; y < height; ++y) {
    const uint8_t* src_row = src + y * src_stride;
    uint8_t* dst_y = y_plane + y * dst_stride;
    for (int x = 0; x < width; x += 2) {
      dst_y[x] = src_row[x * 2];
      if (x + 1 < width) {
        dst_y[x + 1] = src_row[x * 2 + 2];
      }
    }
  }

  for (int y = 0; y < height; y += 2) {
    const uint8_t* src_row = src + y * src_stride;
    uint8_t* dst_uv = uv_plane + (y / 2) * dst_stride;
    for (int x = 0; x < width; x += 2) {
      dst_uv[x] = src_row[x * 2 + 1];
      if (x + 1 < width) {
        dst_uv[x + 1] = src_row[x * 2 + 3];
      }
    }
  }
}

bool alloc_camera_frame(MppBufferGroup group,
                        const CameraMmapRuntime& camera,
                        const v4l2_buffer& v4l2_buf,
                        int64_t frame_id,
                        DecodedFramePtr& out) {
  const int width = camera.width();
  const int height = camera.height();
  const int hor_stride = align16(width);
  const int ver_stride = align16(height);
  const size_t nv12_size = static_cast<size_t>(hor_stride) * ver_stride * 3 / 2;

  MppBuffer buffer = nullptr;
  if (mpp_buffer_get(group, &buffer, nv12_size) != MPP_OK || !buffer) {
    std::fprintf(stderr, "mpp_buffer_get camera frame failed size=%zu\n", nv12_size);
    return false;
  }

  auto owner = std::make_shared<MppBufferOwner>(buffer);
  uint8_t* dst = static_cast<uint8_t*>(mpp_buffer_get_ptr(buffer));
  if (!dst) {
    std::fprintf(stderr, "mpp_buffer_get_ptr camera frame failed\n");
    return false;
  }
  std::memset(dst, 0, nv12_size);

  const MappedBuffer& src_buffer = camera.buffer(v4l2_buf.index);
  const uint8_t* src = static_cast<const uint8_t*>(src_buffer.start);
  yuyv_to_nv12(src, width, height, camera.yuyv_stride(), dst, hor_stride, ver_stride);

  MppFrame frame = nullptr;
  if (!make_mpp_frame(width, height, hor_stride, ver_stride, buffer, frame_id, frame)) {
    return false;
  }

  auto decoded = std::make_shared<DecodedFrame>();
  decoded->frame_id = frame_id;
  decoded->pts = static_cast<int64_t>(v4l2_buf.timestamp.tv_sec) * 1000000LL + v4l2_buf.timestamp.tv_usec;
  decoded->width = width;
  decoded->height = height;
  decoded->hor_stride = hor_stride;
  decoded->ver_stride = ver_stride;
  decoded->format = MPP_FMT_YUV420SP;
  decoded->dma_fd = mpp_buffer_get_fd(buffer);
  decoded->frame = frame;
  decoded->owner = owner;
  out = std::move(decoded);
  return true;
}

}  // namespace

void camera_source_thread(const CameraSourceConfig& config, StopFlag& stop, DecodedFrameQueue& output) {
  const auto stage_start = std::chrono::steady_clock::now();
  int64_t frame_id = 0;

  CameraMmapRuntime camera;
  MppBufferGroup group = nullptr;

  if (!camera.open_device(config.device) || !camera.configure(config) || !camera.init_mmap()) {
    stop.request_stop();
    output.close();
    return;
  }

  if (mpp_buffer_group_get_internal(&group, MPP_BUFFER_TYPE_DRM) != MPP_OK || !group) {
    std::fprintf(stderr, "mpp_buffer_group_get_internal camera DRM failed\n");
    stop.request_stop();
    output.close();
    return;
  }

  if (!camera.start()) {
    mpp_buffer_group_put(group);
    stop.request_stop();
    output.close();
    return;
  }

  while (!stop.stop_requested()) {
    if (config.frame_limit > 0 && frame_id >= config.frame_limit) {
      break;
    }
    if (!camera.wait_frame()) {
      stop.request_stop();
      break;
    }

    v4l2_buffer buf{};
    if (!camera.dequeue(buf)) {
      continue;
    }

    DecodedFramePtr decoded;
    const bool ok = alloc_camera_frame(group, camera, buf, frame_id, decoded);
    const bool requeued = camera.requeue(buf);
    if (!ok || !requeued) {
      stop.request_stop();
      break;
    }

    if (config.verbose || frame_id % 60 == 0) {
      std::fprintf(stderr,
                   "camera frame=%ld pts=%ld size=%dx%d stride=%dx%d dma_fd=%d\n",
                   decoded->frame_id,
                   decoded->pts,
                   decoded->width,
                   decoded->height,
                   decoded->hor_stride,
                   decoded->ver_stride,
                   decoded->dma_fd);
    }

    ++frame_id;
    if (!output.push(std::move(decoded))) {
      break;
    }
  }

  if (group) {
    mpp_buffer_group_put(group);
  }

  const auto stage_end = std::chrono::steady_clock::now();
  const double elapsed_ms = std::chrono::duration<double, std::milli>(stage_end - stage_start).count();
  const double fps = elapsed_ms > 0.0 ? static_cast<double>(frame_id) * 1000.0 / elapsed_ms : 0.0;
  std::fprintf(stderr,
               "[PERF] camera_source frames=%ld elapsed_ms=%.3f fps=%.2f\n",
               frame_id,
               elapsed_ms,
               fps);
  std::fprintf(stderr, "camera_source thread finished, frames=%ld\n", frame_id);
  output.close();
}

}  // namespace rkai