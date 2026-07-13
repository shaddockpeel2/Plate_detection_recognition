# RK3588 MP4 YOLO Pipeline

这是一个面向 RK3588 / aarch64 Linux 的 C++ 视频推理流水线示例。项目支持两类输入：MP4 文件和 USB 摄像头。MP4 输入经过 FFmpeg 解封装、Rockchip MPP 硬解后进入推理链路；摄像头输入通过 V4L2 采集 `YUYV` 帧，并在输入分支内转换为 `NV12` 后接入同一条推理链路。后续统一经过 RGA 预处理、RKNN 推理、后处理/OCR/OSD，再通过 MPP 编码输出 MP4。目标是车牌识别，涉及 YOLO 和 OCR。

## 功能链路

```text
MP4 输入
  -> FFmpeg 解封装
  -> Rockchip MPP 硬解
  -> DecodedFrameQueue

USB 摄像头输入
  -> V4L2 采集 YUYV
  -> 输入分支转换为 NV12
  -> DecodedFrameQueue

DecodedFrameQueue
  -> RGA 图像预处理
  -> RKNN Runtime 推理
  -> 后处理 / OCR / OSD
  -> Rockchip MPP 硬编码
  -> MP4 输出
```

新增摄像头功能只作为输入分支接入 `DecodedFrameQueue`，尽量复用原有 MP4 后半段处理链路。

## 支持环境

当前项目主要面向：

```text
芯片平台：RK3588
系统架构：aarch64 Linux
构建系统：CMake + g++
推理后端：RKNN Runtime
视频硬件链路：Rockchip MPP + RGA
```

不建议直接在 x86 Linux 上构建运行，除非你自行替换 MPP/RGA/RKNN 依赖并调整代码路径。

## 系统依赖

需要系统已安装：

```bash
sudo apt install -y cmake g++ pkg-config
```

还需要 FFmpeg 开发库，并且 `pkg-config` 能找到：

```text
libavformat
libavcodec
libavutil
```

可以用下面命令检查：

```bash
pkg-config --modversion libavformat libavcodec libavutil
```

注意：`ffmpeg --version` 显示的是命令行程序版本；本项目编译时实际以 `pkg-config` 找到的开发库版本为准。

## 项目内置 Rockchip 依赖

项目已经内置 RK3588 相关的 Rockchip 头文件和动态库：

```text
third_party/rockchip/mpp/
third_party/rockchip/rga/RK3588/
third_party/rockchip/rknn/RK3588/Linux/librknn_api/
```

默认构建会优先使用这些项目内依赖。

如果你希望使用外部 Rockchip SDK，可以在配置 CMake 时指定：

```bash
cmake -S . -B build -DRKNPU2_DIR=/path/to/rknpu2
```

其中 `/path/to/rknpu2` 应包含类似结构：

```text
3rdparty/mpp
3rdparty/rga/RK3588
runtime/RK3588/Linux/librknn_api
```

## 构建

推荐在项目根目录执行：

```bash
cmake -S . -B build
cmake --build build --target rk_mp4_yolo_stage5 -j$(nproc)
```

默认会启用：

```text
USE_VENDOR_RK_LIBS=ON
```

也就是优先链接项目内的 `third_party/rockchip` 依赖。

## 运行目标检测 Demo

当前程序支持通过命令行参数指定输入视频、模型和输出路径：

```bash
./build/rk_mp4_yolo_stage5 \
  ./video/test-video/5s.mp4 \
  ./models/car-v8/v8-car-relu-3588.rknn \
  ./video/output-video/output-5s.mp4
```

参数顺序：

```text
argv[1] 输入视频路径
argv[2] RKNN 模型路径
argv[3] 输出视频路径
```

当前仓库保留 demo 输入视频：

```text
video/test-video/5s.mp4
```

输出目录：

```text
video/output-video/
```

该目录属于运行产物，默认被 `.gitignore` 忽略。

## 运行目标检测+ocr Demo

当前程序支持通过命令行参数指定输入视频、模型和输出路径：

```bash
./build/rk_mp4_yolo_stage5 "/home/cat/mpp-main/yolo26videeotest/main2/video/test-video/5s.mp4" "/home/cat/mpp-main/yolo26videeotest/main2/models/car-v8/v8-car-relu-3588.rknn" "/home/cat/mpp-main/yolo26videeotest/main2/output_stage5_chinese_osd.mp4" "/home/cat/mpp-main/yolo26videeotest/main2/ppocrv5/PP-OCRv5_mobile_rec_license_plate.rknn" "/home/cat/mpp-main/yolo26videeotest/main2/ppocrv5/model/license_plate_dict.txt" 30 90
```

参数顺序：

```text
argv[1] 输入视频路径
argv[2] RKNN 模型路径
argv[3] 输出视频路径
argv[4] ocr模路径
argv[5] ocr文字符
```

当前仓库保留 demo 输入视频：

```text
video/test-video/8s.mp4
```

输出目录：

```text
video/output-video/
```

该目录属于运行产物，默认被 `.gitignore` 忽略。

## 运行摄像头推理并保存 MP4

摄像头作为独立输入分支接入 `DecodedFrameQueue`，后续 RGA 预处理、RKNN 推理、YOLO/OCR、OSD 和 MPP 编码链路保持复用。

当前第一版支持 USB 摄像头 `YUYV` 采集，并在摄像头输入模块内部转换为 `NV12` 后接入主链路。建议先使用当前摄像头支持的 `640x480 @ 25fps`：

```bash
./build/rk_mp4_yolo_stage5 \
  --input camera \
  --device /dev/video0 \
  --width 640 \
  --height 480 \
  --fps 25 \
  --frame-limit 300 \
  --model ./models/car-v8/v8-car-relu-3588.rknn \
  --output ./video/output-video/camera-output.mp4 \
  --ocr-model ./ppocrv5/PP-OCRv5_mobile_rec_license_plate.rknn \
  --ocr-vocab ./ppocrv5/model/license_plate_dict.txt
```

摄像头能力可用下面命令确认：

```bash
ls -l /dev/video*
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
```

## 远程车牌事件上报

远程上报链路采用“HTTP 上传截图 + MQTT 发布事件 JSON”的拆分设计：

```text
YOLO / ByteTrack / OCR
  -> 上传阈值与去重
  -> 上传队列
  -> 上传线程
  -> 裁剪截图并编码 JPEG
  -> HTTP 上传图片
  -> MQTT 发布事件 JSON
```

关键约束：

```text
MQTT 只传事件和 image_url，不传图片二进制
网络请求只发生在上传线程，不阻塞检测、OCR、OSD 和编码主链路
上传队列满时丢弃新事件，不反压主链路
默认关闭远程上报，必须显式打开 --upload-event-log
```

### 服务器端服务

服务器需要准备三类能力：

```text
1883  MQTT Broker，例如 Mosquitto
8000  HTTP 图片上传服务，接收 multipart/form-data 字段 file
80    图片静态访问服务，例如 http://124.220.49.93/images/xxx.jpg
```

安全组需要放行：

```text
TCP 1883
TCP 8000
TCP 80
```

图片上传服务约定：

```text
请求：POST /upload
字段：file
返回：JSON
```

返回值支持两种形式：

```json
{"url":"/images/rk3588-001_123_7.jpg"}
```

或：

```json
{"url":"http://124.220.49.93/images/rk3588-001_123_7.jpg"}
```

开发板会把返回的 `url` 写进 MQTT 事件的 `image_url` 字段。

可先在服务器或开发板上用一张本地图片验证 HTTP 上传：

```bash
curl -F "file=@test.jpg" http://124.220.49.93:8000/upload
```

期望返回类似：

```json
{"url":"/images/test.jpg"}
```

并确认图片可以访问：

```bash
curl -I http://124.220.49.93/images/test.jpg
```

### 远端 MQTT 订阅验证

当前默认 topic：

```text
devices/rk3588-001/plate/events
devices/rk3588-001/status/heartbeat
devices/rk3588-001/status/error
```

订阅全部远程上报消息：

```bash
mosquitto_sub -h 124.220.49.93 -p 1883 \
  -u rk3588 -P 'Rk3588_Mqtt_123' \
  -t 'devices/rk3588-001/#' -v
```

只订阅车牌事件：

```bash
mosquitto_sub -h 124.220.49.93 -p 1883 \
  -u rk3588 -P 'Rk3588_Mqtt_123' \
  -t 'devices/rk3588-001/plate/events' -v
```

收到的事件结构类似：

```json
{
  "device_id": "rk3588-001",
  "event_id": "rk3588-001-123456-7",
  "frame_id": 123456,
  "timestamp_ms": 1720000000000,
  "track_id": 7,
  "plate_text": "粤B12345",
  "plate_score": 0.91,
  "detect_score": 0.86,
  "bbox": {
    "x1": 120,
    "y1": 240,
    "x2": 360,
    "y2": 300
  },
  "image_url": "http://124.220.49.93/images/rk3588-001_123456_7.jpg"
}
```

### 开发板运行命令

摄像头输入示例：

```bash
./build/rk_mp4_yolo_stage5 \
  --input camera \
  --device /dev/video0 \
  --width 640 \
  --height 480 \
  --fps 25 \
  --frame-limit 300 \
  --model ./models/car-v8/v8-car-relu-3588.rknn \
  --output ./video/output-video/camera-output.mp4 \
  --ocr-model ./ppocrv5/PP-OCRv5_mobile_rec_license_plate.rknn \
  --ocr-vocab ./ppocrv5/model/license_plate_dict.txt \
  --upload-event-log on \
  --upload-min-detect-score 0.50 \
  --upload-min-plate-score 0.80 \
  --upload-track-cooldown-frames 90 \
  --upload-plate-cooldown-frames 1800 \
  --upload-queue-capacity 32 \
  --upload-snapshot-dir ./upload_snapshots \
  --upload-url http://124.220.49.93:8000/upload \
  --upload-public-base-url http://124.220.49.93 \
  --upload-jpeg-quality 85 \
  --upload-http-timeout-ms 3000 \
  --mqtt on \
  --mqtt-host 124.220.49.93 \
  --mqtt-port 1883 \
  --mqtt-username rk3588 \
  --mqtt-password Rk3588_Mqtt_123 \
  --mqtt-client-id rk3588-001 \
  --mqtt-topic devices/rk3588-001/plate/events \
  --mqtt-heartbeat-topic devices/rk3588-001/status/heartbeat \
  --mqtt-error-topic devices/rk3588-001/status/error \
  --mqtt-heartbeat-interval-sec 30 \
  --mqtt-timeout-ms 3000
```

MP4 输入示例：

```bash
./build/rk_mp4_yolo_stage5 \
  --input mp4 \
  --video ./video/test-video/9s.mp4 \
  --model ./models/car-v8/v8-car-relu-3588.rknn \
  --output ./video/output-video/remote-report-test.mp4 \
  --ocr-model ./ppocrv5/PP-OCRv5_mobile_rec_license_plate.rknn \
  --ocr-vocab ./ppocrv5/model/license_plate_dict.txt \
  --upload-event-log on \
  --upload-url http://124.220.49.93:8000/upload \
  --upload-public-base-url http://124.220.49.93 \
  --mqtt on \
  --mqtt-host 124.220.49.93 \
  --mqtt-port 1883 \
  --mqtt-username rk3588 \
  --mqtt-password Rk3588_Mqtt_123 \
  --mqtt-client-id rk3588-001 \
  --mqtt-topic devices/rk3588-001/plate/events \
  --mqtt-heartbeat-topic devices/rk3588-001/status/heartbeat \
  --mqtt-error-topic devices/rk3588-001/status/error
```

### 远程上报参数

```text
--upload-event-log on/off              开启上传事件判定和上传线程
--device-id rk3588-001                 设备 ID
--upload-min-detect-score 0.50         检测置信度阈值
--upload-min-plate-score 0.80          OCR 车牌置信度阈值
--upload-track-cooldown-frames 90      同一 track_id 冷却帧数
--upload-plate-cooldown-frames 1800    同一 plate_text 冷却帧数
--upload-queue-capacity 32             上传队列容量
--upload-snapshot-dir ./upload_snapshots
--upload-url http://124.220.49.93:8000/upload
--upload-public-base-url http://124.220.49.93
--upload-jpeg-quality 85
--upload-http-timeout-ms 3000
--mqtt on/off
--mqtt-host 124.220.49.93
--mqtt-port 1883
--mqtt-username rk3588
--mqtt-password Rk3588_Mqtt_123
--mqtt-client-id rk3588-001
--mqtt-topic devices/rk3588-001/plate/events
--mqtt-heartbeat-topic devices/rk3588-001/status/heartbeat
--mqtt-error-topic devices/rk3588-001/status/error
--mqtt-heartbeat-interval-sec 30
--mqtt-timeout-ms 3000
```

### 验收步骤

1. 先验证 HTTP 图片服务：

```bash
curl -F "file=@test.jpg" http://124.220.49.93:8000/upload
```

2. 再打开 MQTT 订阅：

```bash
mosquitto_sub -h 124.220.49.93 -p 1883 \
  -u rk3588 -P 'Rk3588_Mqtt_123' \
  -t 'devices/rk3588-001/#' -v
```

3. 启动开发板程序，观察日志：

```text
[UPLOAD_EVENT] ...
[UPLOAD_THREAD] ... image_path=... image_url=...
[PERF] upload_event_thread ... http_uploaded=... mqtt_published=...
```

4. 打开 MQTT 收到的 `image_url`，确认图片可访问。

### 断网验证

断网或服务器不可达时，预期行为是：

```text
检测、OCR、OSD、视频保存继续运行
上传线程统计 http_failed 或 mqtt_failed 增加
上传队列满时 dropped 增加
程序不退出，主链路不被网络拖慢
网络恢复后，后续新事件继续上传和发布
```

建议按顺序验证：

```text
1. 正常联网运行，确认 http_uploaded 和 mqtt_published 增加
2. 临时关闭服务器 8000 端口，确认 http_failed 增加但视频继续保存
3. 临时关闭 MQTT 1883 端口，确认 mqtt_failed 增加但视频继续保存
4. 恢复服务，制造新车牌事件，确认后续事件能继续上报
```

当前版本不做本地失败缓存；断网期间失败的事件会丢弃，只保证主链路稳定和网络恢复后的新事件继续上报。

## 模型文件

当前项目包含两个模型目录：

```text
models/car/
models/car-v8/
```

默认建议使用：

```text
models/car-v8/v8-car-relu-3588.rknn
```

## 目录说明

```text
include/                 C++ 头文件
src/                     C++ 源码
models/                  RKNN 模型和标签文件
video/test-video/        Demo 输入视频
video/output-video/      运行输出视频，默认忽略
third_party/rockchip/    项目内置 Rockchip MPP/RGA/RKNN 依赖
build/                   CMake 构建目录，默认忽略
```

## 常见问题

### 1. 找不到 FFmpeg 开发库

如果 CMake 报错找不到 `libavformat`、`libavcodec` 或 `libavutil`，先检查：

```bash
pkg-config --modversion libavformat libavcodec libavutil
```

如果该命令失败，需要安装对应的 FFmpeg development packages。

### 2. 链接或运行时找不到 Rockchip 库

默认情况下，构建出的二进制会携带 `RUNPATH`，优先从项目内加载：

```text
third_party/rockchip/mpp/Linux/aarch64
third_party/rockchip/rga/RK3588/lib/Linux/aarch64
third_party/rockchip/rknn/RK3588/Linux/librknn_api/aarch64
```

可以用下面命令确认：

```bash
ldd ./build/rk_mp4_yolo_stage5 | grep -E 'rockchip_mpp|rga|rknn'
```

期望看到路径指向项目内的 `third_party/rockchip`。

### 3. 不能在非 RK3588 机器上运行

这是预期行为。本项目依赖 RK3588 板端的 MPP、RGA、RKNN Runtime 和对应驱动环境。

### 4. 当前项目还在完善，理论上摄像头识别更快，他可能不需要解码 ，因为有的直接输出V4L2数据

## 车牌白名单继电器

继电器功能默认关闭。启用后，OCR 识别到白名单车牌并满足置信度门槛时，第 1 路继电器会吸合 2 秒后强制释放。

硬件参数已经验证：`/dev/ttyS0`、`9600 8N1`、Modbus 地址 `1`、线圈地址 `0`。第 1 路继电器的 `COM0/NO0` 当前控制 3.3V LED。

白名单文件是 `config/plate_whitelist.txt`，每行写一个车牌；空行和以 `#` 开头的注释行会忽略。例如：

```text
京A12345
粤B12345D
```

启用示例：

```bash
./build/rk_mp4_yolo_stage5 \
  --input mp4 \
  --video ./video/test-video/5s.mp4 \
  --model ./models/car-v8/v8-car-relu-3588.rknn \
  --output ./video/output-video/relay-test.mp4 \
  --ocr-model ./ppocrv5/PP-OCRv5_mobile_rec_license_plate.rknn \
  --ocr-vocab ./ppocrv5/model/license_plate_dict.txt \
  --relay-enabled on \
  --relay-whitelist ./config/plate_whitelist.txt
```

默认安全策略：检测分数至少 `0.50`、OCR 分数至少 `0.90`、同一车牌 `30` 秒内只允许触发一次。继电器吸合时不接受新的触发请求，也不会延长当前 2 秒脉冲。程序启动、退出、串口错误时都会发送关闭第 1 路的命令。

可选参数：

```text
--relay-device /dev/ttyS0
--relay-baud 9600
--relay-address 1
--relay-channel 0
--relay-pulse-ms 2000
--relay-plate-cooldown-sec 30
--relay-min-detect-score 0.50
--relay-min-plate-score 0.90
--relay-timeout-ms 500
--relay-verbose
```

### 5. 白名单配置错误

如果启用继电器后白名单文件不存在或为空，程序会记录 `[RELAY] whitelist gate disabled`，不会触发继电器；视频识别仍会继续运行。


## 开源注意事项

`third_party/rockchip` 中包含 Rockchip 相关头文件和动态库。公开分发前请确认这些二进制和头文件的授权条款是否允许随仓库发布。