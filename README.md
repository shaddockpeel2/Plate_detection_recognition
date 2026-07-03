# RK3588 MP4 YOLO Pipeline

这是一个面向 RK3588 / aarch64 Linux 的 C++ 视频推理流水线示例。项目从 MP4 文件读取视频帧，经过 Rockchip MPP 硬解、RGA 预处理、RKNN 推理、后处理/OSD，再通过 MPP 编码输出 MP4。

## 功能链路

```text
MP4 输入
  -> FFmpeg 解封装
  -> Rockchip MPP 硬解
  -> RGA 图像预处理
  -> RKNN Runtime 推理
  -> 后处理 / OSD
  -> Rockchip MPP 硬编码
  -> MP4 输出
```

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

## 运行 Demo

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

## 开源注意事项

`third_party/rockchip` 中包含 Rockchip 相关头文件和动态库。公开分发前请确认这些二进制和头文件的授权条款是否允许随仓库发布。