cmake_minimum_required(VERSION 3.10)
project(rk_mp4_yolo_stage5 CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(USE_VENDOR_RK_LIBS "Link Rockchip vendor MPP/RGA/RKNN libraries from this repository" ON)
set(RKNPU2_DIR "" CACHE PATH "Optional external rknpu2 SDK path")

set(ROCKCHIP_VENDOR_DIR ${CMAKE_CURRENT_LIST_DIR}/third_party/rockchip)
if(RKNPU2_DIR)
  set(MPP_DIR ${RKNPU2_DIR}/3rdparty/mpp)
  set(RGA_DIR ${RKNPU2_DIR}/3rdparty/rga/RK3588)
  set(RKNN_RUNTIME_DIR ${RKNPU2_DIR}/runtime/RK3588/Linux/librknn_api)
else()
  set(MPP_DIR ${ROCKCHIP_VENDOR_DIR}/mpp)
  set(RGA_DIR ${ROCKCHIP_VENDOR_DIR}/rga/RK3588)
  set(RKNN_RUNTIME_DIR ${ROCKCHIP_VENDOR_DIR}/rknn/RK3588/Linux/librknn_api)
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED libavformat libavcodec libavutil)

add_executable(rk_mp4_yolo_stage5
  src/main.cpp
  src/decoder_thread.cpp
  src/rknn_input_runtime.cpp
  src/preprocess_thread.cpp
  src/inference_thread.cpp
  src/postprocess_osd_thread.cpp
  src/encoder_writer_thread.cpp
)

target_include_directories(rk_mp4_yolo_stage5 PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/include
  ${MPP_DIR}/include
  ${RGA_DIR}/include
  ${RKNN_RUNTIME_DIR}/include
  ${FFMPEG_INCLUDE_DIRS}
)

if(USE_VENDOR_RK_LIBS)
  target_link_directories(rk_mp4_yolo_stage5 PRIVATE
    ${MPP_DIR}/Linux/aarch64
    ${RGA_DIR}/lib/Linux/aarch64
    ${RKNN_RUNTIME_DIR}/aarch64
  )
  set_target_properties(rk_mp4_yolo_stage5 PROPERTIES
    BUILD_RPATH "${MPP_DIR}/Linux/aarch64;${RGA_DIR}/lib/Linux/aarch64;${RKNN_RUNTIME_DIR}/aarch64"
  )
endif()

target_link_libraries(rk_mp4_yolo_stage5 PRIVATE
  ${FFMPEG_LIBRARIES}
  rockchip_mpp
  rga
  rknn_api
  pthread
  dl
)

target_compile_options(rk_mp4_yolo_stage5 PRIVATE
  ${FFMPEG_CFLAGS_OTHER}
  -Wall
  -Wextra
)