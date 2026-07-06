# MQTT 远程车牌上报操作说明

本文档只说明本项目中 MQTT 相关的部署、配置、测试和手机接收操作。

当前远程上报链路是：

```text
RK3588 开发板
  -> 检测 / ByteTrack / OCR
  -> 截图 JPEG
  -> HTTP 上传图片
  -> MQTT 发布事件 JSON
  -> 服务器 Mosquitto Broker
  -> 手机 / 外网电脑 / 后台订阅
```

核心原则：

```text
MQTT 只传事件 JSON 和 image_url
MQTT 不传图片二进制
图片通过 HTTP URL 打开
网络发送只发生在上传线程，不阻塞检测主链路
```

## 1. 当前使用的 MQTT 参数

```text
服务器 IP: 124.220.49.93
MQTT 端口: 1883
用户名: rk3588
密码: Rk3588_Mqtt_123
设备 ID: rk3588-001
```

开发板发布 topic：

```text
devices/rk3588-001/plate/events
devices/rk3588-001/status/heartbeat
devices/rk3588-001/status/error
```

手机或外网设备推荐订阅：

```text
devices/rk3588-001/#
```

这个通配 topic 会同时收到：

```text
车牌事件
心跳状态
错误状态
```

## 2. Topic 说明

### 2.1 车牌事件

```text
devices/rk3588-001/plate/events
```

开发板识别到符合阈值和去重规则的车牌后，会发布一条事件 JSON。

示例：

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

重点字段：

```text
plate_text    车牌号
plate_score   OCR 置信度
detect_score  检测置信度
image_url     截图访问地址
track_id      ByteTrack 目标 ID
```

### 2.2 心跳状态

```text
devices/rk3588-001/status/heartbeat
```

用于确认上传线程还在工作。

示例：

```json
{
  "device_id": "rk3588-001",
  "type": "heartbeat",
  "timestamp_ms": 1720000000000,
  "consumed": 10,
  "snapshot_saved": 10,
  "snapshot_failed": 0,
  "http_uploaded": 10,
  "http_failed": 0,
  "mqtt_published": 10,
  "mqtt_failed": 0,
  "queue_pushed": 10,
  "queue_dropped": 0,
  "detail": "ok"
}
```

### 2.3 错误状态

```text
devices/rk3588-001/status/error
```

截图失败或 HTTP 上传失败时会发布错误状态。

示例：

```json
{
  "device_id": "rk3588-001",
  "type": "error",
  "timestamp_ms": 1720000000000,
  "consumed": 12,
  "snapshot_failed": 1,
  "http_failed": 0,
  "mqtt_failed": 0,
  "queue_dropped": 0,
  "detail": "snapshot_failed"
}
```

可能的 `detail`：

```text
snapshot_failed
http_upload_failed
```

注意：如果 MQTT 自己断开，错误 topic 也发不出去。这种情况只会在开发板本地统计 `mqtt_failed`。

## 3. 服务器端检查

### 3.1 检查 MQTT 端口是否监听

在服务器上执行：

```bash
sudo ss -lntp | grep 1883
```

期望看到类似：

```text
0.0.0.0:1883
```

或：

```text
:::1883
```

如果只看到：

```text
127.0.0.1:1883
```

说明 Mosquitto 只监听本机，手机和外网设备无法连接。需要修改 Mosquitto 配置。

### 3.2 Mosquitto 配置参考

```conf
listener 1883 0.0.0.0
allow_anonymous false
password_file /mosquitto/config/passwd
```

修改后重启 Mosquitto。

如果你使用 Docker 部署，确认容器端口映射包含：

```text
1883:1883
```

### 3.3 腾讯云安全组

需要放行：

```text
TCP 1883  MQTT 订阅和发布
TCP 8000  开发板 HTTP 上传图片
TCP 80    手机或浏览器访问图片
```

## 4. 服务器命令行测试

### 4.1 订阅全部设备消息

```bash
mosquitto_sub -h 124.220.49.93 -p 1883 \
  -u rk3588 -P 'Rk3588_Mqtt_123' \
  -t 'devices/rk3588-001/#' -v
```

### 4.2 手动发布测试消息

另开一个终端执行：

```bash
mosquitto_pub -h 124.220.49.93 -p 1883 \
  -u rk3588 -P 'Rk3588_Mqtt_123' \
  -t 'devices/rk3588-001/plate/events' \
  -m '{"plate_text":"TEST123","image_url":"http://124.220.49.93/images/test.jpg"}'
```

订阅端应看到：

```text
devices/rk3588-001/plate/events {"plate_text":"TEST123","image_url":"http://124.220.49.93/images/test.jpg"}
```

如果服务器能收到，说明 Broker 内部和账号密码正常。

## 5. 外网电脑测试

在任意外网电脑安装客户端：

```bash
sudo apt install -y mosquitto-clients
```

订阅：

```bash
mosquitto_sub -h 124.220.49.93 -p 1883 \
  -u rk3588 -P 'Rk3588_Mqtt_123' \
  -t 'devices/rk3588-001/#' -v
```

如果外网电脑能收到，说明公网 MQTT 通路正常，手机也应该能收到。

## 6. 手机 IoT MQTT Panel 操作

### 6.1 连接配置

在 IoT MQTT Panel 中创建连接：

```text
Connection name: rk3588-cloud
Client ID: phone-test
Broker address: 124.220.49.93
Port: 1883
Network protocol: TCP
Username: rk3588
Password: Rk3588_Mqtt_123
Auto connect: 开启
Clean session: 开启
TLS/SSL: 关闭
```

连接成功后，说明手机已经能连上公网 MQTT Broker。

### 6.2 创建接收面板

进入 Dashboard 后：

```text
1. 点击右下角 +
2. 新建 Log / Text / Display / Subscriber 类型组件
3. Topic 填 devices/rk3588-001/#
4. 保存
```

如果 App 不支持 `#` 通配符，就分别创建三个组件：

```text
devices/rk3588-001/plate/events
devices/rk3588-001/status/heartbeat
devices/rk3588-001/status/error
```

推荐先用 Log 组件显示完整 JSON。确认链路通了以后，再拆成单独字段显示。

### 6.3 手机接收测试

在服务器发布一条测试消息：

```bash
mosquitto_pub -h 124.220.49.93 -p 1883 \
  -u rk3588 -P 'Rk3588_Mqtt_123' \
  -t 'devices/rk3588-001/plate/events' \
  -m '{"plate_text":"TEST123","image_url":"http://124.220.49.93/images/test.jpg"}'
```

手机面板应显示这条 JSON。

如果手机能看到这条消息，说明：

```text
服务器 MQTT -> 手机
```

已经通了。

## 7. 开发板运行参数

### 7.1 MP4 测试命令

建议先用 MP4 测，变量少，方便定位问题。

```bash
./build/rk_mp4_yolo_stage5 \
  --input mp4 \
  --video ./video/test-video/9s.mp4 \
  --model ./models/car-v8/v8-car-relu-3588.rknn \
  --output ./video/output-video/remote-report-test.mp4 \
  --ocr-model ./ppocrv5/PP-OCRv5_mobile_rec_license_plate.rknn \
  --ocr-vocab ./ppocrv5/model/license_plate_dict.txt \
  --upload-event-log on \
  --upload-min-detect-score 0.50 \
  --upload-min-plate-score 0.80 \
  --upload-track-cooldown-frames 30 \
  --upload-plate-cooldown-frames 300 \
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
  --mqtt-error-topic devices/rk3588-001/status/error \
  --mqtt-heartbeat-interval-sec 10
```

### 7.2 摄像头运行命令

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

## 8. 开发板参数说明

```text
--mqtt on/off                            是否启用 MQTT 发布
--mqtt-host 124.220.49.93                MQTT Broker 地址
--mqtt-port 1883                         MQTT Broker 端口
--mqtt-username rk3588                   MQTT 用户名
--mqtt-password Rk3588_Mqtt_123          MQTT 密码
--mqtt-client-id rk3588-001              MQTT client id
--mqtt-topic devices/rk3588-001/plate/events
--mqtt-heartbeat-topic devices/rk3588-001/status/heartbeat
--mqtt-error-topic devices/rk3588-001/status/error
--mqtt-heartbeat-interval-sec 30         心跳间隔，单位秒
--mqtt-timeout-ms 3000                   MQTT socket 超时
```

注意：

```text
--upload-event-log on 必须打开，否则不会创建上传线程，也不会发 MQTT
--upload-url 必须正确，否则 image_url 可能无法形成完整闭环
```

## 9. 图片 URL 验证

MQTT 事件中的 `image_url` 应能在手机浏览器打开。

命令行验证：

```bash
curl -I http://124.220.49.93/images/test.jpg
```

期望：

```text
HTTP/1.1 200 OK
```

如果 MQTT 能收到但图片打不开，问题不在 MQTT，而在 HTTP 图片服务或 Nginx 静态访问路径。

## 10. 常见问题

### 10.1 手机显示 connected，但没有消息

按顺序检查：

```text
1. 手机订阅 topic 是否是 devices/rk3588-001/#
2. 服务器 mosquitto_pub 手动测试手机是否能收到
3. 开发板是否真的产生 [UPLOAD_EVENT]
4. 开发板日志中 mqtt_published 是否增加
5. 是否订错设备 ID，例如 rk3588-001 写成 rk3588_001
```

### 10.2 服务器能收到，手机收不到

通常是手机面板没有订阅 topic，或者订阅组件没有保存。

建议先用 Log 类型组件订阅：

```text
devices/rk3588-001/#
```

### 10.3 手机连接失败

根据错误判断：

```text
Connection timeout     腾讯云安全组或服务器防火墙没放行 1883
Connection refused     Mosquitto 没监听公网，或端口映射错误
Bad username/password  用户名密码错误
Not authorized         Mosquitto ACL 权限限制
```

### 10.4 开发板提示 mqtt_failed 增加

说明开发板连不上 MQTT Broker 或认证失败。

检查：

```text
--mqtt-host
--mqtt-port
--mqtt-username
--mqtt-password
腾讯云安全组 1883
Mosquitto 用户密码
```

### 10.5 有心跳但没有车牌事件

说明 MQTT 通了，但检测事件没有产生。

检查：

```text
是否有车牌目标
OCR 是否识别成功
--upload-min-detect-score 是否太高
--upload-min-plate-score 是否太高
冷却参数是否太长
```

临时降低阈值：

```bash
--upload-min-detect-score 0.30 \
--upload-min-plate-score 0.50 \
--upload-track-cooldown-frames 30 \
--upload-plate-cooldown-frames 300
```

## 11. 当前方案边界

当前手机端是直接通过 MQTT App 订阅 Broker。

优点：

```text
简单
实时
不需要额外后台
适合当前验证
```

限制：

```text
手机需要打开 MQTT App 才方便查看
没有系统级推送通知
1883 是明文 MQTT，公网暴露有安全风险
消息不落库，错过就看不到历史
```

如果后续要做正式使用，建议升级为：

```text
MQTT Broker
  -> 服务器后台订阅服务
  -> 数据库存储事件
  -> Web 页面查看图片和车牌
  -> 手机浏览器或 App 推送通知
```

这样手机不需要一直开 MQTT App，也能看历史记录。