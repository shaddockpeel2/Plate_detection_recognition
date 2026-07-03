import argparse
import time
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RKNN_PATH = PROJECT_ROOT / "PP-OCRv5_mobile_rec_license_plate.rknn"
DEFAULT_IMAGE_PATH = PROJECT_ROOT / "image.jpg"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "PP-OCRv5_mobile_rec_license_plate_onnx" / "inference.yml"


def load_charset(config_path):
    chars = []
    in_dict = False
    with open(str(config_path), "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped == "character_dict:":
                in_dict = True
                continue
            if in_dict:
                if stripped.startswith("- "):
                    value = stripped[2:].strip()
                    value = value.strip("'\"")
                    chars.append(value)
                elif stripped and not line.startswith("  "):
                    break
    if not chars:
        raise RuntimeError("Cannot load character_dict from {}".format(config_path))
    return chars


def resize_with_padding(image_path, height, width):
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError("Cannot read image: {}".format(image_path))

    src_h, src_w = image.shape[:2]
    ratio = float(src_w) / max(float(src_h), 1.0)
    resized_w = min(width, max(1, int(np.ceil(height * ratio))))

    image = cv2.resize(image, (resized_w, height), interpolation=cv2.INTER_LINEAR)
    image = image.astype("float32") / 255.0
    image = (image - 0.5) / 0.5

    padded = np.zeros((height, width, 3), dtype="float32")
    padded[:, :resized_w, :] = image
    return np.expand_dims(padded, axis=0)


def to_runtime_layout(input_tensor, input_layout):
    if input_layout == "nhwc":
        return input_tensor, "nhwc"
    return input_tensor.transpose(0, 3, 1, 2), None


def normalize_probs(sequence):
    row_sums = np.sum(sequence, axis=-1)
    is_probability = (
        np.min(sequence) >= 0
        and np.max(sequence) <= 1
        and np.allclose(row_sums, 1.0, atol=1e-3)
    )
    if is_probability:
        return sequence

    logits = sequence - np.max(sequence, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def ctc_decode(output, charset):
    output = np.asarray(output)
    if output.ndim == 2:
        output = np.expand_dims(output, axis=0)

    probs = normalize_probs(output[0])
    indexes = np.argmax(probs, axis=-1)
    scores = np.max(probs, axis=-1)

    text = []
    selected_scores = []
    previous = None
    for index, score in zip(indexes, scores):
        index = int(index)
        if index != 0 and index != previous:
            char_index = index - 1
            if 0 <= char_index < len(charset):
                text.append(charset[char_index])
                selected_scores.append(float(score))
        previous = index

    confidence = float(np.mean(selected_scores)) if selected_scores else 0.0
    return "".join(text), confidence


def create_runtime(args):
    if args.runtime == "toolkit":
        from rknn.api import RKNN

        rknn = RKNN(verbose=args.verbose)
        ret = rknn.load_rknn(path=str(args.model))
        if ret != 0:
            raise RuntimeError("Load RKNN failed, ret={}".format(ret))

        ret = rknn.init_runtime(target=args.target, device_id=args.device_id)
        if ret != 0:
            raise RuntimeError("Init RKNN runtime failed, ret={}".format(ret))
        return rknn

    from rknnlite.api.rknn_lite import RKNNLite

    rknn = RKNNLite(verbose=args.verbose)
    ret = rknn.load_rknn(str(args.model))
    if ret != 0:
        raise RuntimeError("Load RKNNLite failed, ret={}".format(ret))

    if args.target in ["rk3576", "rk3588"]:
        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    else:
        ret = rknn.init_runtime()
    if ret != 0:
        raise RuntimeError("Init RKNNLite runtime failed, ret={}".format(ret))
    return rknn


def run_once(rknn, input_tensor, data_format):
    if data_format is None:
        outputs = rknn.inference(inputs=[input_tensor])
    else:
        outputs = rknn.inference(inputs=[input_tensor], data_format=[data_format])
    if outputs is None or len(outputs) == 0:
        raise RuntimeError("RKNN inference returned empty outputs")
    return outputs[0]


def benchmark(rknn, input_tensor, data_format, warmup, loops):
    for _ in range(warmup):
        run_once(rknn, input_tensor, data_format)

    output = None
    costs = []
    for _ in range(loops):
        start = time.perf_counter()
        output = run_once(rknn, input_tensor, data_format)
        costs.append((time.perf_counter() - start) * 1000)

    if output is None:
        output = run_once(rknn, input_tensor, data_format)
    return output, costs


def parse_args():
    parser = argparse.ArgumentParser(description="Test PP-OCRv5 license plate rec RKNN inference.")
    parser.add_argument("--model", type=Path, default=DEFAULT_RKNN_PATH, help="RKNN model path.")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE_PATH, help="Cropped license plate image path.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="inference.yml path.")
    parser.add_argument("--target", default="rk3588", help="rk3562/rk3566/rk3568/rk3576/rk3588.")
    parser.add_argument("--runtime", default="lite", choices=["lite", "toolkit"], help="lite for board, toolkit for PC simulator/device.")
    parser.add_argument("--device-id", default=None, help="Toolkit runtime device id, e.g. 192.168.1.10:5555.")
    parser.add_argument("--width", type=int, default=320, help="Input width used when converting RKNN.")
    parser.add_argument("--height", type=int, default=48, help="Input height used when converting RKNN.")
    parser.add_argument("--input-layout", default="nhwc", choices=["nhwc", "nchw"], help="Runtime input layout. RKNN Runtime usually expects nhwc input even if model layout is NCHW.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup inference count.")
    parser.add_argument("--loops", type=int, default=50, help="Benchmark inference count.")
    parser.add_argument("--verbose", action="store_true", help="Enable RKNN verbose log.")
    return parser.parse_args()


def print_latency(costs):
    if not costs:
        return
    costs_np = np.asarray(costs, dtype=np.float32)
    print(
        "Latency ms: avg={:.3f}, min={:.3f}, max={:.3f}, p95={:.3f}".format(
            costs_np.mean(),
            costs_np.min(),
            costs_np.max(),
            np.percentile(costs_np, 95),
        )
    )


def main():
    args = parse_args()
    if not args.model.exists():
        raise RuntimeError("RKNN model does not exist: {}".format(args.model))
    if not args.config.exists():
        raise RuntimeError("Config does not exist: {}".format(args.config))

    charset = load_charset(args.config)
    input_tensor_nhwc = resize_with_padding(args.image, args.height, args.width)
    input_tensor, data_format = to_runtime_layout(input_tensor_nhwc, args.input_layout)

    load_start = time.perf_counter()
    rknn = create_runtime(args)
    load_cost = (time.perf_counter() - load_start) * 1000

    output, costs = benchmark(rknn, input_tensor, data_format, args.warmup, args.loops)
    text, confidence = ctc_decode(output, charset)

    print("RKNN loaded: {}".format(args.model))
    print("Runtime: {}".format(args.runtime))
    print("Runtime input shape: {}".format(input_tensor.shape))
    print("Runtime data_format: {}".format(data_format))
    print("Model load ms: {:.3f}".format(load_cost))
    print("Output shape: {}".format(np.asarray(output).shape))
    print_latency(costs)
    print("Image : {}".format(args.image))
    print("Text  : {}".format(text))
    print("Score : {:.4f}".format(confidence))

    rknn.release()


if __name__ == "__main__":
    main()