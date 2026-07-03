// Licensed under the Apache License, Version 2.0

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "ppocrv5.h"

std::vector<std::string> ocr_vocab;

static bool load_vocab(const char* vocab_path)
{
    std::ifstream file(vocab_path);
    if (!file.is_open()) {
        printf("failed to open vocab: %s\n", vocab_path);
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!line.empty()) {
            ocr_vocab.push_back(line);
        }
    }

    if (ocr_vocab.empty()) {
        printf("empty vocab: %s\n", vocab_path);
        return false;
    }
    return true;
}

static cv::Mat resize_with_padding(const cv::Mat& bgr_image, int target_h, int target_w, int* resized_w_out)
{
    const float ratio = bgr_image.cols / std::max(1.0f, static_cast<float>(bgr_image.rows));
    const int resized_w = std::min(target_w, std::max(1, static_cast<int>(std::ceil(target_h * ratio))));

    cv::Mat resized;
    cv::resize(bgr_image, resized, cv::Size(resized_w, target_h), 0, 0, cv::INTER_LINEAR);
    resized.convertTo(resized, CV_32FC3);
    resized = (resized - 127.5f) / 127.5f;

    cv::Mat padded = cv::Mat::zeros(target_h, target_w, CV_32FC3);
    resized.copyTo(padded(cv::Rect(0, 0, resized_w, target_h)));

    if (resized_w_out != nullptr) {
        *resized_w_out = resized_w;
    }
    return padded;
}

static int get_output_sequence_len(const rknn_app_context_t* app_ctx)
{
    const rknn_tensor_attr& attr = app_ctx->output_attrs[0];
    for (int i = 0; i < attr.n_dims; ++i) {
        if (attr.dims[i] == 40 || attr.dims[i] == app_ctx->model_width / 8) {
            return attr.dims[i];
        }
    }
    return app_ctx->model_width / 8;
}

static int get_output_classes(const rknn_app_context_t* app_ctx)
{
    const rknn_tensor_attr& attr = app_ctx->output_attrs[0];
    for (int i = attr.n_dims - 1; i >= 0; --i) {
        if (attr.dims[i] > 1 && attr.dims[i] != get_output_sequence_len(app_ctx)) {
            return attr.dims[i];
        }
    }
    return static_cast<int>(ocr_vocab.size()) + 1;
}

static ppocr_rec_result ctc_decode(const float* out_data, int seq_len, int class_num)
{
    ppocr_rec_result result;
    memset(&result, 0, sizeof(result));

    std::string text;
    float score_sum = 0.0f;
    int selected_count = 0;
    int last_index = -1;

    for (int n = 0; n < seq_len; ++n) {
        const float* row = out_data + n * class_num;
        int argmax_idx = 0;
        float max_value = row[0];
        for (int j = 1; j < class_num; ++j) {
            if (row[j] > max_value) {
                max_value = row[j];
                argmax_idx = j;
            }
        }

        if (argmax_idx != 0 && argmax_idx != last_index) {
            const int char_index = argmax_idx - 1;
            if (char_index >= 0 && char_index < static_cast<int>(ocr_vocab.size())) {
                text += ocr_vocab[char_index];
                score_sum += max_value;
                selected_count += 1;
            }
        }
        last_index = argmax_idx;
    }

    result.str_size = selected_count;
    result.score = selected_count > 0 ? score_sum / selected_count : 0.0f;
    snprintf(result.str, sizeof(result.str), "%s", text.c_str());
    return result;
}

static int inference_license_plate_rec(rknn_app_context_t* app_ctx, const cv::Mat& bgr_image, ppocr_rec_result* result)
{
    const int input_h = app_ctx->model_height > 0 ? app_ctx->model_height : IMAGE_HEIGHT;
    const int input_w = app_ctx->model_width > 0 ? app_ctx->model_width : 320;

    int resized_w = 0;
    cv::Mat input_image = resize_with_padding(bgr_image, input_h, input_w, &resized_w);

    int ret;

    rknn_input input;
    memset(&input, 0, sizeof(input));
    input.index = 0;
    input.type = RKNN_TENSOR_FLOAT32;
    input.fmt = RKNN_TENSOR_NHWC;
    input.size = input_w * input_h * app_ctx->model_channel * sizeof(float);
    input.buf = input_image.data;

    ret = rknn_inputs_set(app_ctx->rknn_ctx, 1, &input);
    if (ret < 0) {
        printf("rknn_inputs_set fail! ret=%d\n", ret);
        return ret;
    }

    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    if (ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return ret;
    }

    rknn_output output;
    memset(&output, 0, sizeof(output));
    output.want_float = 1;
    ret = rknn_outputs_get(app_ctx->rknn_ctx, 1, &output, NULL);
    if (ret < 0) {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        return ret;
    }

    const int seq_len = get_output_sequence_len(app_ctx);
    const int class_num = get_output_classes(app_ctx);
    *result = ctc_decode(static_cast<float*>(output.buf), seq_len, class_num);

    printf("preprocess: original=%dx%d, resized=%dx%d, padded=%dx%d\n",
           bgr_image.cols, bgr_image.rows, resized_w, input_h, input_w, input_h);
    printf("decode: seq_len=%d, class_num=%d, vocab_size=%zu\n", seq_len, class_num, ocr_vocab.size());

    rknn_outputs_release(app_ctx->rknn_ctx, 1, &output);
    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 4) {
        printf("Usage: %s <rec_model_path> <vocab_path> <plate_image_path>\n", argv[0]);
        printf("e.g.  : %s ./model/PP-OCRv5_mobile_rec_license_plate.rknn ./model/license_plate_dict.txt ./model/image.jpg\n", argv[0]);
        return -1;
    }

    const char* model_path = argv[1];
    const char* vocab_path = argv[2];
    const char* image_path = argv[3];

    if (!load_vocab(vocab_path)) {
        return -1;
    }

    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        printf("cv::imread %s fail!\n", image_path);
        return -1;
    }

    rknn_app_context_t rec_ctx;
    memset(&rec_ctx, 0, sizeof(rec_ctx));

    int ret = init_ppocr_rec_model(model_path, &rec_ctx);
    if (ret != 0) {
        printf("init_ppocr_rec_model fail! ret=%d model_path=%s\n", ret, model_path);
        return -1;
    }

    TIMER timer;
    ppocr_rec_result result;
    timer.tik();
    ret = inference_license_plate_rec(&rec_ctx, image, &result);
    timer.tok();
    timer.print_time("license_plate_rec");

    if (ret == 0) {
        printf("Text : %s\n", result.str);
        printf("Score: %.6f\n", result.score);
    }

    release_ppocr_model(&rec_ctx);
    return ret;
}